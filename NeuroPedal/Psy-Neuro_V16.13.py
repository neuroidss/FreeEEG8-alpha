import sys, serial, threading, random, time, torch, math, contextlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from collections import deque
from evdev import InputDevice, list_devices, ecodes

# --- КОНФИГУРАЦИЯ ---
FS = 44100
EEG_FS = 250.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCTAVES = 4

N_MOTOR = 21   # Параметры геймпада
N_AUDIO = 10   # Огибающие звука (струны, бочка и т.д.)
N_EEG = 8      # Каналы ЭЭГ
N_THALAMUS = 2 # Таламус: sin и cos фазы такта (Reference Frame)

CONFIG = {
    "EEG_WIN": 250,      
    "TRAJ_LEN": 60,      # 1 секунда паттернов
    "LR": 5e-4,
    "REPLAY_SIZE": 15000,
    "MAX_BATCH": 128     
}

dyn_batch = 4        
dyn_sleep = 0.05      
audio_underruns = 0   

if torch.cuda.is_available():
    audio_stream = torch.cuda.Stream(priority=-1) 
    train_stream = torch.cuda.Stream(priority=0)  
else:
    audio_stream = train_stream = None

# ==========================================
# 1. THOUSAND BRAINS ARCHITECTURE (Numenta Style)
# ==========================================
class CorticalColumn(nn.Module):
    """Сенсорная или моторная колонка. Извлекает признаки из своей модальности."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=5, padding=2), nn.ELU(),
            nn.Conv1d(32, out_channels, kernel_size=5, padding=2), nn.ELU()
        )
    def forward(self, x):
        # x: (Batch, Channels, Time) -> (Batch, out_channels, Time)
        return self.net(x)

class ThousandBrainsArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        
        # 1. ИЗОЛИРОВАННЫЕ КОЛОНКИ (Сенсоры)
        self.col_eeg = CorticalColumn(N_EEG, 32)
        self.col_motor = CorticalColumn(N_MOTOR, 32)
        self.col_audio = CorticalColumn(N_AUDIO, 32)
        
        # 4-я ЗОНА: ТАЛАМУС (Reference Frame / Временная сетка)
        self.col_thalamus = CorticalColumn(N_THALAMUS, 16)
        
        # 2. АССОЦИАТИВНАЯ ЗОНА (Thinking Area / Voting)
        # Объединяет выходы всех колонок для формирования единого консенсуса
        self.association_area = nn.GRU(32 + 32 + 32 + 16, 256, num_layers=2, batch_first=True)
        
        # 3. МОТОРНЫЕ И АУДИО ВЫХОДЫ (Декодеры предсказаний)
        self.pred_motor = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Linear(128, N_MOTOR), nn.Sigmoid())
        self.pred_audio = nn.Sequential(nn.Linear(256, 128), nn.ELU(), nn.Linear(128, N_AUDIO), nn.Sigmoid())

    def forward(self, eeg, motor, audio, thalamus):
        # Перевод в (Batch, Channels, Time)
        e = eeg.permute(0, 2, 1)     
        m = motor.permute(0, 2, 1)   
        a = audio.permute(0, 2, 1)   
        t = thalamus.permute(0, 2, 1)
        
        # Колонки работают независимо
        feat_e = self.col_eeg(e)     
        feat_m = self.col_motor(m)   
        feat_a = self.col_audio(a)   
        feat_t = self.col_thalamus(t)
        
        # ЭЭГ имеет другую частоту (250), сжимаем до длины паттерна (60)
        feat_e = nn.functional.adaptive_avg_pool1d(feat_e, CONFIG["TRAJ_LEN"])
        
        # Ассоциативная зона (Слияние голосов колонок)
        context = torch.cat([feat_e, feat_m, feat_a, feat_t], dim=1).permute(0, 2, 1)
        out, _ = self.association_area(context)
        
        # Текущий консенсус мозга о том, что должно произойти дальше
        state = out[:, -1, :] 
        
        nxt_motor = self.pred_motor(state)
        nxt_audio = self.pred_audio(state)
        
        return nxt_motor, nxt_audio

model = ThousandBrainsArchitecture().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

# ==========================================
# 2. РАЗДЕЛЬНЫЕ ВСЕЛЕННЫЕ (STRICT ISOLATION)
# ==========================================
# Общее для обеих вселенных
eeg_hist = deque([np.zeros(N_EEG)] * CONFIG["EEG_WIN"], maxlen=CONFIG["EEG_WIN"])
thalamus_clock = 0.0

# ВСЕЛЕННАЯ 1: Учитель (Реальный геймпад + Звук правого уха)
real_motor_hist = deque([np.zeros(N_MOTOR)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
real_audio_hist = deque([np.zeros(N_AUDIO)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
params_real = torch.zeros(N_MOTOR, device=DEVICE) 
params_real[19] = 0.2857 # Tempo

# ВСЕЛЕННАЯ 2: Исполнитель (Виртуальный геймпад + Звук левого уха)
virt_motor_hist = deque([np.zeros(N_MOTOR)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
virt_audio_hist = deque([np.zeros(N_AUDIO)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
params_virt = torch.zeros(N_MOTOR, device=DEVICE) 
params_virt[19] = 0.2857 # Tempo

replay_buf = deque(maxlen=CONFIG["REPLAY_SIZE"])
prev_real_state = None

rhythm_locked = True
last_strum_time = time.time()
strum_history = deque([0.5, 0.5], maxlen=4)

# --- GPU СИНТЕЗАТОР (V16) ---
class GPU_Synth:
    def __init__(self):
        self.phases = torch.zeros((6, OCTAVES), device=DEVICE)
        self.sub_phases = torch.zeros(OCTAVES, device=DEVICE)
        self.zap_phases = torch.zeros(OCTAVES, device=DEVICE)
        self.kick_phases = torch.zeros(OCTAVES, device=DEVICE)
        self.amps = torch.zeros(6, device=DEVICE)
        self.delay_buf = torch.zeros(FS * 3, device=DEVICE)
        self.ptr = 0
        self.ratios = torch.tensor([1.0, 1.33, 1.78, 2.37, 2.84, 3.75], device=DEVICE)
        self.current_audio_env = torch.zeros(N_AUDIO, device=DEVICE) 

    def render(self, p, frames):
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)

        t = torch.arange(frames, device=DEVICE)
        root = p[0]
        base_f = 41.20 * (2.0 ** (root * 2.0))
        oct_idx = torch.arange(OCTAVES, device=DEVICE)
        weights = (torch.sin(np.pi * (oct_idx + root) / OCTAVES)**2).view(1, -1, 1)

        r_mod = self.ratios.clone()
        if p[18] > 0.5: r_mod[3] *= 0.89 
        if p[8] > 0.01: r_mod[5] *= (1.0 * (1-p[8]) + 0.89 * p[8]) 
        
        for i in range(6):
            if p[1+i] > 0.4: self.amps[i] = 2.0

        f_g = base_f * r_mod
        steps_g = 2 * np.pi * (f_g.unsqueeze(1) * (2.0 ** oct_idx)) / FS
        p_g = self.phases.unsqueeze(2) + steps_g.unsqueeze(2) * t
        
        sustain_val = 0.992 + p[10] * 0.0075
        if p[20] > 0.5: sustain_val = 1.0 
        
        wave = (((p_g % (2 * np.pi)) / np.pi - 1.0) * weights * (self.amps.view(-1,1,1) * torch.pow(sustain_val, t))).sum(dim=(0,1))
        self.phases = (p_g[:,:,-1] + steps_g.mean()) % (2 * np.pi)

        s_env = torch.zeros(1, device=DEVICE)
        if p[14] > 0.1: 
            steps_s = 2 * np.pi * ((base_f * 0.5) * (2.0 ** oct_idx)) / FS
            p_s = self.sub_phases.unsqueeze(1) + steps_s.unsqueeze(1) * t
            wave += (torch.sin(p_s) * weights.squeeze(0) * p[14]).sum(dim=0)
            self.sub_phases = (p_s[:,-1] + steps_s) % (2 * np.pi)
            s_env = torch.tensor([p[14]], device=DEVICE)

        z_env_t = torch.zeros(1, device=DEVICE)
        if p[15] > 0.1: 
            z_env = torch.exp(-t / (FS * 0.05))
            steps_z = 2 * np.pi * ((base_f * 20.0 * z_env + 100.0).unsqueeze(0) * (2.0 ** oct_idx).unsqueeze(1)) / FS
            p_z = self.zap_phases.unsqueeze(1) + torch.cumsum(steps_z, dim=1)
            wave += (torch.sin(p_z) * weights.squeeze(0) * 0.6 * p[15]).sum(dim=0)
            self.zap_phases = (p_z[:,-1] + steps_z[:,-1]) % (2 * np.pi)
            z_env_t = z_env[-1:] * p[15]
        
        k_env_t = torch.zeros(1, device=DEVICE)
        tempo_param = torch.clamp(p[19], 0.0, 1.0)
        beat_s = tempo_param * FS * 1.4 + FS * 0.1
        
        if p[11] > 0.5: 
            p_beat = ((self.ptr + t) % beat_s) / beat_s
            k_env = torch.exp(-p_beat * 15.0)
            steps_k = 2 * np.pi * ((k_env * 150.0 + base_f).unsqueeze(0) * (2.0 ** oct_idx).unsqueeze(1)) / FS
            p_k = self.kick_phases.unsqueeze(1) + torch.cumsum(steps_k, dim=1)
            wave += (torch.sin(p_k) * weights.squeeze(0) * k_env * 1.5).sum(dim=0)
            self.kick_phases = (p_k[:,-1] + steps_k[:,-1]) % (2 * np.pi)
            h_env = torch.exp(-(( (self.ptr + t) % (beat_s/4) ) / (beat_s/4)) * 30.0)
            wave += torch.randn(frames, device=DEVICE) * h_env * 0.2
            k_env_t = k_env[-1:]

        noise_lvl = torch.tensor([p[12] + p[13]], device=DEVICE)
        if noise_lvl.item() > 0.1: 
            ns = torch.randn(frames, device=DEVICE)
            if p[12] > 0.1: wave += torch.cumsum(ns * 0.01, dim=0) * p[12]
            if p[13] > 0.1: wave += ns * 0.05 * p[13]

        acid_val = 1.2 + p[17] * 6.8 
        wave = torch.tanh(wave * acid_val)
        
        warp_val = 0.85 + p[7] * 0.3
        delay_t = int(beat_s.item() * warp_val.item())
        delay_t = max(256, min(int(FS * 3) - frames, delay_t))
        idx = (self.ptr - delay_t + t) % int(FS * 3)
        echo = self.delay_buf[idx]
        fb_val = p[9] * 0.85
        if p[20] > 0.5: fb_val = 0.99
        self.delay_buf[(self.ptr + t) % int(FS * 3)] = wave + echo * fb_val
        wave_out = wave + echo * 0.4
        
        if p[16] > 0.4: wave_out = torch.roll(wave_out, frames // 4)
            
        self.ptr = (self.ptr + frames) % int(FS * 3)
        self.amps *= (sustain_val ** frames)
        
        self.current_audio_env = torch.cat([self.amps.clone(), s_env, z_env_t, k_env_t, noise_lvl])
        return wave_out * 0.3

# --- СТРОГАЯ ИЗОЛЯЦИЯ АУДИО ---
synth_real = GPU_Synth() # Синтезатор ТВОИХ рук (Правое ухо)
synth_virt = GPU_Synth() # Синтезатор МОЗГА (Левое ухо / Зал)

def audio_callback(outdata, frames, time_info, status):
    global audio_underruns
    if status.output_underflow: audio_underruns += 1

    stream_ctx = torch.cuda.stream(audio_stream) if audio_stream else contextlib.nullcontext()
    with stream_ctx:
        with torch.no_grad():
            sig_real = synth_real.render(params_real, frames) 
            sig_virt = synth_virt.render(params_virt, frames) 
            
            # АБСОЛЮТНОЕ РАЗДЕЛЕНИЕ ВЫХОДОВ
            outdata[:, 1] = sig_real.cpu().numpy() # Правое ухо - Только реальный геймпад
            outdata[:, 0] = sig_virt.cpu().numpy() # Левое ухо - Только виртуальный геймпад

# --- INPUT WORKER (РЕАЛЬНЫЙ ГЕЙМПАД) ---
def find_gamepad():
    for path in list_devices():
        try:
            d = InputDevice(path)
            if "sony" in d.name.lower() or "wireless" in d.name.lower(): return d
        except: pass
    return None

def gamepad_worker():
    global params_real, rhythm_locked, last_strum_time
    dev = find_gamepad()
    if not dev: print("🎮 Controller not found! Retrying..."); return

    fired_down, fired_up = [False]*6,[False]*6

    for event in dev.read_loop():
        if event.type == ecodes.EV_ABS:
            v = event.value
            if event.code == ecodes.ABS_X or event.code == ecodes.ABS_Y:
                lx = (dev.absinfo(ecodes.ABS_X).value - 128)/128.0
                ly = (dev.absinfo(ecodes.ABS_Y).value - 128)/128.0
                if (lx**2 + ly**2) > 0.1:
                    params_real[0] = (np.arctan2(lx, -ly) + np.pi) / (2 * np.pi)
            
            elif event.code == ecodes.ABS_RX: params_real[7] = v/255.0 
            elif event.code == ecodes.ABS_Z:  params_real[8] = v/255.0 
            elif event.code == ecodes.ABS_RZ: params_real[9] = params_real[10] = v/255.0 

            elif event.code == ecodes.ABS_RY: 
                ry = -(v-128)/128.0
                if abs(ry) < 0.1: fired_down, fired_up = [False]*6,[False]*6
                th = np.linspace(0.12, 0.85, 6)
                trig = False
                if ry > 0.1:
                    for i in range(6):
                        if ry > th[i] and not fired_down[i]:
                            params_real[1+i] = 1.0; fired_down[i] = trig = True
                elif ry < -0.1:
                    for i in range(6):
                        if ry < -th[i] and not fired_up[i]:
                            params_real[6-i] = 1.0; fired_up[i] = trig = True
                
                if trig and not rhythm_locked and params_real[20] < 0.5:
                    t_now = time.time()
                    interval = t_now - last_strum_time
                    if 0.1 < interval < 1.5:
                        strum_history.append(interval)
                        params_real[19] = np.clip((np.mean(strum_history) - 0.1) / 1.4, 0.0, 1.0)
                    last_strum_time = t_now

                threading.Timer(0.08, lambda: params_real[1:7].fill_(0)).start()
            
            elif event.code == ecodes.ABS_HAT0Y:
                if v == -1: params_real[11] = 1.0 
                elif v == 1: rhythm_locked = False 
                elif v == 0:
                    params_real[11] = 0.0; rhythm_locked = True
            
            elif event.code == ecodes.ABS_HAT0X:
                params_real[12] = 0.5 if v == -1 else 0.0 
                params_real[13] = 0.5 if v == 1 else 0.0

        elif event.type == ecodes.EV_KEY:
            st = float(event.value)
            if event.code == 304: params_real[14] = 0.8 if st else 0.0 
            if event.code == 307: params_real[15] = 1.0 if st else 0.0 
            if event.code == 308: params_real[16] = st 
            if event.code == 305: params_real[20] = st 
            if event.code == 310: params_real[18] = st 
            if event.code == 311: params_real[17] = st 

# --- INFERENCE WORKER (ТАЛАМУС И СВЯЗЬ ВСЕЛЕННЫХ) ---
def get_thalamus_clock(t_len):
    """Генерация временной сетки (Reference Frame) для синхронизации"""
    global thalamus_clock
    t_phase = np.linspace(thalamus_clock, thalamus_clock + 1.0, t_len)
    sin_rf = np.sin(2 * np.pi * t_phase).reshape(-1, 1)
    cos_rf = np.cos(2 * np.pi * t_phase).reshape(-1, 1)
    thalamus_clock += 1/60.0
    return np.concatenate([sin_rf, cos_rf], axis=1)

def inference_worker():
    global params_virt, prev_real_state
    while True:
        # Затухание огибающих для РЕАЛЬНОГО геймпада
        params_real[1:7] *= 0.6 
        
        # 1. ЧТЕНИЕ СЕНСОРОВ (Извлечение состояний)
        c_eeg = np.array(eeg_hist)
        c_thal = get_thalamus_clock(CONFIG["TRAJ_LEN"])
        
        c_motor_real = params_real.detach().cpu().numpy()
        c_audio_real = synth_real.current_audio_env.detach().cpu().numpy()
        
        real_motor_hist.append(c_motor_real)
        real_audio_hist.append(c_audio_real)
        
        # 2. ПОДГОТОВКА ДАННЫХ ДЛЯ УЧИТЕЛЯ (Правое ухо -> Буфер обучения)
        is_user_active = (c_motor_real[1:7].max() > 0.1) or (c_motor_real[11] > 0.5) or (c_motor_real[14:17].max() > 0.1)
        
        if prev_real_state is not None and is_user_active:
            # Сохраняем связку: (Прошлое ЭЭГ, Прошлая Реал.Моторика, Прошлое Реал.Аудио, Таламус) -> (Текущая Реал.Моторика, Текущее Реал.Аудио)
            replay_buf.append((*prev_real_state, c_motor_real, c_audio_real))
            
        prev_real_state = (c_eeg.copy(), np.array(real_motor_hist), np.array(real_audio_hist), c_thal.copy())
        
        # 3. ИНФЕРЕНС ИСПОЛНИТЕЛЯ (ВИРТУАЛЬНЫЙ ГЕЙМПАД / ЛЕВОЕ УХО)
        # !!! СТРОГАЯ ИЗОЛЯЦИЯ: Мы кормим сеть ТОЛЬКО ВИРТУАЛЬНОЙ историей, реальная сюда не попадает !!!
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                t_e = torch.tensor(c_eeg, device=DEVICE).float().unsqueeze(0)
                t_m_virt = torch.tensor(np.array(virt_motor_hist), device=DEVICE).float().unsqueeze(0)
                t_a_virt = torch.tensor(np.array(virt_audio_hist), device=DEVICE).float().unsqueeze(0)
                t_thal = torch.tensor(c_thal, device=DEVICE).float().unsqueeze(0)
                
                nxt_motor_virt, nxt_audio_virt = model(t_e, t_m_virt, t_a_virt, t_thal)
                
                pred_m_v = torch.nan_to_num(nxt_motor_virt[0], nan=0.0)
        
        # Обновляем Виртуальный Геймпад
        params_virt[1:7] = pred_m_v[1:7]
        params_virt[11] = pred_m_v[11]
        params_virt[16] = pred_m_v[16]
        
        smooth_idx =[0, 7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19, 20]
        params_virt[smooth_idx] = params_virt[smooth_idx] * 0.8 + pred_m_v[smooth_idx] * 0.2

        # Записываем сгенерированный результат в виртуальную историю (Замыкание петли)
        c_motor_virt = params_virt.detach().cpu().numpy()
        c_audio_virt = synth_virt.current_audio_env.detach().cpu().numpy()
        virt_motor_hist.append(c_motor_virt)
        virt_audio_hist.append(c_audio_virt)

        time.sleep(1/60.0)

# --- НЕПРЕРЫВНОЕ ОБУЧЕНИЕ (Вселенная Учителя) ---
shared_loss = 0.0
def training_loop():
    global dyn_batch, dyn_sleep, audio_underruns, shared_loss
    criterion = nn.MSELoss()
    
    while True:
        if len(replay_buf) > dyn_batch:
            batch = random.sample(replay_buf, int(dyn_batch))
            h_e, h_m_r, h_a_r, h_t, y_m_r, y_a_r = zip(*batch)
            
            t_he = torch.tensor(np.array(h_e), device=DEVICE).float()
            t_hm = torch.tensor(np.array(h_m_r), device=DEVICE).float()
            t_ha = torch.tensor(np.array(h_a_r), device=DEVICE).float()
            t_ht = torch.tensor(np.array(h_t), device=DEVICE).float()
            
            t_ym = torch.tensor(np.array(y_m_r), device=DEVICE).float()
            t_ya = torch.tensor(np.array(y_a_r), device=DEVICE).float()
            
            t0 = time.perf_counter()
            stream_ctx = torch.cuda.stream(train_stream) if train_stream else contextlib.nullcontext()
            
            with stream_ctx:
                with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                    model.train()
                    optimizer.zero_grad(set_to_none=True)
                    
                    # Сеть учится маппить: (ЭЭГ + Реал.Моторика + Реал.Звук + Таламус) -> (След. Реал.Моторика + След. Реал.Звук)
                    p_m, p_a = model(t_he, t_hm, t_ha, t_ht)
                    loss = criterion(p_m, t_ym) + criterion(p_a, t_ya)
                
                loss_val = loss.item()
                if math.isnan(loss_val) or math.isinf(loss_val):
                    optimizer.zero_grad(set_to_none=True)
                else:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    shared_loss = loss_val
                
                if train_stream: train_stream.synchronize() 
            
            step_time = time.perf_counter() - t0
            
            if audio_underruns > 0:
                dyn_batch = max(2, int(dyn_batch * 0.5))
                dyn_sleep = min(0.05, dyn_sleep + 0.005)
                audio_underruns = 0
            else:
                if step_time > 0.006: dyn_batch = max(2, int(dyn_batch * 0.9))
                elif step_time < 0.002: dyn_batch = min(CONFIG["MAX_BATCH"], int(dyn_batch + 1))
                dyn_sleep = 0.005 + step_time * 1.5
        time.sleep(dyn_sleep)

def serial_worker():
    while True:
        eeg_hist.append(np.random.randn(8) * 0.01)
        time.sleep(1/EEG_FS)

# --- GUI ---
class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Thousand Brains: Multimodal Sensory-Motor Integration")
        self.setGeometry(100, 100, 1200, 900)
        cw = QWidget(); self.setCentralWidget(cw); l = QVBoxLayout(cw)
        
        self.lbl = QLabel()
        self.lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #00FFCC; background: #111; padding: 10px;")
        l.addWidget(self.lbl)
        
        self.plot_w = pg.GraphicsLayoutWidget()
        l.addWidget(self.plot_w, stretch=1)
        
        self.p_rhythm = self.plot_w.addPlot(title="Micro-Patterns (Yellow: Right Ear/REAL | Blue: Left Ear/VIRTUAL)")
        self.u_rhythm_c = self.p_rhythm.plot(pen=pg.mkPen('y', width=2))
        self.b_rhythm_c = self.p_rhythm.plot(pen=pg.mkPen('c', width=2))
        self.plot_w.nextRow()
        
        self.p_loss = self.plot_w.addPlot(title="Prediction Error (Learning Progress)")
        self.loss_curve = self.p_loss.plot(pen=pg.mkPen('r', width=2))
        
        self.hist_u_r, self.hist_b_r = deque([0]*150, maxlen=150), deque([0]*150, maxlen=150)
        self.hist_loss = deque([0]*150, maxlen=150)
        
        self.tmr = QTimer(); self.tmr.timeout.connect(self.upd); self.tmr.start(30)

    def upd(self):
        # Визуализируем ритмы: Реальные (User) и Виртуальные (Model)
        ur = params_real[1:7].max().item() + params_real[11].item() + params_real[16].item() 
        br = params_virt[1:7].max().item() + params_virt[11].item() + params_virt[16].item() 
        self.hist_u_r.append(ur)
        self.hist_b_r.append(br)
        self.hist_loss.append(shared_loss)
        
        self.u_rhythm_c.setData(list(self.hist_u_r))
        self.b_rhythm_c.setData(list(self.hist_b_r))
        self.loss_curve.setData(list(self.hist_loss))
        
        self.lbl.setText(f"DJ Mode: STRICT ISOLATION | Loss: {shared_loss:.4f} | RTX Batch: {int(dyn_batch)}")

if __name__ == '__main__':
    app = QApplication(sys.argv); w = Win(); w.show()
    threading.Thread(target=serial_worker, daemon=True).start()
    threading.Thread(target=gamepad_worker, daemon=True).start()
    threading.Thread(target=inference_worker, daemon=True).start()
    threading.Thread(target=training_loop, daemon=True).start()
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=FS, blocksize=512):
        sys.exit(app.exec())
