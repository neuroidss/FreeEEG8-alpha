import sys, serial, threading, random, time, torch, math, contextlib
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget
from PyQt6.QtCore import QTimer
import pyqtgraph as pg
from collections import deque
from evdev import InputDevice, list_devices, ecodes

# --- КОНФИГУРАЦИЯ ---
FS = 44100
EEG_FS = 250.0
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OCTAVES = 5 

SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 921600

N_MOTOR = 22    
N_AUDIO = 10   
N_EEG = 8      
N_THALAMUS = 10 

CONFIG = {
    "EEG_WIN": 250,      
    "TRAJ_LEN": 128,     
    "LR": 5e-4,          
    "REPLAY_SIZE": 15000,
    "MAX_BATCH": 96     
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
# 1. MASSIVE SPATIOTEMPORAL HETERARCHY
# ==========================================
class SpatiotemporalColumn(nn.Module):
    def __init__(self, in_channels, channels=64):
        super().__init__()
        self.l1 = nn.Conv1d(in_channels, channels, kernel_size=3, padding=1, dilation=1)
        self.l2 = nn.Conv1d(channels, channels, kernel_size=3, padding=2, dilation=2)
        self.l3 = nn.Conv1d(channels, channels, kernel_size=3, padding=4, dilation=4)
        self.l4 = nn.Conv1d(channels, channels, kernel_size=3, padding=8, dilation=8)
        self.l5 = nn.Conv1d(channels, channels, kernel_size=3, padding=16, dilation=16)
        self.l6 = nn.Conv1d(channels, channels, kernel_size=3, padding=32, dilation=32)
        self.act = nn.GELU()

    def forward(self, x):
        x1 = self.act(self.l1(x))
        x2 = self.act(self.l2(x1)) + x1
        x3 = self.act(self.l3(x2)) + x2
        x4 = self.act(self.l4(x3)) + x3
        x5 = self.act(self.l5(x4)) + x4
        x6 = self.act(self.l6(x5)) + x5
        return torch.cat([x1, x2, x3, x4, x5, x6], dim=1) 

class DeepSpatiotemporalBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.col_eeg = SpatiotemporalColumn(N_EEG, 64)       
        self.col_motor = SpatiotemporalColumn(N_MOTOR, 64)   
        self.col_audio = SpatiotemporalColumn(N_AUDIO, 64)   
        self.col_thal = SpatiotemporalColumn(N_THALAMUS, 32) 
        
        combined_dim = 384 + 384 + 384 + 192 
        
        self.het_l1 = nn.Conv1d(combined_dim, 512, kernel_size=3, padding=1, dilation=1)
        self.het_l2 = nn.Conv1d(512, 512, kernel_size=3, padding=4, dilation=4)
        self.het_l3 = nn.Conv1d(512, 512, kernel_size=3, padding=16, dilation=16)
        self.act = nn.GELU()
        
        heterarchy_out_dim = combined_dim + 512 * 3
        
        self.action_generator = nn.Sequential(
            nn.Linear(heterarchy_out_dim, 1024), self.act,
            nn.LayerNorm(1024),
            nn.Linear(1024, 512), self.act,
            nn.LayerNorm(512),
            nn.Linear(512, N_MOTOR), nn.Sigmoid()
        )
        
        self.forward_model = nn.Sequential(
            nn.Linear(heterarchy_out_dim + N_MOTOR, 512), self.act,
            nn.LayerNorm(512),
            nn.Linear(512, N_AUDIO), nn.Sigmoid()
        )

    def forward(self, eeg, motor, audio, thalamus):
        e = eeg.permute(0, 2, 1)     
        m = motor.permute(0, 2, 1)   
        a = audio.permute(0, 2, 1)   
        t = thalamus.permute(0, 2, 1)
        
        feat_e = nn.functional.adaptive_avg_pool1d(self.col_eeg(e), CONFIG["TRAJ_LEN"])
        feat_m = self.col_motor(m)   
        feat_a = self.col_audio(a)   
        feat_t = self.col_thal(t)
        
        context_0 = torch.cat([feat_e, feat_m, feat_a, feat_t], dim=1) 
        
        context_1 = self.act(self.het_l1(context_0))
        context_2 = self.act(self.het_l2(context_1)) + context_1
        context_3 = self.act(self.het_l3(context_2)) + context_2
        
        state_0 = context_0[:, :, -1] 
        state_1 = context_1[:, :, -1]
        state_2 = context_2[:, :, -1]
        state_3 = context_3[:, :, -1] 
        
        global_heterarch_state = torch.cat([state_0, state_1, state_2, state_3], dim=1)
        
        next_action = self.action_generator(global_heterarch_state)
        efference_copy = torch.cat([global_heterarch_state, next_action], dim=1)
        expected_audio = self.forward_model(efference_copy)
        
        return next_action, expected_audio

model = DeepSpatiotemporalBrain().to(DEVICE)
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
scaler = torch.amp.GradScaler('cuda' if torch.cuda.is_available() else 'cpu')

# --- БУФЕРЫ ПАМЯТИ И ФЛАГИ ---
eeg_hist = deque([np.zeros(N_EEG)] * CONFIG["EEG_WIN"], maxlen=CONFIG["EEG_WIN"])
real_motor_hist = deque([np.zeros(N_MOTOR)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
real_audio_hist = deque([np.zeros(N_AUDIO)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])

params_real = torch.zeros(N_MOTOR, device=DEVICE) 
params_real[19] = 0.2857 
params_real[21] = 0.5 

virt_motor_hist = deque([np.zeros(N_MOTOR)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
virt_audio_hist = deque([np.zeros(N_AUDIO)] * CONFIG["TRAJ_LEN"], maxlen=CONFIG["TRAJ_LEN"])
params_virt = torch.zeros(N_MOTOR, device=DEVICE) 
params_virt[19] = 0.2857 
params_virt[21] = 0.5

expected_audio_env = np.zeros(N_AUDIO)
prev_real_state = None

rhythm_locked = True
last_strum_time = time.time()
strum_history = deque([0.5, 0.5], maxlen=4)
global_frame_counter = 0

virt_fired_down = [False]*6
virt_fired_up = [False]*6

lesson_mode = False
exam_mode = False
real_fired_down = [False]*6 
real_fired_up = [False]*6

replay_buf = deque(maxlen=CONFIG["REPLAY_SIZE"]) 
priority_buf = deque(maxlen=180) 

# ==========================================
# 2. РЕАЛЬНАЯ ЧИТАЛКА ЭЭГ
# ==========================================
def serial_worker():
    global eeg_hist
    try: 
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
        print(f"🧠 EEG Hardware Connected: {SERIAL_PORT}")
    except Exception: 
        ser = None
        print(f"⚠️ EEG Hardware not found. Using SIMULATOR.")
    
    buf = bytearray()
    while True:
        if ser:
            data = ser.read(1024)
            if not data: 
                time.sleep(0.001)
                continue
            buf.extend(data)
            while len(buf) >= 33:
                if buf[0] == 0xA0 and buf[32] == 0xC0:
                    s =[((buf[2+ch*3]<<16) | (buf[3+ch*3]<<8) | buf[4+ch*3]) for ch in range(8)]
                    s =[v - 0x1000000 if v & 0x800000 else v for v in s]
                    val = np.array(s, dtype=np.float32) / 1e6
                    eeg_hist.append(val)
                    del buf[:33]
                else: 
                    buf.pop(0)
        else:
            eeg_hist.append(np.random.randn(8) * 0.01)
            time.sleep(1/EEG_FS)

# --- GPU СИНТЕЗАТОР (PERFECTED SHEPARD-RISSET) ---
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
        self.f_base = 32.7 

    def render(self, p, frames):
        p = torch.nan_to_num(p, nan=0.0, posinf=1.0, neginf=0.0)
        t = torch.arange(frames, device=DEVICE)
        
        pos = p[0] 
        oct_idx = torch.arange(OCTAVES, device=DEVICE).float()
        phi = (pos + oct_idx / OCTAVES) % 1.0  
        weights = (torch.sin(np.pi * phi) ** 2) 
        freq_mult = 2.0 ** (phi * OCTAVES)

        r_mod = self.ratios.clone()
        if p[18] > 0.5: r_mod[3] *= 0.89 
        if p[8] > 0.01: r_mod[5] *= (1.0 * (1-p[8]) + 0.89 * p[8]) 
        
        for i in range(6):
            if p[1+i] > 0.4: self.amps[i] = 2.0
            
        sustain_val = 0.992 + p[10] * 0.0075
        if p[20] > 0.5: sustain_val = 1.0 

        f_g = self.f_base * r_mod.unsqueeze(1) * freq_mult.unsqueeze(0)
        steps_g = 2 * np.pi * f_g / FS
        p_g = self.phases.unsqueeze(2) + steps_g.unsqueeze(2) * t
        
        w_g = weights.view(1, OCTAVES, 1) * self.amps.view(-1, 1, 1) * torch.pow(sustain_val, t)
        wave = (((p_g % (2 * np.pi)) / np.pi - 1.0) * w_g).sum(dim=(0, 1))
        
        self.phases = (p_g[:, :, -1] + steps_g) % (2 * np.pi)

        s_env = torch.zeros(1, device=DEVICE)
        if p[14] > 0.1: 
            f_s = self.f_base * 0.5 * freq_mult
            steps_s = 2 * np.pi * f_s / FS
            p_s = self.sub_phases.unsqueeze(1) + steps_s.unsqueeze(1) * t
            wave += (torch.sin(p_s) * weights.view(OCTAVES, 1) * p[14]).sum(dim=0)
            self.sub_phases = (p_s[:, -1] + steps_s) % (2 * np.pi)
            s_env = torch.tensor([p[14]], device=DEVICE)

        z_env_t = torch.zeros(1, device=DEVICE)
        if p[15] > 0.1: 
            z_env = torch.exp(-t / (FS * 0.05))
            base_z = self.f_base * 20.0 * z_env + 100.0
            f_z = base_z.unsqueeze(0) * freq_mult.unsqueeze(1)
            steps_z = 2 * np.pi * f_z / FS
            p_z = self.zap_phases.unsqueeze(1) + torch.cumsum(steps_z, dim=1)
            wave += (torch.sin(p_z) * weights.view(OCTAVES, 1) * 0.6 * p[15]).sum(dim=0)
            self.zap_phases = (p_z[:, -1] + steps_z[:, -1]) % (2 * np.pi)
            z_env_t = z_env[-1:] * p[15]
        
        k_env_t = torch.zeros(1, device=DEVICE)
        tempo_param = torch.clamp(p[19], 0.0, 1.0)
        beat_s = tempo_param * FS * 1.4 + FS * 0.1
        
        if p[11] > 0.5: 
            p_beat = ((self.ptr + t) % beat_s) / beat_s
            k_env = torch.exp(-p_beat * 15.0)
            base_k = k_env * 150.0 + self.f_base
            f_k = base_k.unsqueeze(0) * freq_mult.unsqueeze(1)
            steps_k = 2 * np.pi * f_k / FS
            p_k = self.kick_phases.unsqueeze(1) + torch.cumsum(steps_k, dim=1)
            wave += (torch.sin(p_k) * weights.view(OCTAVES, 1) * k_env * 1.5).sum(dim=0)
            self.kick_phases = (p_k[:, -1] + steps_k[:, -1]) % (2 * np.pi)
            
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

synth_real = GPU_Synth()
synth_virt = GPU_Synth()

def audio_callback(outdata, frames, time_info, status):
    global audio_underruns
    if status.output_underflow: audio_underruns += 1

    stream_ctx = torch.cuda.stream(audio_stream) if audio_stream else contextlib.nullcontext()
    with stream_ctx:
        with torch.no_grad():
            sig_real = synth_real.render(params_real, frames) 
            sig_virt = synth_virt.render(params_virt, frames) 
            outdata[:, 1] = sig_real.cpu().numpy() 
            outdata[:, 0] = sig_virt.cpu().numpy() 

def find_gamepad():
    for path in list_devices():
        try:
            d = InputDevice(path)
            if "sony" in d.name.lower() or "wireless" in d.name.lower(): return d
        except: pass
    return None

def gamepad_worker():
    global params_real, rhythm_locked, last_strum_time
    global lesson_mode, exam_mode
    
    dev = find_gamepad()
    if not dev: print("🎮 Controller not found! Retrying..."); return

    fired_down, fired_up = [False]*6,[False]*6

    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY:
            st = float(event.value)
            
            if event.code == 315: # OPTIONS
                lesson_mode = (st > 0)
            if event.code == 314: # SHARE
                exam_mode = (st > 0)

            if event.code == 304: params_real[14] = 0.8 if st else 0.0 
            if event.code == 307: params_real[15] = 1.0 if st else 0.0 
            if event.code == 308: params_real[16] = st 
            if event.code == 305: params_real[20] = st 
            if event.code == 310: params_real[18] = st 
            if event.code == 311: params_real[17] = st 

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
                params_real[21] = (ry + 1.0) / 2.0
                
                if abs(ry) < 0.1: fired_down, fired_up =[False]*6,[False]*6
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
                elif v == 0: params_real[11] = 0.0; rhythm_locked = True
            
            elif event.code == ecodes.ABS_HAT0X:
                params_real[12] = 0.5 if v == -1 else 0.0 
                params_real[13] = 0.5 if v == 1 else 0.0

def get_thalamus_clocks(t_len):
    global global_frame_counter
    t = np.arange(global_frame_counter, global_frame_counter + t_len)
    
    ultra_slow_freq = t / (EEG_FS * 60 * 2.5) 
    slow_freq = t / (EEG_FS * 30.0) 
    meso_freq = t / (EEG_FS * 4.0)
    fast_freq = t / (EEG_FS * 0.5)
    ultra_fast_freq = t / (EEG_FS * 0.05)

    clocks = np.stack([
        np.sin(2 * np.pi * ultra_slow_freq), np.cos(2 * np.pi * ultra_slow_freq),
        np.sin(2 * np.pi * slow_freq),       np.cos(2 * np.pi * slow_freq),
        np.sin(2 * np.pi * meso_freq),       np.cos(2 * np.pi * meso_freq),
        np.sin(2 * np.pi * fast_freq),       np.cos(2 * np.pi * fast_freq),
        np.sin(2 * np.pi * ultra_fast_freq), np.cos(2 * np.pi * ultra_fast_freq)
    ], axis=1)
    
    global_frame_counter += 1 
    return clocks

def inference_worker():
    global params_virt, prev_real_state, expected_audio_env
    global virt_fired_down, virt_fired_up
    global real_fired_down, real_fired_up

    while True:
        params_real[1:7] *= 0.6 
        tick = global_frame_counter % 120 
        
        # Симуляция ЭЭГ (работает и на уроке, и на экзамене)
        if lesson_mode or exam_mode:
            sim_eeg = np.zeros((CONFIG["EEG_WIN"], N_EEG))
            t_arr = np.arange(global_frame_counter - CONFIG["EEG_WIN"], global_frame_counter)
            sim_eeg[:, 0] = np.sin(t_arr / 120 * 2 * np.pi) 
            sim_eeg[:, 1] = np.sin(t_arr / 15 * 2 * np.pi)  
            c_eeg = sim_eeg
        else:
            c_eeg = np.array(eeg_hist)
            
        # Симуляция моторики (ТОЛЬКО на уроке)
        if lesson_mode:
            params_real[0] = 0.1 if tick < 60 else 0.4 
            params_real[20] = 1.0 
            
            phase = tick % 15
            if phase < 5:
                ry = 0.9 if (tick % 30) < 15 else -0.9 
            else:
                ry = 0.0
                
            params_real[21] = (ry + 1.0) / 2.0
            
            th = np.linspace(0.12, 0.85, 6)
            if abs(ry) < 0.1:
                real_fired_down, real_fired_up = [False]*6, [False]*6
                
            if ry > 0.1:
                for i in range(6):
                    if ry > th[i] and not real_fired_down[i]:
                        params_real[1+i] = 1.0
                        real_fired_down[i] = True
                        threading.Timer(0.08, lambda idx=1+i: params_real.__setitem__(idx, 0.0)).start()
            elif ry < -0.1:
                for i in range(6):
                    if ry < -th[i] and not real_fired_up[i]:
                        params_real[6-i] = 1.0
                        real_fired_up[i] = True
                        threading.Timer(0.08, lambda idx=6-i: params_real.__setitem__(idx, 0.0)).start()
        
        c_thal = get_thalamus_clocks(CONFIG["TRAJ_LEN"])
        c_motor_real = params_real.detach().cpu().numpy()
        c_audio_real = synth_real.current_audio_env.detach().cpu().numpy()
        
        real_motor_hist.append(c_motor_real)
        real_audio_hist.append(c_audio_real)
        
        # --- ФУНДАМЕНТАЛЬНОЕ ИСПРАВЛЕНИЕ: ВРЕМЯ НЕПРЕРЫВНО ---
        # Мы удалили фильтр is_user_active. 
        # Мозг всегда активен. Тишина - это тоже паттерн (пауза в ритме).
        # Записываем КАЖДЫЙ кадр в буферы памяти для идеального чувства времени.
        if prev_real_state is not None:
            experience = (*prev_real_state, c_motor_real, c_audio_real)
            replay_buf.append(experience)
            priority_buf.append(experience) 
            
        prev_real_state = (c_eeg.copy(), np.array(real_motor_hist), np.array(real_audio_hist), c_thal.copy())
        
        with torch.no_grad():
            with torch.amp.autocast('cuda' if torch.cuda.is_available() else 'cpu'):
                t_e = torch.tensor(c_eeg, device=DEVICE).float().unsqueeze(0)
                t_m_virt = torch.tensor(np.array(virt_motor_hist), device=DEVICE).float().unsqueeze(0)
                t_a_virt = torch.tensor(np.array(virt_audio_hist), device=DEVICE).float().unsqueeze(0)
                t_thal = torch.tensor(c_thal, device=DEVICE).float().unsqueeze(0)
                
                nxt_motor_virt, expected_audio = model(t_e, t_m_virt, t_a_virt, t_thal)
                pred_m_v = torch.nan_to_num(nxt_motor_virt[0], nan=0.0)
                expected_audio_env = torch.nan_to_num(expected_audio[0], nan=0.0).cpu().numpy()
        
        params_virt[21] = pred_m_v[21]
        ry_v = pred_m_v[21].item() * 2.0 - 1.0 
        
        if abs(ry_v) < 0.1:
            virt_fired_down, virt_fired_up = [False]*6,[False]*6
            
        th = np.linspace(0.12, 0.85, 6)
        if ry_v > 0.1:
            for i in range(6):
                if ry_v > th[i] and not virt_fired_down[i]:
                    params_virt[1+i] = 1.0
                    virt_fired_down[i] = True
                    threading.Timer(0.08, lambda idx=1+i: params_virt.__setitem__(idx, 0.0)).start()
        elif ry_v < -0.1:
            for i in range(6):
                if ry_v < -th[i] and not virt_fired_up[i]:
                    params_virt[6-i] = 1.0
                    virt_fired_up[i] = True
                    threading.Timer(0.08, lambda idx=6-i: params_virt.__setitem__(idx, 0.0)).start()
        
        params_virt[0] = pred_m_v[0] 
        params_virt[20] = pred_m_v[20] 
        params_virt[11] = pred_m_v[11]
        params_virt[16] = pred_m_v[16]
        
        smooth_idx =[7, 8, 9, 10, 12, 13, 14, 15, 17, 18, 19] 
        params_virt[smooth_idx] = params_virt[smooth_idx] * 0.8 + pred_m_v[smooth_idx] * 0.2

        c_motor_virt = params_virt.detach().cpu().numpy()
        c_audio_virt = synth_virt.current_audio_env.detach().cpu().numpy()
        virt_motor_hist.append(c_motor_virt)
        virt_audio_hist.append(c_audio_virt)

        time.sleep(1/60.0)

# --- TRAINING LOOP ---
shared_loss = 0.0
def training_loop():
    global dyn_batch, dyn_sleep, audio_underruns, shared_loss
    mse_none = nn.MSELoss(reduction='none') 
    
    while True:
        if len(replay_buf) > dyn_batch and len(priority_buf) > dyn_batch * 0.7:
            priority_samples_count = int(dyn_batch * 0.7) 
            long_term_samples_count = dyn_batch - priority_samples_count 
            
            batch = random.sample(priority_buf, priority_samples_count) + random.sample(replay_buf, long_term_samples_count)
            random.shuffle(batch)
            
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
                    
                    p_m, p_a = model(t_he, t_hm, t_ha, t_ht)
                    
                    loss_m = mse_none(p_m, t_ym)
                    weight_m = torch.ones_like(loss_m)
                    weight_m[:, 1:7] = 10.0 
                    weight_m[:, 21] = 10.0  
                    
                    loss_m_weighted = (loss_m * weight_m).mean()
                    loss_a = mse_none(p_a, t_ya).mean()
                    
                    loss = loss_m_weighted + loss_a
                
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
                dyn_batch = max(2, int(dyn_batch * 0.5)); dyn_sleep = min(0.05, dyn_sleep + 0.005); audio_underruns = 0
            else:
                if step_time > 0.006: dyn_batch = max(2, int(dyn_batch * 0.9))
                elif step_time < 0.002: dyn_batch = min(CONFIG["MAX_BATCH"], int(dyn_batch + 1))
                dyn_sleep = 0.005 + step_time * 1.5
        time.sleep(dyn_sleep)

class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Deep Psy-Cortex V21.9 (Continuous Time Mode)")
        self.setGeometry(100, 100, 1200, 900)
        cw = QWidget(); self.setCentralWidget(cw); l = QVBoxLayout(cw)
        
        self.lbl = QLabel()
        self.lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #00FFCC; background: #111; padding: 10px;")
        l.addWidget(self.lbl)
        
        self.plot_w = pg.GraphicsLayoutWidget()
        l.addWidget(self.plot_w, stretch=1)
        
        self.p_rhythm = self.plot_w.addPlot(title="Virtual Thumb tracking (Yellow: Real | Blue: AI)")
        self.u_rhythm_c = self.p_rhythm.plot(pen=pg.mkPen('y', width=2))
        self.b_rhythm_c = self.p_rhythm.plot(pen=pg.mkPen('c', width=2))
        self.plot_w.nextRow()
        
        self.p_audio = self.plot_w.addPlot(title="Sensory Anticipation (Magenta: Expected | Green: Actual)")
        self.exp_audio_c = self.p_audio.plot(pen=pg.mkPen('m', width=2))
        self.act_audio_c = self.p_audio.plot(pen=pg.mkPen('g', width=2))
        
        self.hist_u_r, self.hist_b_r = deque([0]*150, maxlen=150), deque([0]*150, maxlen=150)
        self.hist_exp_a, self.hist_act_a = deque([0]*150, maxlen=150), deque([0]*150, maxlen=150)
        
        self.tmr = QTimer(); self.tmr.timeout.connect(self.upd); self.tmr.start(30)

    def upd(self):
        ur = params_real[21].item() 
        br = params_virt[21].item()
        
        self.hist_u_r.append(ur)
        self.hist_b_r.append(br)
        
        self.hist_exp_a.append(expected_audio_env[8])
        actual_audio = synth_virt.current_audio_env.detach().cpu().numpy()
        self.hist_act_a.append(actual_audio[8])
        
        self.u_rhythm_c.setData(list(self.hist_u_r))
        self.b_rhythm_c.setData(list(self.hist_b_r))
        self.exp_audio_c.setData(list(self.hist_exp_a))
        self.act_audio_c.setData(list(self.hist_act_a))
        
        status = "LIVE DJ MODE"
        if lesson_mode:
            status = "🎓 LESSON MODE (HOLDING OPTIONS...)"
            self.lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #FFFF00; background: #111; padding: 10px;")
        elif exam_mode:
            status = "📝 EXAM MODE (HOLDING SHARE...)"
            self.lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #FF00FF; background: #111; padding: 10px;")
        else:
            self.lbl.setStyleSheet("font-size: 16px; font-weight: bold; color: #00FFCC; background: #111; padding: 10px;")
            
        self.lbl.setText(f"{status} | Loss: {shared_loss:.4f} | RTX Batch: {int(dyn_batch)}")

if __name__ == '__main__':
    app = QApplication(sys.argv); w = Win(); w.show()
    threading.Thread(target=serial_worker, daemon=True, name="EEG_Reader").start()
    threading.Thread(target=gamepad_worker, daemon=True, name="Gamepad_Reader").start()
    threading.Thread(target=inference_worker, daemon=True, name="Inference_Loop").start()
    threading.Thread(target=training_loop, daemon=True, name="Training_Loop").start()
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=FS, blocksize=512):
        sys.exit(app.exec())
