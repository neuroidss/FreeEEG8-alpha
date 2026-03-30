import sys
import asyncio
import numpy as np
import sounddevice as sd
import torch
import time
from PyQt6.QtWidgets import *
from PyQt6.QtCore import pyqtSignal, QThread, Qt
from bleak import BleakScanner, BleakClient

# ==========================================
# 1. КОНФИГУРАЦИЯ
# ==========================================
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()
CMD_UUID     = "c0de0001-36e1-4688-b7f5-ea07361b26a8".lower()

MAX_DEVICES = 16
CHANNELS = 8
EEG_FS = 250
AUDIO_FS = 44100
BUFFER_SEC = 2
BUFFER_SIZE = EEG_FS * BUFFER_SEC

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

RADIUS = 10.0
ANGLES_RAD = np.deg2rad(np.array([-72, -36, 36, 72, 108, 144, -144, -108]))
X_pins, Y_pins = RADIUS * np.cos(ANGLES_RAD), RADIUS * np.sin(ANGLES_RAD)
dX, dY = X_pins - 10.0, Y_pins - 0.0
SPATIAL_GPU = torch.tensor(np.linalg.pinv(np.column_stack([dX, dY])), dtype=torch.float32, device=DEVICE)

hive_state = {
    'users':[{
        'active': False, 'mac': None, 'stage': 4, 'verified': False,
        'delta_phase': 0.0, 'theta_phase': 0.0, 'delta_freq': 2.0,
        'delta_amp': 0.0, 'theta_amp': 0.0, 'alpha_amp': 0.0, 'gamma_amp': 0.0,
        'lead_x': 0.0, 'lead_y': 0.0, 'uv_scale': 1.0
    } for _ in range(MAX_DEVICES)]
}

# ==========================================
# 2. АУДИО ДВИЖОК — MOTOR IMAGERY (RHYTHM & TONALITY)
# ==========================================
class HiveAudioEngine:
    def __init__(self):
        self.timer_kick = np.ones(MAX_DEVICES) * 10.0
        self.timer_bass = np.ones(MAX_DEVICES) * 10.0
        self.timer_hat  = np.ones(MAX_DEVICES) * 10.0
        self.last_d = np.zeros(MAX_DEVICES)
        self.last_t = np.zeros(MAX_DEVICES)
        self.phase_lead_c = np.zeros(MAX_DEVICES)
        self.phase_lead_m = np.zeros(MAX_DEVICES)
        self.gamma_angle_mem = np.zeros(MAX_DEVICES)

        self.delay_buf = np.zeros((2, AUDIO_FS * 2))
        self.delay_ptr = 0

        # --- ОГИБАЮЩИЕ ИНСТРУМЕНТА (Гитара/Палочки) ---
        self.env_rhythm = 0.0    # Правая рука: Удары, энергия, атака
        self.env_tonality = 0.0  # Левая рука: Аккорды, слайды (спектральный центроид)

        # ===================== НАСТРОЙКИ ГРОМКОСТИ =====================
        self.master_gain   = 0.0018      
        self.kick_gain     = 1.40
        self.bass_gain     = 1.15
        self.gamma_gain    = 0.48        
        self.lead_gain     = 0.58
        self.hat_gain      = 0.65

    def process_audio(self, indata, outdata, frames, time_info, status):
        outdata.fill(0)
        dt = 1.0 / AUDIO_FS
        t_arr = np.arange(frames) * dt

        # =========================================================
        # 1. АНАЛИЗ РУК МУЗЫКАНТА (Без вмешательства в сам звук)
        # =========================================================
        instrument_in = indata[:, 0]
        window = np.hanning(frames)
        spec = np.abs(np.fft.rfft(instrument_in * window))
        freqs = np.fft.rfftfreq(frames, d=dt)

        # ПРАВАЯ РУКА (Ритм/Энергия)
        energy = np.mean(spec)
        rhythm_raw = np.clip(energy * 15.0, 0.0, 1.0) # Нормализуем от 0 до 1
        
        # ЛЕВАЯ РУКА (Тональность/Слайд - Спектральный центроид)
        tonality_raw = 0.0
        if energy > 0.005: # Если есть звук, вычисляем его высоту/яркость
            centroid = np.sum(freqs * spec) / np.sum(spec)
            tonality_raw = np.clip(centroid / 1500.0, 0.0, 1.0) # От 0 до 1 (до 1500 Гц)

        # Сглаживание (RC-фильтр), чтобы параметры менялись органично
        self.env_rhythm = self.env_rhythm * 0.8 + rhythm_raw * 0.2
        self.env_tonality = self.env_tonality * 0.9 + tonality_raw * 0.1

        psy_L = np.zeros(frames)
        psy_R = np.zeros(frames)

        for i in range(MAX_DEVICES):
            u = hive_state['users'][i]
            if not u['active'] or not u['verified']: continue

            d = u['delta_phase']
            t = u['theta_phase']
            g = u['gamma_amp']
            a = u['alpha_amp']
            angle = np.arctan2(u['lead_y'], u['lead_x'])

            if d > 0 and self.last_d[i] <= 0: self.timer_kick[i] = 0
            if t > 1.57 and self.last_t[i] <= 1.57 and g > 0.08: self.timer_hat[i] = 0
            self.last_d[i] = d
            self.last_t[i] = t

            # --- KICK (Дельта мозга + Ритм Правой Руки) ---
            if self.timer_kick[i] < 0.38:
                T = self.timer_kick[i] + t_arr
                # ПООЩРЕНИЕ УДАРА: Если физический удар совпал с бочкой мозга, 
                # затухание (16) уменьшается, бочка звучит дольше и плотнее (Sustain).
                # ТИШИНА: decay = 16.0 (Идеальный оригинал)
                decay = 16.0 - (self.env_rhythm * 6.0) 
                kick = np.sin(2*np.pi*(48*T - 3.5*np.exp(-55*T))) * np.exp(-decay*T) * np.clip(u['delta_amp']*2.2, 0.6, 1.6)
                psy_L += kick * self.kick_gain
                self.timer_kick[i] += frames*dt

            # --- BASS (Тета мозга + Ритм Правой Руки) ---
            if self.timer_bass[i] < 0.48:
                T = self.timer_bass[i] + t_arr
                bass_osc = 2*((T%1)-0.5) * np.exp(-13*T) * np.clip(u['theta_amp']*1.9, 0.35, 1.0)
                # ПООЩРЕНИЕ РИТМА: Физический удар добавляет сатурации (жирности) басу.
                # ТИШИНА: sat_boost = 0.0 (Идеальный оригинал)
                sat_boost = self.env_rhythm * 3.0
                bass = np.tanh(bass_osc * (2.2 + u['theta_amp']*5 + sat_boost))
                psy_L += bass * self.bass_gain
                self.timer_bass[i] += frames*dt

            # --- SPATIAL VOICE (Гамма мозга + Тональность Левой Руки) ---
            if g > 0.14:
                spatial_pos = (angle + np.pi) / (2 * np.pi)
                freq = 160 + spatial_pos * 1740
                if abs(angle - self.gamma_angle_mem[i]) < 0.45:
                    freq = 0.68*freq + 0.32*(160 + (self.gamma_angle_mem[i] + np.pi)/(2*np.pi)*1740)
                pc = self.phase_lead_c[i] + np.arange(1, frames+1) * freq * dt
                self.phase_lead_c[i] = pc[-1] % 1.0
                
                # ПООЩРЕНИЕ АККОРДА: Яркий аккорд меняет текстуру (гармоники) гамма-голоса.
                # ТИШИНА: texture = 3.7 (Идеальный оригинал)
                texture = 3.7 + (self.env_tonality * 2.5)
                voice = np.sin(2*np.pi*pc) * (0.7 + 0.8*np.sin(2*np.pi*pc * texture))
                
                voice *= g * 1.65 * np.exp(-((t*3.8)%1)*11)
                psy_R += voice * self.gamma_gain
                self.gamma_angle_mem[i] = angle

            # --- ALPHA LEAD (Альфа мозга + Слайды Левой Руки + Удары Правой) ---
            lead_vol = np.clip((a - 0.06)*6, 0, 0.72)
            if lead_vol > 0.01:
                # ПООЩРЕНИЕ СЛАЙДА: Движение по грифу меняет высоту тона (Pitch Bend)
                # ТИШИНА: slide_mod = 0.0 (Идеальный оригинал)
                slide_mod = self.env_tonality * 100.0 
                c_freq = (150 + u['lead_x'] * 360) + slide_mod
                
                pc = self.phase_lead_m[i] + np.arange(1, frames+1) * c_freq * dt
                pm = self.phase_lead_m[i] + np.arange(1, frames+1) * (c_freq*0.5) * dt
                self.phase_lead_m[i] = pm[-1] % 1.0
                
                # ПООЩРЕНИЕ РИТМА: Удар по струнам открывает FM-фильтр лида (делает его "квакающим")
                # ТИШИНА: fm_depth = 3.5 (Идеальный оригинал)
                fm_depth = 3.5 + (self.env_rhythm * 2.5)
                
                lead = np.sin(2*np.pi*pc + fm_depth*np.sin(2*np.pi*pm))
                psy_R += lead * lead_vol * self.lead_gain * np.exp(-((t*2)%1)*7)

        # === ФИНАЛЬНОЕ СВЕДЕНИЕ (Без звука гитары в миксе!) ===
        mix = psy_L + psy_R
        mix = np.tanh(mix * 2.2)                    # limiter
        mix *= self.master_gain

        active =[u for u in hive_state['users'] if u['active']]
        drive = np.clip(np.mean([u['gamma_amp'] for u in active]) * 1.8, 0, 1.0) if active else 0.0
        mix = np.tanh(mix * (1.0 + drive * 2.5))

        delay_amount = 0.5
        if active:
            mean_g = np.mean([u['gamma_amp'] for u in active])
            mean_a = np.mean([u['alpha_amp'] for u in active])
            delay_amount = np.clip(0.75 - mean_g + mean_a, 0.2, 0.9)

        d_samp = int(0.28 * AUDIO_FS)
        idx = (self.delay_ptr - d_samp + np.arange(frames)) % (AUDIO_FS*2)
        echo_L = self.delay_buf[0, idx]
        echo_R = self.delay_buf[1, idx]

        write = (self.delay_ptr + np.arange(frames)) % (AUDIO_FS*2)
        self.delay_buf[0, write] = mix + echo_R * 0.55
        self.delay_buf[1, write] = mix * 0.75 + echo_L * 0.55
        self.delay_ptr = (self.delay_ptr + frames) % (AUDIO_FS*2)

        outdata[:,0] = np.tanh(mix + echo_L * delay_amount)
        outdata[:,1] = np.tanh(mix * 0.9 + echo_R * delay_amount)


# ==========================================
# 4. BLE + GPU DSP
# ==========================================
class SingleDeviceManager:
    def __init__(self, device, slot_id, worker_ref):
        self.device, self.slot, self.mac, self.worker = device, slot_id, device.address, worker_ref
        self.is_connected = False
        self.is_connecting = True # ЗАЩИТА ОТ ГОНКИ ПОДКЛЮЧЕНИЙ
        self.client = None
        self.presets = {4: (7, 5, 0, 0x000F)}
        self.target_stage = self.current_stage = 4
        self.read_event = asyncio.Event()
        self.last_read_data = None

    def cmd_callback(self, s, data):
        if len(data) == 3:
            self.last_read_data = data
            self.read_event.set()

    async def write_and_verify(self, addr, val, mask=None, desc=""):
        for _ in range(6):
            cmd = bytes([addr, (val>>8)&0xFF, val&0xFF, (mask>>8)&0xFF, mask&0xFF]) if mask else bytes([addr, (val>>8)&0xFF, val&0xFF])
            await self.client.write_gatt_char(CMD_UUID, cmd, response=False)
            await asyncio.sleep(0.18)
            after = await self.get_reg_value(addr)
            if after is not None and ((after & mask == val & mask) if mask else after == val):
                return True
            await asyncio.sleep(0.25)
        return False

    async def get_reg_value(self, addr):
        self.read_event.clear()
        try:
            await self.client.write_gatt_char(CMD_UUID, bytes([addr]), response=False)
            await asyncio.wait_for(self.read_event.wait(), timeout=1.0)
            return (self.last_read_data[1]<<8) | self.last_read_data[2] if self.last_read_data and self.last_read_data[0] == addr else None
        except:
            return None

    async def apply_stage_settings(self, stage):
        hive_state['users'][self.slot]['verified'] = False
        sps, pga, mux, dc = self.presets[stage]
        try:
            self.worker.log_msg.emit(f"🛠 [{self.slot}] Stage {stage}...")
            await self.write_and_verify(0x08, dc, 0x000F, "DC")
            await self.write_and_verify(0x03, sps << 2, 0x001C, "SPS")
            pga_val = (pga<<12)|(pga<<8)|(pga<<4)|pga
            await self.write_and_verify(0x04, pga_val, desc="PGA1")
            await self.write_and_verify(0x05, pga_val, desc="PGA2")
            for i in range(8):
                await self.write_and_verify(0x09 + i*5, mux, 0x0003, f"MUX{i}")
            self.current_stage = stage
            hive_state['users'][self.slot]['stage'] = stage
            gain_real = {0:1,2:4,3:8,4:16,5:32}[pga]
            hive_state['users'][self.slot]['uv_scale'] = (1.2 / gain_real / 8388607.0) * 1e6
            hive_state['users'][self.slot]['verified'] = True
            self.worker.log_msg.emit(f"✅ [{self.slot}] Stage {stage} OK")
        except Exception as e:
            self.worker.log_msg.emit(f"❌ [{self.slot}] {e}")

    async def connect_and_loop(self):
        try:
            async with BleakClient(self.device, timeout=15.0) as client:
                self.client = client
                self.is_connected = True
                self.is_connecting = False
                hive_state['users'][self.slot].update({'active': True, 'mac': self.mac})
                await client.start_notify(CMD_UUID, self.cmd_callback)
                self.worker.log_msg.emit(f"📡[{self.slot}] Подключён")
                await self.apply_stage_settings(self.target_stage)
                await client.start_notify(DATA_UUID, self.data_rx_handler)
                while client.is_connected:
                    if self.current_stage != self.target_stage:
                        await self.apply_stage_settings(self.target_stage)
                    await asyncio.sleep(0.5)
        finally:
            self.is_connected = self.is_connecting = False
            hive_state['users'][self.slot]['active'] = False
            self.worker.log_msg.emit(f"⚠️ [{self.slot}] Отключён")

    def data_rx_handler(self, s, data):
        if len(data) >= 33 and data[0] == 0xA0 and data[32] == 0xC0:
            uv = hive_state['users'][self.slot].get('uv_scale', 1.0)
            sample = np.zeros(CHANNELS, dtype=np.float32)
            for i in range(CHANNELS):
                start = 2 + i * 3
                val = (data[start] << 16) | (data[start+1] << 8) | data[start+2]
                if val & 0x800000:
                    val -= 0x1000000
                sample[i] = val * uv
            t = self.worker.data_tensor
            idx = self.slot * CHANNELS
            t[idx:idx+CHANNELS, :] = torch.roll(t[idx:idx+CHANNELS, :], -1, dims=1)
            t[idx:idx+CHANNELS, -1] = torch.tensor(sample, dtype=torch.float32, device=DEVICE)


class GpuDspWorker(QThread):
    ui_update = pyqtSignal(dict)
    def __init__(self, data_tensor):
        super().__init__()
        self.data_tensor = data_tensor

    def run(self):
        while True:
            ui_data = {}
            with torch.no_grad():
                for i in range(MAX_DEVICES):
                    u = hive_state['users'][i]
                    if not u['active'] or not u['verified']: continue
                    raw = self.data_tensor[i*CHANNELS:(i+1)*CHANNELS, :]
                    centered = raw - torch.mean(raw, dim=1, keepdim=True)
                    vec_2d = torch.matmul(SPATIAL_GPU, centered)
                    glob = torch.mean(centered, dim=0)

                    a_d, f_d, p_d = self.extract(glob, 1.0, 4.0)
                    a_t, f_t, p_t = self.extract(glob, 4.0, 8.0)
                    a_a, _, _ = self.extract(vec_2d[0], 8.0, 13.0)
                    a_g, _, _ = self.extract(glob, 30.0, 48.0)

                    u.update({
                        'delta_phase': p_d, 'theta_phase': p_t,
                        'delta_freq': f_d, 'delta_amp': a_d,
                        'theta_amp': a_t, 'alpha_amp': a_a, 'gamma_amp': a_g,
                        'lead_x': vec_2d[0,-1].item(), 'lead_y': vec_2d[1,-1].item()
                    })
                    ui_data[i] = {'delta': a_d, 'alpha': a_a, 'gamma': a_g}

            self.ui_update.emit(ui_data)
            time.sleep(0.018)

    def extract(self, sig, low, high):
        N = sig.shape[-1]
        fft = torch.fft.fft(sig)
        freqs = torch.fft.fftfreq(N, 1/EEG_FS, device=DEVICE)
        mask = (freqs >= low) & (freqs <= high)
        an = torch.zeros_like(fft)
        an[mask] = fft[mask] * 2.0
        analytic = torch.fft.ifft(an)
        amp = torch.mean(torch.abs(analytic[-150:])).item()
        phase = torch.angle(analytic[-1]).item()
        inst_freq = torch.mean((torch.angle(analytic[1:]*torch.conj(analytic[:-1])) * (EEG_FS/(2*np.pi)))[-100:]).item()
        return amp, inst_freq, phase


class BleFleetWorker(QThread):
    log_msg = pyqtSignal(str)
    def __init__(self, data_tensor):
        super().__init__()
        self.data_tensor = data_tensor
        self.managers =[None] * MAX_DEVICES

    def run(self):
        asyncio.run(self.fleet_loop())

    def set_device_stage(self, slot_id, stage):
        if self.managers[slot_id]:
            self.managers[slot_id].target_stage = stage

    async def fleet_loop(self):
        while True:
            try:
                devices = await BleakScanner.discover(timeout=2.0, service_uuids=[SERVICE_UUID])
                for d in devices:
                    if not any(m and m.mac == d.address for m in self.managers):
                        for i in range(MAX_DEVICES):
                            if not (self.managers[i] and (self.managers[i].is_connected or self.managers[i].is_connecting)):
                                self.log_msg.emit(f"🔎 Найден {d.address} → слот {i}")
                                mgr = SingleDeviceManager(d, i, self)
                                self.managers[i] = mgr
                                asyncio.create_task(mgr.connect_and_loop())
                                break
            except:
                pass
            await asyncio.sleep(2)


class SlotWidget(QFrame):
    def __init__(self, slot_id, fleet):
        super().__init__()
        self.slot_id = slot_id
        self.fleet = fleet
        self.setStyleSheet("background:#111; border-radius:8px; border:2px solid #333;")
        layout = QVBoxLayout(self)
        self.lbl_title = QLabel(f"СЛОТ {slot_id} [OFFLINE]")
        self.lbl_title.setStyleSheet("font-weight:bold; color:#0f0;")
        layout.addWidget(self.lbl_title)
        self.lbl_status = QLabel("—")
        self.lbl_status.setStyleSheet("font-family:monospace; color:#aaa;")
        layout.addWidget(self.lbl_status)

    def update_data(self, u, dsp):
        if not u['active']:
            self.lbl_title.setText(f"СЛОТ {self.slot_id} [OFFLINE]")
            self.lbl_status.setText("—")
            return
        self.lbl_title.setText(f"СЛОТ {self.slot_id} [{u['mac'][-5:]}]")
        self.lbl_status.setText(f"Δ {u['delta_amp']:.2f}  α {u['alpha_amp']:.2f}  γ {u['gamma_amp']:.2f}")


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("OPEN-AIR HIVE PSYTRANCE — Motor Imagery Sync")
        self.resize(1400, 900)
        self.setStyleSheet("background-color:#050505; color:white;")

        self.data_tensor = torch.zeros((MAX_DEVICES*CHANNELS, BUFFER_SIZE), dtype=torch.float32, device=DEVICE)
        self.fleet = BleFleetWorker(self.data_tensor)
        self.fleet.log_msg.connect(self.log_msg)
        self.fleet.start()

        self.dsp = GpuDspWorker(self.data_tensor)
        self.dsp.ui_update.connect(self.update_ui)
        self.dsp.start()

        self.audio = HiveAudioEngine()
        
        # Двунаправленный поток: Слушаем микрофон/гитару/барабаны (вход), выводим стерео
        self.stream = sd.Stream(channels=(2, 2), samplerate=AUDIO_FS,
                                callback=self.audio.process_audio, blocksize=512)
        self.stream.start()

        self.init_ui()

    def init_ui(self):
        cw = QWidget()
        self.setCentralWidget(cw)
        layout = QVBoxLayout(cw)

        self.log_area = QTextEdit()
        self.log_area.setMaximumHeight(140)
        self.log_area.setStyleSheet("background:#000; color:#0f0;")
        layout.addWidget(self.log_area)

        grid = QGridLayout()
        self.slots =[SlotWidget(i, self.fleet) for i in range(MAX_DEVICES)]
        for i, s in enumerate(self.slots):
            grid.addWidget(s, i//4, i%4)
        layout.addLayout(grid)

        self.lbl_global = QLabel("BPM: —   |   STYLE: COLLECTIVE PSY")
        self.lbl_global.setStyleSheet("font-size:18px; font-weight:bold; color:#0ff;")
        layout.addWidget(self.lbl_global)

    def log_msg(self, text):
        self.log_area.append(text)

    def update_ui(self, ui_data):
        active_count = 0
        total_delta_freq = 0.0

        for i, d in ui_data.items():
            u = hive_state['users'][i]
            self.slots[i].update_data(u, d)
            if u['active']:
                active_count += 1
                total_delta_freq += u.get('delta_freq', 2.0)

        if active_count > 0:
            bpm = (total_delta_freq / active_count) * 60
            self.lbl_global.setText(f"BPM: {bpm:.0f}   |   {active_count} ЧЕЛОВЕК В ХАЙВЕ   |   MOTOR IMAGERY SYNC")
        else:
            self.lbl_global.setText("BPM: —   |   0 ЧЕЛОВЕК В ХАЙВЕ   |   WAITING...")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
