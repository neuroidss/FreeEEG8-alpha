import numpy as np
import torch
import sounddevice as sd
import threading
import time
from evdev import InputDevice, list_devices, ecodes
from collections import deque

# --- КОНФИГУРАЦИЯ ---
FS = 44100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_STRINGS = 6
OCTAVES_TOTAL = 4
BATCH_SIZE = 256

class NeuroLockV16:
    def __init__(self):
        self.device_path = self.find_sony_pad()
        self.gamepad = InputDevice(self.device_path)
        
        # GPU ресурсы
        self.phases = torch.zeros((NUM_STRINGS, OCTAVES_TOTAL), device=DEVICE)
        self.sub_phases = torch.zeros(OCTAVES_TOTAL, device=DEVICE)
        self.zap_phases = torch.zeros(OCTAVES_TOTAL, device=DEVICE)
        self.kick_phases = torch.zeros(OCTAVES_TOTAL, device=DEVICE)
        
        self.ratios = torch.tensor([1.0, 1.33, 1.78, 2.37, 2.84, 3.75], device=DEVICE)
        self.amps = torch.zeros(NUM_STRINGS, device=DEVICE)
        
        self.delay_max = FS * 3
        self.delay_buffer = torch.zeros((2, self.delay_max), device=DEVICE)
        self.delay_ptr = 0
        
        # Флаги управления
        self.drums_held = False
        self.lock_held = False

        self.state = {
            "root_pos": 0.0, "minor": 1.0, "sept": 0.0,
            "sustain": 0.99, "feedback": 0.0, "warp": 1.0,
            "delay_samples": int(FS * 0.5),
            "rhythm_locked": False, # ФИЧА: Блокировка темпа
            "noise_l": 0.0, "noise_r": 0.0, "acid": 1.2,
            "zap": 0.0, "sub": 0.0, "drums": False, "glitch": False,
            "last_strum": time.time(), "start_time": time.time(), "is_running": True
        }
        
        self.strum_history = deque([0.5, 0.5], maxlen=4)
        self.fired_down = [False]*6; self.fired_up = [False]*6

    def find_sony_pad(self):
        for dev in [InputDevice(path) for path in list_devices()]:
            if "sony" in dev.name.lower() or "wireless controller" in dev.name.lower():
                print(f"✅ Neuro-Guitar V16 Linked: {dev.name}")
                return dev.path
        raise Exception("Геймпад не найден!")

    def input_worker(self):
        print("🎮 UP: Drums | DOWN: Lock Rhythm | L/R: Noise")
        for event in self.gamepad.read_loop():
            if event.type == ecodes.EV_ABS:
                v = event.value
                if event.code == ecodes.ABS_X or event.code == ecodes.ABS_Y:
                    lx = (self.gamepad.absinfo(ecodes.ABS_X).value - 128)/128.0
                    ly = (self.gamepad.absinfo(ecodes.ABS_Y).value - 128)/128.0
                    if (lx**2 + ly**2) > 0.1:
                        self.state["root_pos"] = (np.arctan2(lx, -ly) + np.pi) / (2 * np.pi)
                elif event.code == ecodes.ABS_RX: self.state["warp"] = 0.85 + (v/255.0)*0.3
                elif event.code == ecodes.ABS_RY: self.process_strum(-(v-128)/128.0)
                elif event.code == ecodes.ABS_Z: self.state["sept"] = v / 255.0
                elif event.code == ecodes.ABS_RZ: 
                    self.state["feedback"] = (v/255.0)*0.85; self.state["sustain"] = 0.992 + (v/255.0)*0.0075
                
                # D-PAD Логика
                elif event.code == ecodes.ABS_HAT0Y: # Вертикаль (Up/Down)
                    if v == -1 and not self.drums_held: # Нажали Вверх
                        self.state["drums"] = not self.state["drums"]
                        self.drums_held = True
                        print(f"🥁 Drums: {'ON' if self.state['drums'] else 'OFF'}")
                    elif v == 1 and not self.lock_held: # Нажали Вниз
                        self.state["rhythm_locked"] = not self.state["rhythm_locked"]
                        self.lock_held = True
                        status = "LOCKED 🔒" if self.state["rhythm_locked"] else "ADAPTIVE 🔄"
                        print(f"⏱ Rhythm: {status}")
                    elif v == 0: 
                        self.drums_held = False
                        self.lock_held = False
                
                elif event.code == ecodes.ABS_HAT0X: # Горизонталь (Noise)
                    self.state["noise_l"] = 0.5 if v == -1 else 0.0
                    self.state["noise_r"] = 0.5 if v == 1 else 0.0

            elif event.type == ecodes.EV_KEY:
                st = bool(event.value)
                if event.code == 304: self.state["sub"] = 0.8 if st else 0.0
                if event.code == 307: self.state["zap"] = 1.0 if st else 0.0
                if event.code == 308: self.state["glitch"] = st
                if event.code == 305: self.state["acid"] = 8.0 if st else 1.2
                if event.code == 310: self.state["minor"] = 0.89 if st else 1.0
                if event.code == 311: self.state["acid"] = 4.0 if st else 1.2

    def process_strum(self, ry):
        if abs(ry) < 0.1:
            for i in range(6): self.fired_down[i] = self.fired_up[i] = False
        t_now = time.time()
        th = torch.linspace(0.12, 0.85, 6)
        trig = False
        if ry > 0.1:
            for i in range(6):
                if ry > th[i] and not self.fired_down[i]:
                    self.amps[i] = 2.0; self.fired_down[i] = True; trig = True
        elif ry < -0.1:
            for i in range(6):
                if ry < -th[i] and not self.fired_up[i]:
                    self.amps[5-i] = 1.8; self.fired_up[i] = True; trig = True
        
        # Обновляем темп только если ритм НЕ заблокирован
        if trig and not self.state["rhythm_locked"] and (t_now - self.state["last_strum"]) > 0.08:
            interval = t_now - self.state["last_strum"]
            if 0.1 < interval < 1.5:
                self.strum_history.append(interval)
                self.state["delay_samples"] = int(np.mean(self.strum_history) * FS)
            self.state["last_strum"] = t_now

    def audio_callback(self, outdata, frames, time_info, status):
        with torch.no_grad():
            t = torch.arange(frames, device=DEVICE)
            pos = self.state["root_pos"]
            base_f = 41.20 * (2.0 ** pos)
            oct_idx = torch.arange(OCTAVES_TOTAL, device=DEVICE)
            weights = (torch.sin(np.pi * (oct_idx + pos) / OCTAVES_TOTAL)**2).view(1, -1, 1)
            
            # --- ГИТАРА ---
            f_guitar = base_f * self.ratios
            f_guitar[3] *= self.state["minor"]
            f_guitar[5] *= (1.0 * (1-self.state["sept"]) + 0.89 * self.state["sept"])
            steps_g = 2 * np.pi * (f_guitar.unsqueeze(1) * (2.0 ** oct_idx)) / FS
            p_g = self.phases.unsqueeze(2) + steps_g.unsqueeze(2) * t
            wave = (((p_g % (2 * np.pi)) / np.pi - 1.0) * weights * (self.amps.view(-1,1,1) * torch.pow(self.state["sustain"], t))).sum(dim=(0,1))
            self.phases = (p_g[:,:,-1] + steps_g.mean()) % (2 * np.pi)

            # --- BASS (Cross) ---
            if self.state["sub"] > 0:
                steps_s = 2 * np.pi * ((base_f * 0.5) * (2.0 ** oct_idx)) / FS
                p_s = self.sub_phases.unsqueeze(1) + steps_s.unsqueeze(1) * t
                wave += (torch.sin(p_s) * weights.squeeze(0) * self.state["sub"]).sum(dim=0)
                self.sub_phases = (p_s[:,-1] + steps_s) % (2 * np.pi)

            # --- ZAP (Triangle) ---
            if self.state["zap"] > 0:
                z_env = torch.exp(-t / (FS * 0.05))
                steps_z = 2 * np.pi * ((base_f * 20.0 * z_env + 100.0).unsqueeze(0) * (2.0 ** oct_idx).unsqueeze(1)) / FS
                p_z = self.zap_phases.unsqueeze(1) + torch.cumsum(steps_z, dim=1)
                wave += (torch.sin(p_z) * weights.squeeze(0) * 0.6 * self.state["zap"]).sum(dim=0)
                self.zap_phases = (p_z[:,-1] + steps_z[:,-1]) % (2 * np.pi)

            # --- DRUMS ---
            if self.state["drums"]:
                beat_s = float(self.state["delay_samples"])
                p_beat = ((self.delay_ptr + t) % beat_s) / beat_s
                k_env = torch.exp(-p_beat * 15.0)
                steps_k = 2 * np.pi * ((k_env * 150.0 + base_f).unsqueeze(0) * (2.0 ** oct_idx).unsqueeze(1)) / FS
                p_k = self.kick_phases.unsqueeze(1) + torch.cumsum(steps_k, dim=1)
                wave += (torch.sin(p_k) * weights.squeeze(0) * k_env * 1.5).sum(dim=0)
                self.kick_phases = (p_k[:,-1] + steps_k[:,-1]) % (2 * np.pi)
                h_env = torch.exp(-(( (self.delay_ptr + t) % (beat_s/4) ) / (beat_s/4)) * 30.0)
                wave += torch.randn(frames, device=DEVICE) * h_env * 0.2

            # --- NOISE ---
            if self.state["noise_l"] > 0 or self.state["noise_r"] > 0:
                noise_sig = torch.randn(frames, device=DEVICE)
                if self.state["noise_l"] > 0: wave += torch.cumsum(noise_sig * 0.01, dim=0) * self.state["noise_l"]
                if self.state["noise_r"] > 0: wave += noise_sig * 0.05 * self.state["noise_r"]

            wave = torch.tanh(wave * self.state["acid"])

            # --- DELAY ---
            delay_t = int(self.state["delay_samples"] * self.state["warp"])
            delay_t = max(256, min(self.delay_max - frames, delay_t))
            idx = (self.delay_ptr - delay_t + t) % self.delay_max
            echo = self.delay_buffer[0, idx]
            self.delay_buffer[0, (self.delay_ptr + t) % self.delay_max] = wave + echo * self.state["feedback"]
            wave_out = wave + echo * 0.4
            if self.state["glitch"]: wave_out = torch.roll(wave_out, frames // 4)
            self.delay_ptr = (self.delay_ptr + frames) % self.delay_max
            self.amps *= (self.state["sustain"] ** frames)
            outdata[:] = wave_out.repeat(2, 1).cpu().numpy().T * 0.3

    def run(self):
        threading.Thread(target=self.input_worker, daemon=True).start()
        with sd.OutputStream(channels=2, callback=self.audio_callback, samplerate=FS, blocksize=BATCH_SIZE):
            print("\n--- 🎸 PSY-GUITAR V16: RHYTHM LOCK ---")
            print("D-PAD UP: Drums ON/OFF")
            print("D-PAD DOWN: LOCK/UNLOCK Rhythm (Freeze tempo)")
            print("D-PAD L/R: Cosmic Noise")
            while True: time.sleep(1)

if __name__ == "__main__":
    NeuroLockV16().run()
