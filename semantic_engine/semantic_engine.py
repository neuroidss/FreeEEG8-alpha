#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEURO-WORLD v39.0: PURE NEUROFEEDBACK
- НИКАКОГО АВТО-ДВИЖЕНИЯ: Мир полностью статичен без сигнала мозга.
- НИКАКОГО QWEN в цикле рендера: Только базовый промпт + 56-D вектор из мозга.
- Qwen работает в фоне как "угадыватель", не влияя на картинку.
- Идеальная среда для обучения мозга: только ты и твой интент.
"""

import sys
import asyncio
import threading
import time
import traceback
import numpy as np
import torch
import cv2
from PyQt6.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QTextEdit
from PyQt6.QtCore import pyqtSignal, QObject, Qt, QTimer
from PyQt6.QtGui import QPixmap, QImage
from bleak import BleakScanner, BleakClient
from scipy.signal import butter, sosfiltfilt, hilbert, iirnotch, lfilter
from PIL import Image

from diffusers import StableDiffusionImg2ImgPipeline, LCMScheduler, AutoencoderTiny
from transformers import AutoProcessor, AutoModelForCausalLM

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================
DEVICE = torch.device('cuda')
SD_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DTYPE = torch.float16 

W, H = 512, 384
LCM_STEPS = 3
EEG_FS = 250.0

UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6 
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()

# =============================================================================
# ГЛОБАЛЬНЫЙ ДЕБАГГЕР
# =============================================================================
class DebugLogger(QObject):
    log_signal = pyqtSignal(str, str)

logger = DebugLogger()
def update_log(key, val):
    logger.log_signal.emit(key, str(val))

# =============================================================================
# EEG PROCESSOR
# =============================================================================
class EEGProcessor(QObject, threading.Thread):
    def __init__(self):
        super().__init__()
        threading.Thread.__init__(self, daemon=True)
        self.buffer = np.zeros((8, 500), dtype=np.float32)
        self.lock = threading.Lock()
        self.latest_vector = torch.zeros(56, device=DEVICE, dtype=DTYPE)
        
        self.b50, self.a50 = iirnotch(50.0, 30.0, fs=EEG_FS)
        self.b100, self.a100 = iirnotch(100.0, 30.0, fs=EEG_FS)
        
        self.theta = butter(3,[4, 8], btype='band', fs=EEG_FS, output='sos')
        self.slow_gamma = butter(4,[31, 51], btype='band', fs=EEG_FS, output='sos')
        self.fast_gamma = butter(4, [61, 102], btype='band', fs=EEG_FS, output='sos')
        self.pairs =[(i, j) for i in range(8) for j in range(i + 1, 8)]
        
        self.packet_counter = 0
        self.managed_addresses = {}

    def _notification_handler(self, sender, data):
        if len(data) >= 33 and data[0] == 0xA0 and data[32] == 0xC0:
            self.packet_counter += 1
            if self.packet_counter == 1:
                update_log("BLE", "🚀 Пакеты идут")
            if self.packet_counter % 10 == 0:
                update_log("PACKETS", str(self.packet_counter))
                
            sample = np.zeros(8, dtype=np.float32)
            for i in range(8):
                start = 2 + i * 3
                val = (data[start] << 16) | (data[start+1] << 8) | data[start+2]
                if val & 0x800000: val -= 0x1000000
                sample[i] = val * UV_SCALE
                
            with self.lock:
                self.buffer = np.roll(self.buffer, -1, axis=1)
                self.buffer[:, -1] = sample

    async def device_manager(self, device):
        addr = device.address
        update_log("BLE", f"🔌 Закреплен: {device.name or addr}")
        while True:
            try:
                async with BleakClient(device, timeout=10.0) as client:
                    update_log("BLE", f"✅ ПОДКЛЮЧЕН: {device.name or addr}")
                    self.packet_counter = 0
                    await client.start_notify(DATA_UUID, self._notification_handler)
                    while client.is_connected:
                        await asyncio.sleep(1)
                    update_log("BLE", f"⚠️ ПОТЕРЯ СВЯЗИ: {device.name or addr}")
            except Exception: pass
            await asyncio.sleep(3)

    async def main_scanner(self):
        while True:
            update_log("BLE", "🔍 Скан...")
            try:
                devices = await BleakScanner.discover(timeout=3.0, service_uuids=[SERVICE_UUID])
                for d in devices:
                    if d.address not in self.managed_addresses:
                        self.managed_addresses[d.address] = True
                        asyncio.create_task(self.device_manager(d))
                        break 
            except Exception: pass
            await asyncio.sleep(5)

    def run(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main_scanner())

    def process_cycle(self):
        with self.lock:
            if not np.any(self.buffer[:, -40:]): return
            raw_data = self.buffer.copy()

        try:
            clean_50 = lfilter(self.b50, self.a50, raw_data, axis=1)
            clean_100 = lfilter(self.b100, self.a100, clean_50, axis=1)

            theta_data = sosfiltfilt(self.theta, clean_100, axis=1)
            slow_gamma_all = sosfiltfilt(self.slow_gamma, clean_100, axis=1)
            fast_gamma_all = sosfiltfilt(self.fast_gamma, clean_100, axis=1)

            global_theta = np.mean(theta_data, axis=0)
            window = global_theta[-125:]
            crossings = np.where((window[:-1] < 0) & (window[1:] >= 0))[0]
            
            rhythm_pct = 0.0
            if len(crossings) >= 3:
                periods = np.diff(crossings)
                mean_p = np.mean(periods)
                if mean_p > 0:
                    std_p = np.std(periods)
                    rhythm_pct = max(0.0, 100.0 - (std_p / mean_p) * 200.0)

            update_log("RHYTHM", f"{rhythm_pct:.0f}%")

            t_phase = np.angle(hilbert(global_theta))
            phase_40 = t_phase[-40:]
            past_idx = np.where(phase_40 < 0)[0]
            future_idx = np.where(phase_40 >= 0)[0]

            h_slow_40 = hilbert(slow_gamma_all, axis=1)[:, -40:]
            h_fast_40 = hilbert(fast_gamma_all, axis=1)[:, -40:]

            slow_ciplv = np.zeros(28)
            fast_ciplv = np.zeros(28)

            if len(past_idx) > 2 and len(future_idx) > 2:
                past_data = h_slow_40[:, past_idx]
                future_data = h_fast_40[:, future_idx]
                for idx, (p1, p2) in enumerate(self.pairs):
                    slow_ciplv[idx] = np.abs(np.imag(np.mean(past_data[p1] * np.conj(past_data[p2]))))
                    fast_ciplv[idx] = np.abs(np.imag(np.mean(future_data[p1] * np.conj(future_data[p2]))))

            wm_vector = np.concatenate([slow_ciplv, fast_ciplv])
            v_max = np.max(wm_vector)
            if v_max > 0: wm_vector = wm_vector / v_max

            wm_vector = wm_vector * (rhythm_pct / 100.0)

            with self.lock:
                self.latest_vector = torch.from_numpy(wm_vector).to(device=DEVICE, dtype=DTYPE)

        except Exception as e:
            update_log("EEG_ERR", str(e)[:20])

    def get_latest_vector(self):
        with self.lock:
            return self.latest_vector.clone()

# =============================================================================
# МОДЕЛЬ МИРА (ПАССИВНЫЙ НАБЛЮДАТЕЛЬ QWEN)
# =============================================================================
class WorldObserver(threading.Thread):
    def __init__(self, q_model, q_proc):
        super().__init__(daemon=True)
        self.model = q_model
        self.proc = q_proc
        self.current_image = None
        self.lock = threading.Lock()
        self.running = True

    def update_image(self, img_pil):
        with self.lock:
            self.current_image = img_pil.copy()

    def run(self):
        while self.running:
            img = None
            with self.lock:
                if self.current_image is not None:
                    img = self.current_image.copy()
            if img is None:
                time.sleep(0.5)
                continue

            try:
                messages =[
                    {"role": "user", "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Describe this image in 3 keywords."},
                    ]}
                ]
                text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                inputs = self.proc(text=[text], images=[img], padding=True, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    out_ids = self.model.generate(**inputs, max_new_tokens=6)
                    desc = self.proc.batch_decode(out_ids, skip_special_tokens=True)[0]
                        
                if len(desc) > 2:
                    clean_desc = "".join(c for c in desc if c.isalnum() or c in " ,").strip()
                    update_log("QWEN_VIEW", clean_desc)
                    
            except Exception: pass
            time.sleep(0.5)

# =============================================================================
# ВИЗУАЛЬНЫЙ ДВИЖОК (ТОЛЬКО НЕЙРОФИДБЕК)
# =============================================================================
class VisualEngine(threading.Thread):
    def __init__(self, pipe, eeg_proc, world_observer):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.eeg = eeg_proc
        self.world_observer = world_observer
        self.running = True
        
        self.current_pil = Image.fromarray(np.random.randint(50, 100, (H, W, 3), dtype=np.uint8))
        
        torch.manual_seed(42)
        proj_tmp = torch.empty(56, 768, dtype=torch.float32)
        self.sd_proj = torch.nn.init.orthogonal_(proj_tmp).to(device=DEVICE, dtype=DTYPE) * 2.0

        self.base_prompt = "fractals"
        self.base_embeds = None
        self.signal_obj = None

    def run(self):
        # Однократно кодируем базовый промпт
        with torch.no_grad():
            toks = self.pipe.tokenizer(self.base_prompt, return_tensors="pt", padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True).input_ids.to(DEVICE)
            self.base_embeds = self.pipe.text_encoder(toks)[0]

        while self.running:
            start_t = time.time()
            brain_vec = self.eeg.get_latest_vector()
            frame = self.current_pil.copy()
            intent_mag = torch.max(brain_vec).item()
            update_log("INTENT", f"{intent_mag:.2f}")

            try:
                with torch.no_grad():
                    # 1. Инъекция ЭЭГ
                    brain_sd_embed = (brain_vec @ self.sd_proj).unsqueeze(0).unsqueeze(0)
                    final_prompt_embeds = self.base_embeds + (brain_sd_embed * 1.5)

                    # 2. Физика стояния на месте (только микро-шум)
                    img_np = np.array(frame)
                    noise = np.random.randint(-3, 3, (H, W, 3), dtype=np.int16)
                    warped_np = np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                    warped = Image.fromarray(warped_np)

                    # 3. Динамическая сила (от 0.35 до 0.65)
                    strength = 0.35 + (intent_mag * 0.30)
                    min_safe = (1.0 / LCM_STEPS) + 0.02
                    safe_strength = max(strength, min_safe)

                    gen = self.pipe(
                        prompt_embeds=final_prompt_embeds,
                        image=warped,
                        strength=safe_strength,
                        num_inference_steps=LCM_STEPS,
                        guidance_scale=1.0,
                        output_type="pil"
                    ).images[0]
                    
                    self.current_pil = gen
                    if self.signal_obj:
                        self.signal_obj.emit(self.current_pil)
                    
                    self.world_observer.update_image(self.current_pil)
                    
                    fps = 1.0 / (time.time() - start_t)
                    update_log("SD_FPS", f"{fps:.1f}")
                    
            except Exception as e:
                update_log("SD_ERR", str(e))
            
            elapsed = time.time() - start_t
            if elapsed < 0.12: time.sleep(0.12 - elapsed)

class VisualSignalWrapper(QObject):
    image_ready = pyqtSignal(object)

# =============================================================================
# GUI
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEURO-WORLD: PURE NEUROFEEDBACK")
        self.resize(800, 700)
        self.setStyleSheet("background-color:#050505; color: #0f0;")
        
        cw = QWidget()
        layout = QVBoxLayout(cw)
        self.setCentralWidget(cw)
        
        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas)
        
        self.debug_text = QTextEdit()
        self.debug_text.setReadOnly(True)
        self.debug_text.setFixedHeight(100)
        self.debug_text.setStyleSheet("background-color:#111; color:#0ff; font-family:monospace; font-size:14px; border:1px solid #333;")
        layout.addWidget(self.debug_text)
        
        self.log_data = {
            "BLE": "Ожидание...", "PACKETS": "0", "RHYTHM": "0%", 
            "INTENT": "0.00", "SD_FPS": "0.0 FPS", "QWEN_VIEW": "..."
        }
        
        logger.log_signal.connect(self.on_log_update)
        self.init_core()

    def on_log_update(self, key, val):
        self.log_data[key] = val
        lines = [
            f"[BLE] {self.log_data['BLE']} | [PACKETS] {self.log_data['PACKETS']} | [RHYTHM] {self.log_data['RHYTHM']} | [INTENT] {self.log_data['INTENT']} | [SD_FPS] {self.log_data['SD_FPS']}",
            f"[QWEN] 👀 {self.log_data['QWEN_VIEW']}"
        ]
        self.debug_text.setPlainText("\n".join(lines))

    def init_core(self):
        self.eeg_timer = QTimer()
        self.eeg_timer.timeout.connect(self.process_eeg)
        self.eeg_timer.start(160)
        threading.Thread(target=self._load_models, daemon=True).start()

    def process_eeg(self):
        if hasattr(self, 'eeg') and self.eeg.is_alive():
            self.eeg.process_cycle()

    def _load_models(self):
        try:
            q_model = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True)
            q_proc = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
            
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DTYPE).to(DEVICE)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

            self.eeg = EEGProcessor()
            self.world_observer = WorldObserver(q_model, q_proc)
            self.viz = VisualEngine(pipe, self.eeg, self.world_observer)
            
            self.viz_signal = VisualSignalWrapper()
            self.viz.signal_obj = self.viz_signal.image_ready
            self.viz_signal.image_ready.connect(self.draw)

            self.world_observer.start()
            self.eeg.start()
            self.viz.start()
            
        except Exception:
            traceback.print_exc()

    def draw(self, img):
        q = QImage(img.tobytes("raw", "RGB"), W, H, QImage.Format.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(q).scaled(self.canvas.width(), self.canvas.height(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
