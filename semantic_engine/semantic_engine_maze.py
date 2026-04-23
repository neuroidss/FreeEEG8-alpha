#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEURO-WORLD v42.4: STABLE LATENT MAZE
- ОСНОВА: 100% рабочий v39.0.
- ЛАБИРИНТ: 768-D путь, который не движется без сигнала.
- СТАТИКА: Если нет интента, мир застывает на текущем смысле (strength=0.35).
- ФИКС: Исправлена ошибка '0it' и черного экрана (минимальный порог шагов).
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
# 🧠 ЛАБИРИНТ СМЫСЛОВ В 768-D (БЕЗ АВТО-ДВИЖЕНИЯ)
# =============================================================================
class LatentMaze768D:
    def __init__(self, vector_dim=768, device='cuda', size=5, difficulty=0.8):
        self.dim = vector_dim
        self.device = torch.device(device)
        self.size = size
        self.difficulty = difficulty
        self.path_nodes = []
        self._generate_path()
        self.player_pos = np.copy(self.path_nodes[0])
        self.current_segment_idx = 0
        self.is_finished = False

    def _normalize(self, v):
        norm = np.linalg.norm(v)
        return v / norm if norm > 1e-6 else v

    def _generate_path(self):
        node = self._normalize(np.random.normal(0, 1, self.dim))
        self.path_nodes.append(node)
        for i in range(self.size):
            drift = self._normalize(np.random.normal(0, 1, self.dim))
            drift -= np.dot(drift, node) * node
            drift = self._normalize(drift)
            node = self._normalize(node * np.cos(self.difficulty) + drift * np.sin(self.difficulty))
            self.path_nodes.append(node)
            
    def update(self, brain_thrust_768d, dt=0.1):
        if self.is_finished or brain_thrust_768d is None:
            return torch.from_numpy(self.player_pos).to(self.device), 0.0
        
        curr_n = self.path_nodes[self.current_segment_idx]
        next_n = self.path_nodes[self.current_segment_idx + 1]
        path_dir = self._normalize(next_n - curr_n)
        
        # Насколько сильно юзер "давит" в сторону следующего чекпоинта
        alignment = np.dot(self._normalize(brain_thrust_768d), path_dir)
        speed = max(0, alignment) * dt
        
        # Двигаем позицию
        self.player_pos += path_dir * speed
        
        # Проверка чекпоинта
        if np.dot(next_n - self.player_pos, path_dir) < 0:
            self.current_segment_idx += 1
            self.player_pos = np.copy(next_n)
            if self.current_segment_idx >= self.size: self.is_finished = True
        
        return torch.from_numpy(self.player_pos).to(self.device), speed

# =============================================================================
# КОНФИГУРАЦИЯ
# =============================================================================
DEVICE = torch.device('cuda')
SD_MODEL_ID = "SimianLuo/LCM_Dreamshaper_v7"
QWEN_MODEL_ID = "Qwen/Qwen3.5-0.8B"
DTYPE = torch.float16 

W, H = 512, 384
LCM_STEPS = 6
EEG_FS = 250.0
UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6 
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()

class DebugLogger(QObject):
    log_signal = pyqtSignal(str, str)
logger = DebugLogger()
def update_log(key, val): logger.log_signal.emit(key, str(val))

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
            sample = np.zeros(8, dtype=np.float32)
            for i in range(8):
                start = 2 + i * 3
                val = (data[start] << 16) | (data[start+1] << 8) | data[start+2]
                if val & 0x800000: val -= 0x1000000
                sample[i] = val * UV_SCALE
            with self.lock:
                self.buffer = np.roll(self.buffer, -1, axis=1)
                self.buffer[:, -1] = sample

    async def main_scanner(self):
        while True:
            try:
                devices = await BleakScanner.discover(timeout=3.0, service_uuids=[SERVICE_UUID])
                for d in devices:
                    if d.address not in self.managed_addresses:
                        self.managed_addresses[d.address] = True
                        asyncio.create_task(self.device_manager(d))
            except Exception: pass
            await asyncio.sleep(5)

    async def device_manager(self, device):
        while True:
            try:
                async with BleakClient(device, timeout=10.0) as client:
                    await client.start_notify(DATA_UUID, self._notification_handler)
                    while client.is_connected: await asyncio.sleep(1)
            except Exception: pass
            await asyncio.sleep(3)

    def run(self):
        loop = asyncio.new_event_loop(); asyncio.set_event_loop(loop)
        loop.run_until_complete(self.main_scanner())

    def process_cycle(self):
        with self.lock:
            if not np.any(self.buffer[:, -40:]): return
            raw_data = self.buffer.copy()
        try:
            clean = lfilter(self.b50, self.a50, raw_data, axis=1)
            clean = lfilter(self.b100, self.a100, clean, axis=1)
            theta_data = sosfiltfilt(self.theta, clean, axis=1)
            sg_all = sosfiltfilt(self.slow_gamma, clean, axis=1)
            fg_all = sosfiltfilt(self.fast_gamma, clean, axis=1)
            window = theta_data[0, -125:]
            crossings = np.where((window[:-1] < 0) & (window[1:] >= 0))[0]
            rhythm_pct = 0.0
            if len(crossings) >= 3:
                periods = np.diff(crossings)
                rhythm_pct = max(0.0, 100.0 - (np.std(periods) / np.mean(periods)) * 200.0)
            update_log("RHYTHM", f"{rhythm_pct:.0f}%")
            h_sg = hilbert(sg_all, axis=1)[:, -40:]; h_fg = hilbert(fg_all, axis=1)[:, -40:]
            slow_ciplv = np.zeros(28); fast_ciplv = np.zeros(28)
            for idx, (p1, p2) in enumerate(self.pairs):
                slow_ciplv[idx] = np.abs(np.imag(np.mean(h_sg[p1] * np.conj(h_sg[p2]))))
                fast_ciplv[idx] = np.abs(np.imag(np.mean(h_fg[p1] * np.conj(h_fg[p2]))))
            wm_vector = np.concatenate([slow_ciplv, fast_ciplv])
            v_max = np.max(wm_vector)
            if v_max > 0: wm_vector /= v_max
            wm_vector *= (rhythm_pct / 100.0)
            with self.lock: self.latest_vector = torch.from_numpy(wm_vector).to(device=DEVICE, dtype=DTYPE)
        except Exception: pass

    def get_latest_vector(self):
        with self.lock: return self.latest_vector.clone()

# =============================================================================
# МОДЕЛЬ МИРА (ПАССИВНЫЙ НАБЛЮДАТЕЛЬ QWEN)
# =============================================================================
class WorldObserver(threading.Thread):
    def __init__(self, q_model, q_proc):
        super().__init__(daemon=True)
        self.model, self.proc = q_model, q_proc
        self.current_image = None
        self.lock = threading.Lock()

    def update_image(self, img_pil):
        with self.lock: self.current_image = img_pil.copy()

    def run(self):
        while True:
            img = None
            with self.lock:
                if self.current_image: img = self.current_image.copy()
            if img:
                try:
                    messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "Describe in 3 keywords."}]}]
                    text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.proc(text=[text], images=[img], padding=True, return_tensors="pt").to(DEVICE)
                    with torch.no_grad():
                        out_ids = self.model.generate(**inputs, max_new_tokens=6)
                        desc = self.proc.batch_decode(out_ids, skip_special_tokens=True)[0]
                    update_log("QWEN_VIEW", desc.strip()[:30])
                except Exception: pass
            time.sleep(1.0)

# =============================================================================
# ВИЗУАЛЬНЫЙ ДВИЖОК (INTEGRATED)
# =============================================================================
class VisualEngine(threading.Thread):
    def __init__(self, pipe, eeg_proc, world_observer):
        super().__init__(daemon=True)
        self.pipe, self.eeg, self.world_observer = pipe, eeg_proc, world_observer
        self.running = True
        self.current_pil = Image.fromarray(np.random.randint(50, 100, (H, W, 3), dtype=np.uint8))
        
        proj_tmp = torch.empty(56, 768, dtype=torch.float32)
        self.sd_proj = torch.nn.init.orthogonal_(proj_tmp).to(device=DEVICE, dtype=DTYPE) * 2.0
        
        self.base_prompt = "fractal dreamscape, intricate details"
        self.base_embeds = None
        self.signal_obj = None
        
        # Лабиринт
        self.maze = LatentMaze768D(device='cuda', size=5)

    def run(self):
        with torch.no_grad():
            toks = self.pipe.tokenizer(self.base_prompt, return_tensors="pt", padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True).input_ids.to(DEVICE)
            self.base_embeds = self.pipe.text_encoder(toks)[0]

        while self.running:
            start_t = time.time()
            brain_vec = self.eeg.get_latest_vector()
            intent_mag = torch.max(brain_vec).item()
            update_log("INTENT", f"{intent_mag:.2f}")

            try:
                # 1. Смысловая навигация
                brain_thrust = None
                if intent_mag > 0.05:
                    brain_thrust = (brain_vec @ self.sd_proj).cpu().numpy()
                
                # Обновляем лабиринт (если brain_thrust None - стоим на месте)
                maze_vec, speed = self.maze.update(brain_thrust, dt=0.1)
                
                update_log("MAZE", f"Path {self.maze.current_segment_idx+1}/6")
                update_log("SPEED", f"{speed*100:.1f}")

                if self.maze.is_finished:
                    update_log("MAZE", "🏁 FINISHED")
                    self.maze = LatentMaze768D(device='cuda', size=5)
                    time.sleep(1); continue

                with torch.no_grad():
                    # 2. Инъекция смыслового вектора (Broadcasting fix)
                    # Суммируем maze_vec [768] с base_embeds [1, 77, 768]
                    maze_inj = maze_vec.unsqueeze(0).unsqueeze(0).to(dtype=DTYPE)
                    final_embeds = self.base_embeds + (maze_inj * 1.5)

                    # 3. Рендер (Статика v39.0)
                    img_np = np.array(self.current_pil)
                    noise = np.random.randint(-2, 2, (H, W, 3), dtype=np.int16)
                    warped = Image.fromarray(np.clip(img_np.astype(np.int16) + noise, 0, 255).astype(np.uint8))

                    # Strength 0.35 (стабильность) + добавка от движения
                    strength = 0.35 + (speed * 2.0)
                    strength = min(0.65, strength) # Не выше 0.65, чтобы не развалить образ

                    gen = self.pipe(
                        prompt_embeds=final_embeds,
                        image=warped,
                        strength=strength,
                        num_inference_steps=LCM_STEPS,
                        guidance_scale=1.0,
                        output_type="pil"
                    ).images[0]
                    
                    self.current_pil = gen
                    if self.signal_obj: self.signal_obj.emit(self.current_pil)
                    self.world_observer.update_image(self.current_pil)
                    update_log("SD_FPS", f"{1.0/(time.time()-start_t):.1f}")
                    
            except Exception as e:
                update_log("SD_ERR", str(e)[:30])
            
            wait = 0.12 - (time.time() - start_t)
            if wait > 0: time.sleep(wait)

class VisualSignalWrapper(QObject):
    image_ready = pyqtSignal(object)

# =============================================================================
# GUI
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEURO-WORLD v42.4")
        self.resize(800, 750); self.setStyleSheet("background-color:#050505; color: #0f0;")
        cw = QWidget(); layout = QVBoxLayout(cw); self.setCentralWidget(cw)
        self.canvas = QLabel(); self.canvas.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.canvas)
        self.debug_text = QTextEdit(); self.debug_text.setReadOnly(True); self.debug_text.setFixedHeight(120)
        self.debug_text.setStyleSheet("background-color:#111; color:#0ff; font-family:monospace; border:1px solid #333;")
        layout.addWidget(self.debug_text)
        self.log_data = {"BLE": "Wait...", "RHYTHM": "0%", "INTENT": "0.00", "SD_FPS": "0.0", "MAZE": "Ready", "SPEED": "0.0"}
        logger.log_signal.connect(self.on_log_update); self.init_core()

    def on_log_update(self, k, v):
        self.log_data[k] = v
        lines = [f"[MAZE] {self.log_data['MAZE']} | [SPEED] {self.log_data['SPEED']}",
                 f"[RHYTHM] {self.log_data['RHYTHM']} | [INTENT] {self.log_data['INTENT']} | [FPS] {self.log_data['SD_FPS']}",
                 f"[QWEN] 👀 {self.log_data.get('QWEN_VIEW', '...')}" ]
        self.debug_text.setPlainText("\n".join(lines))

    def init_core(self):
        self.timer = QTimer(); self.timer.timeout.connect(self.process_eeg); self.timer.start(160)
        threading.Thread(target=self._load, daemon=True).start()

    def process_eeg(self):
        if hasattr(self, 'eeg'): self.eeg.process_cycle()

    def _load(self):
        try:
            q_m = AutoModelForCausalLM.from_pretrained(QWEN_MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True)
            q_p = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(SD_MODEL_ID, torch_dtype=DTYPE, safety_checker=None).to(DEVICE)
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DTYPE).to(DEVICE)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
            self.eeg = EEGProcessor()
            self.obs = WorldObserver(q_m, q_p)
            self.viz = VisualEngine(pipe, self.eeg, self.obs)
            self.vsw = VisualSignalWrapper()
            self.viz.signal_obj = self.vsw.image_ready
            self.vsw.image_ready.connect(self.draw)
            self.obs.start(); self.eeg.start(); self.viz.start()
        except Exception: traceback.print_exc()

    def draw(self, img):
        q = QImage(img.tobytes("raw", "RGB"), W, H, QImage.Format.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(q).scaled(self.canvas.width(), self.canvas.height(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv); win = MainWindow(); win.show(); sys.exit(app.exec())
