#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NEURO-WORLD v28.0: UNBOUNDED SEMANTIC ENGINE (STABLE BLE RESTORED)
- Полностью восстановлена твоя оригинальная, рабочая логика BLE-клиента.
- Больше никаких зависаний на "Ждем поток...". Пакеты идут сразу.
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
LCM_STEPS = 2
STRENGTH = 0.65
EEG_FS = 250
UV_SCALE = (1.2 / 8.0 / 8388607.0) * 1e6 # PGA_GAIN = 2
PGA_GAIN_CODE = 3 # Код для PGA=2 (0=1, 1=2, 2=4...)

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()
CMD_UUID     = "c0de0001-36e1-4688-b7f5-ea07361b26a8".lower()

# =============================================================================
# ГЛОБАЛЬНЫЙ ДЕБАГГЕР
# =============================================================================
class DebugLogger(QObject):
    log_signal = pyqtSignal(str, str)

logger = DebugLogger()

def update_log(key, val):
    logger.log_signal.emit(key, str(val))

# =============================================================================
# EEG PROCESSOR (ПОЛНАЯ ИНТЕГРАЦИЯ РАБОЧЕГО ТЕСТЕРА v2.0)
# =============================================================================
class EEGProcessor(QObject, threading.Thread):
    def __init__(self):
        super().__init__()
        threading.Thread.__init__(self, daemon=True)
        self.buffer = np.zeros((8, 500))
        self.lock = threading.Lock()
        self.latest_vector = torch.zeros(56, device=DEVICE, dtype=DTYPE)
        
        # --- ФИЛЬТРЫ ---
        # 1. Режекторные (Notch) для подавления наводок сети
        self.b50, self.a50 = iirnotch(50.0, 30.0, fs=EEG_FS)
        self.b100, self.a100 = iirnotch(100.0, 30.0, fs=EEG_FS)
        
        # 2. Полосовые (Bandpass)
        self.theta = butter(3, [4, 8], btype='band', fs=EEG_FS, output='sos')
        self.slow_gamma = butter(4, [31, 51], btype='band', fs=EEG_FS, output='sos')
        self.fast_gamma = butter(4, [61, 102], btype='band', fs=EEG_FS, output='sos')
        self.pairs = [(i, j) for i in range(8) for j in range(i + 1, 8)]
        
        self.packet_counter = 0
        self.is_running = True

    def _notification_handler(self, sender, data: bytearray):
        """ Прямой перенос из рабочего eeg_tester.py """
        self.packet_counter += 1
        
        if self.packet_counter == 1:
            update_log("BLE", "🚀 ПАКЕТЫ ИДУТ (АКТИВЕН)")
            
        if self.packet_counter % 10 == 0:
            update_log("EEG_PACKETS", str(self.packet_counter))
            
        if len(data) >= 33 and data[0] == 0xA0:
            # Парсинг 8 каналов
            raw = np.zeros(8)
            for i in range(8):
                start = 2 + i * 3
                val = (data[start] << 16) | (data[start+1] << 8) | data[start+2]
                if val & 0x800000: val -= 0x1000000
                raw[i] = val * UV_SCALE
            
            with self.lock:
                self.buffer = np.roll(self.buffer, -1, axis=1)
                self.buffer[:, -1] = raw

    async def run_ble_loop(self):
        """ Основной цикл из eeg_tester.py """
        while self.is_running:
            update_log("BLE", "🔍 Поиск сервиса...")
            device = None
            try:
                # Ищем по SERVICE_UUID как в рабочем тесте
                devices = await BleakScanner.discover(timeout=3.0, service_uuids=[SERVICE_UUID])
                if devices:
                    device = devices[0]
            except Exception as e:
                update_log("BLE", f"❌ Ошибка сканера: {e}")
                await asyncio.sleep(3.0)
                continue

            if not device:
                await asyncio.sleep(1.0)
                continue

            update_log("BLE", f"✅ Найдено: {device.name or device.address}")
            
            try:
                # Используем таймаут 10.0, который починил тестер
                async with BleakClient(device, timeout=10.0) as client:
                    if client.is_connected:
                        update_log("BLE", "🔗 Соединение установлено. Настройка...")
                        
                        # Настройка PGA
                        val = (PGA_GAIN_CODE << 12) | (PGA_GAIN_CODE << 8) | (PGA_GAIN_CODE << 4) | PGA_GAIN_CODE
                        await client.write_gatt_char(CMD_UUID, bytes([0x04, (val>>8)&0xFF, val&0xFF]), response=False)
                        await client.write_gatt_char(CMD_UUID, bytes([0x05, (val>>8)&0xFF, val&0xFF]), response=False)
                        
                        self.packet_counter = 0
                        # Подписываемся на данные (DATA_UUID)
                        await client.start_notify(DATA_UUID, self._notification_handler)
                        
                        while client.is_connected and self.is_running:
                            await asyncio.sleep(1.0)
            except Exception as e:
                update_log("BLE", f"⚠️ Ошибка: {e}")
            
            await asyncio.sleep(2.0)

    def run(self):
        """ Правильный запуск asyncio цикла внутри потока """
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self.run_ble_loop())
        except Exception as e:
            print(f"Asyncio Loop Error: {e}")

    def process_cycle(self):
        with self.lock:
            if not np.any(self.buffer[:, -40:]):
                update_log("EEG_STATE", "Буфер пуст")
                return
            raw_data = self.buffer.copy()

        try:
            # 1. Извлекаем сырую Теменную Тету (Pz - канал 0)
            ch0_raw = raw_data[0, :]
            
            # --- ШАГ 0: ПРЕДВАРИТЕЛЬНАЯ ОЧИСТКА (NOTCH) ---
            # Применяем фильтры 50 и 100 Гц ко всем каналам сразу
            clean_step1 = lfilter(self.b50, self.a50, raw_data, axis=1)
            clean_data = lfilter(self.b100, self.a100, clean_step1, axis=1)

            # --- ШАГ 1: ТЕТА-АНАЛИЗ (Используем очищенные данные) ---
            theta_filt = sosfiltfilt(self.theta, clean_data, axis=1)[0, :]
            
            # --- ПАРАМЕТР 1: AMPLITUDE (Микровольты) ---
            amp = np.sqrt(np.mean(theta_filt[-100:]**2)) * 2.0 # RMS to Peak-to-Peak approx

            # --- ПАРАМЕТР 2: RHYTHMICITY (Phase Jitter) ---
            t_phase = np.angle(hilbert(theta_filt))
            crossings = np.where(np.diff(np.unwrap(t_phase)) < -np.pi)[0]
            
            rhy = 0
            if len(crossings) >= 3:
                periods = np.diff(crossings)
                expected = EEG_FS / 6.0 # 41.6 samples
                # Считаем коэффициент вариации периодов
                rhy = max(0, 100 - (np.std(periods) / expected * 200))
            
            # --- ПАРАМЕТР 3: SNR (Спектральный контраст) ---
            # Считаем энергию в полосе теты против энергии во всем буфере
            total_energy = np.sum(np.abs(hilbert(ch0_raw)))
            theta_energy = np.sum(np.abs(hilbert(theta_filt)))
            snr = (theta_energy / (total_energy + 1e-6)) * 100

            # --- ПАРАМЕТР 4: DC STABILITY (VREF / Контакт) ---
            # Оцениваем дрейф среднего значения за последние 2 секунды
            dc_drift = np.abs(np.mean(ch0_raw[-50:]) - np.mean(ch0_raw[:50]))

            # --- СТРУКТУРНЫЙ ВЫВОД ---
            # Формируем строку: [A: Амплитуда][R: Ритм][S: Контраст][D: Дрейф]
            diag = f"AMP:{amp:4.1f}uV | RHY:{rhy:2.0f}% | SNR:{snr:2.0f}% | DC:{dc_drift:3.0f}"
            
            # РЕШЕНИЕ (Система логических шлюзов)
            is_valid = (amp > 4.0) and (rhy > 35) and (snr > 15) and (dc_drift < 500)
            
            if is_valid:
                update_log("EEG_STATE", f"✅ {diag}")
                idx_start, idx_end = crossings[-2], crossings[-1]
            else:
                update_log("EEG_STATE", f"⚠️ {diag}")
                idx_start, idx_end = 500 - 40, 500

            # --- Дальнейший расчет PAC (Рабочая Память) ---
            slow_gamma_all = sosfiltfilt(self.slow_gamma, raw_data, axis=1)
            fast_gamma_all = sosfiltfilt(self.fast_gamma, raw_data, axis=1)
            
            mid = (idx_start + idx_end) // 2
            h_slow = hilbert(slow_gamma_all, axis=1)
            h_fast = hilbert(fast_gamma_all, axis=1)
            
            past_half = h_slow[:, idx_start:mid]
            future_half = h_fast[:, mid:idx_end]

            slow_ciplv = np.zeros(28)
            fast_ciplv = np.zeros(28)
            for idx, (p1, p2) in enumerate(self.pairs):
                slow_ciplv[idx] = np.abs(np.imag(np.mean(past_half[p1] * np.conj(past_half[p2]))))
                fast_ciplv[idx] = np.abs(np.imag(np.mean(future_half[p1] * np.conj(future_half[p2]))))

            wm_vector = np.concatenate([slow_ciplv, fast_ciplv])
            v_max = np.max(wm_vector)
            if v_max > 0: wm_vector = wm_vector / v_max

            with self.lock:
                self.latest_vector = torch.from_numpy(wm_vector).to(device=DEVICE, dtype=DTYPE)

        except Exception as e:
            update_log("EEG_STATE", f"❌ Анализ: {str(e)[:10]}")

    def get_latest_vector(self):
        with self.lock:
            return self.latest_vector.clone()

# =============================================================================
# МОДЕЛЬ МИРА (АСИНХРОННЫЙ QWEN)
# =============================================================================
class WorldObserver(threading.Thread):
    def __init__(self, q_model, q_proc, eeg_proc):
        super().__init__(daemon=True)
        self.model = q_model
        self.proc = q_proc
        self.eeg = eeg_proc
        self.current_image = None
        self.world_context_text = "abstract flowing neural geometry"
        self.lock = threading.Lock()
        self.running = True
        
        torch.manual_seed(42)
        proj_qwen_tmp = torch.empty(56, self.model.config.hidden_size, dtype=torch.float32)
        self.qwen_proj = torch.nn.init.orthogonal_(proj_qwen_tmp).to(device=DEVICE, dtype=DTYPE) * 0.1

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
                time.sleep(0.1)
                continue

            b_vec = self.eeg.get_latest_vector()

            try:
                update_log("QWEN", "⏳ Думает...")
                messages =[
                    {"role": "system", "content": "Describe the intent in 3 keywords."},
                    {"role": "user", "content": "Decode this neural intent:"}
                ]
                text = self.proc.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                
                try:
                    inputs = self.proc(text=text, images=img, return_tensors="pt").to(DEVICE)
                except (ValueError, TypeError):
                    inputs = self.proc(text, return_tensors="pt").to(DEVICE)
                
                with torch.no_grad():
                    embeds = self.model.get_input_embeddings()(inputs.input_ids)
                    brain_qwen_embed = (b_vec @ self.qwen_proj).unsqueeze(0).unsqueeze(0)
                    embeds[:, -1:, :] += brain_qwen_embed
                    
                    output_ids = self.model.generate(
                        inputs_embeds=embeds,
                        attention_mask=torch.ones(embeds.shape[:2], device=DEVICE),
                        max_new_tokens=6,
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=getattr(self.proc, "tokenizer", self.proc).eos_token_id
                    )
                    desc = getattr(self.proc, "tokenizer", self.proc).decode(output_ids[0], skip_special_tokens=True)
                    
                    if "Decode this neural intent:" in desc:
                        desc = desc.split("Decode this neural intent:")[-1].strip()
                        
                if len(desc) > 2:
                    clean_desc = "".join(c for c in desc if c.isalnum() or c in " ,").strip()[:100]
                    self.world_context_text = clean_desc
                    update_log("QWEN", f"✅ {clean_desc}")
                    
            except Exception as e:
                update_log("QWEN", f"❌ Ошибка: {e}")
                
            time.sleep(0.1)

# =============================================================================
# ВИЗУАЛЬНЫЙ ДВИЖОК (MAX REALTIME)
# =============================================================================
class VisualEngine(threading.Thread):
    def __init__(self, pipe, world_observer, eeg_proc):
        super().__init__(daemon=True)
        self.pipe = pipe
        self.world_observer = world_observer
        self.eeg = eeg_proc
        self.running = True
        
        self.current_pil = Image.fromarray(np.random.randint(20, 230, (H, W, 3), dtype=np.uint8))
        
        torch.manual_seed(42)
        proj_sd_tmp = torch.empty(56, 768, dtype=torch.float32)
        self.sd_proj = torch.nn.init.orthogonal_(proj_sd_tmp).to(device=DEVICE, dtype=DTYPE) * 2.0

        self.last_context_text = ""
        self.base_embeds = None
        self.signal_obj = None

    def run(self):
        while self.running:
            start_t = time.time()
            
            brain_vec = self.eeg.get_latest_vector()
            frame = self.current_pil.copy()
            current_context = self.world_observer.world_context_text

            try:
                with torch.no_grad():
                    if current_context != self.last_context_text or self.base_embeds is None:
                        tokens = self.pipe.tokenizer(current_context, return_tensors="pt", padding="max_length", max_length=self.pipe.tokenizer.model_max_length, truncation=True).input_ids.to(DEVICE)
                        self.base_embeds = self.pipe.text_encoder(tokens)[0]
                        self.last_context_text = current_context

                    brain_sd_embed = (brain_vec @ self.sd_proj).unsqueeze(0).unsqueeze(0)
                    final_prompt_embeds = self.base_embeds + (brain_sd_embed * 1.5)

                    img_np = np.array(frame)
                    M = cv2.getRotationMatrix2D((W//2, H//2), 0, 1.02)
                    warped = Image.fromarray(cv2.warpAffine(img_np, M, (W, H), borderMode=cv2.BORDER_REFLECT_101))

                    gen = self.pipe(
                        prompt_embeds=final_prompt_embeds,
                        image=warped,
                        strength=STRENGTH,
                        num_inference_steps=LCM_STEPS,
                        guidance_scale=1.0,
                        output_type="pil"
                    ).images[0]
                    
                    self.current_pil = gen
                    if self.signal_obj:
                        self.signal_obj.emit(self.current_pil)
                    
                    self.world_observer.update_image(self.current_pil)
                    
                    fps = 1.0 / (time.time() - start_t)
                    update_log("SD_FPS", f"{fps:.1f} FPS")
                    
            except Exception as e:
                update_log("SD_ERROR", str(e))
            
            elapsed = time.time() - start_t
            if elapsed < 0.12:
                time.sleep(0.12 - elapsed)

class VisualSignalWrapper(QObject):
    image_ready = pyqtSignal(object)

# =============================================================================
# GUI
# =============================================================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("NEURO-WORLD: V-JEPA UNBOUNDED")
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
        self.debug_text.setFixedHeight(120)
        self.debug_text.setStyleSheet("background-color:#111; color:#0ff; font-family:monospace; font-size:12px; border:1px solid #333;")
        layout.addWidget(self.debug_text)
        
        self.log_data = {
            "BLE": "Ожидание...", "EEG_PACKETS": "0", "EEG_STATE": "Ожидание...",
            "SD_FPS": "0.0 FPS", "QWEN": "Ожидание..."
        }
        
        logger.log_signal.connect(self.on_log_update)
        self.init_core()

    def on_log_update(self, key, val):
        self.log_data[key] = val
        text = " | ".join([f"[{k}] {v}" for k, v in self.log_data.items()])
        self.debug_text.setPlainText(text)

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
            update_log("QWEN", f"Загрузка {QWEN_MODEL_ID}...")
            q_model = AutoModelForCausalLM.from_pretrained(
                QWEN_MODEL_ID, torch_dtype=DTYPE, device_map="auto", trust_remote_code=True
            )
            q_proc = AutoProcessor.from_pretrained(QWEN_MODEL_ID, trust_remote_code=True)
            
            update_log("SD_FPS", "Загрузка LCM-SD...")
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                SD_MODEL_ID, torch_dtype=DTYPE, safety_checker=None
            ).to(DEVICE)
            pipe.vae = AutoencoderTiny.from_pretrained("madebyollin/taesd", torch_dtype=DTYPE).to(DEVICE)
            pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

            self.eeg = EEGProcessor()
            self.world_observer = WorldObserver(q_model, q_proc, self.eeg)
            
            self.viz = VisualEngine(pipe, self.world_observer, self.eeg)
            self.viz_signal = VisualSignalWrapper()
            self.viz.signal_obj = self.viz_signal.image_ready
            self.viz_signal.image_ready.connect(self.draw)

            self.world_observer.start()
            self.eeg.start()
            self.viz.start()
            
            update_log("SD_FPS", "Готово!")
            
        except Exception as e:
            update_log("QWEN", f"FATAL ERROR: {e}")
            traceback.print_exc()

    def draw(self, img):
        q = QImage(img.tobytes("raw", "RGB"), W, H, QImage.Format.Format_RGB888)
        self.canvas.setPixmap(QPixmap.fromImage(q).scaled(self.canvas.width(), self.canvas.height(), Qt.AspectRatioMode.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
