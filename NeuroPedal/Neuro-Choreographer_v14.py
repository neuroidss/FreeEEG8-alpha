import sys, serial, threading, random, time, torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import sounddevice as sd
from PyQt6.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QLabel
from PyQt6.QtCore import QTimer, Qt, QPoint
from PyQt6.QtGui import QCursor
import pyqtgraph.opengl as gl
import pyqtgraph as pg
from collections import deque
from pynput import keyboard

# ==========================================
# 1. КОНФИГУРАЦИЯ (Real-time Spatio-Temporal)
# ==========================================
SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 921600
FS = 250.0

CONFIG = {
    "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
    "EEG_WIN": 200,        
    "BATCH": 512,          
    "LR": 8e-4,
    "REPLAY_SIZE": 15000,  
    "SAFE_ZONE": 15.0,
    "G_CONSTANT": 0.1,
    "SENSITIVITY": 12.0
}

# ==========================================
# 2. МОДЕЛЬ
# ==========================================
class NeuralBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(8, 128), nn.LayerNorm(128), nn.ELU(),
            nn.Linear(128, 256), nn.LayerNorm(256), nn.ELU()
        )
        # Глубокая память последовательностей (учит иероглифы и остановки)
        self.memory = nn.GRU(256, 256, num_layers=3, batch_first=True)
        self.motor = nn.Linear(256, 2)

    def forward(self, x, h=None):
        b, t, c = x.size()
        feat = self.encoder(x.reshape(-1, c)).reshape(b, t, -1)
        out, h_new = self.memory(feat, h)
        return self.motor(out[:, -1, :]), h_new

model = NeuralBrain().to(CONFIG["DEVICE"])
optimizer = optim.AdamW(model.parameters(), lr=CONFIG["LR"])
scaler = torch.amp.GradScaler('cuda')

# ==========================================
# 3. ГЛОБАЛЬНОЕ СОСТОЯНИЕ
# ==========================================
replay_buffer = deque(maxlen=CONFIG["REPLAY_SIZE"])
eeg_history = deque(maxlen=CONFIG["EEG_WIN"])
buffer_lock, viz_lock, model_lock = threading.Lock(), threading.Lock(), threading.Lock()

is_drawing = False
# Короткие хвосты для лучшего визуального фидбека
user_pts = deque(maxlen=60)
brain_pts = deque(maxlen=100)
brain_pos = np.array([0.0, 0.0])
h_live = None

# Для расчета скорости мыши в реальном времени
last_abs_mouse_pos = None

viz_shared = {"loss": 0.0, "pred": np.zeros(2), "is_learning": False}

# ==========================================
# 4. ЦИКЛ ОБУЧЕНИЯ
# ==========================================
def training_loop():
    global viz_shared
    criterion = nn.MSELoss()
    while True:
        if len(replay_buffer) < CONFIG["BATCH"]:
            time.sleep(0.5); continue
        with buffer_lock:
            batch = random.sample(replay_buffer, CONFIG["BATCH"])
        
        eeg, target_v = [torch.tensor(np.array(t), device=CONFIG["DEVICE"]).float() for t in zip(*batch)]
        
        with model_lock:
            model.train()
            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                pred_v, _ = model(eeg)
                loss = criterion(pred_v, target_v)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            
        with viz_lock: viz_shared["loss"] = loss.item()
        time.sleep(0.002)

# ==========================================
# 5. ОБРАБОТКА (С ФИКСОМ ОСТАНОВКИ)
# ==========================================
def process_data(raw_eeg):
    global brain_pos, h_live, last_abs_mouse_pos
    
    # 1. Сбор окна ЭЭГ
    with buffer_lock:
        eeg_history.append(raw_eeg)
        if len(eeg_history) < CONFIG["EEG_WIN"]: return
        current_win = np.array(eeg_history)

    # 2. ОПРОС МЫШИ (Hardware-level)
    # Спрашиваем систему о положении курсора ПРЯМО СЕЙЧАС
    current_global_pos = QCursor.pos()
    mouse_v = [0.0, 0.0]
    
    if last_abs_mouse_pos is not None:
        # Считаем реальное смещение между тиками ЭЭГ
        dx = (current_global_pos.x() - last_abs_mouse_pos.x()) / 10.0
        dy = -(current_global_pos.y() - last_abs_mouse_pos.y()) / 10.0
        mouse_v = [dx, dy]
    
    last_abs_mouse_pos = current_global_pos

    # 3. ЗАПИСЬ В БУФЕР (Теперь ловит и остановки!)
    if is_drawing:
        with buffer_lock:
            replay_buffer.append((current_win, mouse_v))

    # 4. ИНФЕРЕНС (Независимый)
    with torch.no_grad():
        with model_lock:
            model.eval()
            eeg_t = torch.tensor(current_win, device=CONFIG["DEVICE"]).float().unsqueeze(0)
            pred_v_t, h_live = model(eeg_t, h_live)
        pred_v = pred_v_t[0].cpu().numpy()

    # 5. ОБНОВЛЕНИЕ ПОЗИЦИИ
    dist = np.linalg.norm(brain_pos)
    gravity = np.zeros(2)
    if dist > CONFIG["SAFE_ZONE"]:
        gravity = -(brain_pos / (dist + 1e-6)) * (dist - CONFIG["SAFE_ZONE"]) * CONFIG["G_CONSTANT"]
    
    # Мозг управляет курсором напрямую через вектор скорости
    brain_pos += (pred_v * CONFIG["SENSITIVITY"] + gravity)
    brain_pos = np.clip(brain_pos, -40, 40)

    with viz_lock:
        viz_shared.update({"pred": pred_v, "is_learning": is_drawing})

# ==========================================
# 6. СЕРВИСНЫЙ СЛОЙ
# ==========================================
def serial_worker():
    try: ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.01)
    except: ser = None; print("SIMULATION")
    buf = bytearray()
    while True:
        if ser:
            data = ser.read(1024)
            if not data: continue
            buf.extend(data)
            while len(buf) >= 33:
                if buf[0] == 0xA0 and buf[32] == 0xC0:
                    s = [((buf[2+ch*3]<<16)|(buf[3+ch*3]<<8)|buf[4+ch*3]) for ch in range(8)]
                    s = [v-0x1000000 if v&0x800000 else v for v in s]
                    process_data(np.array(s, dtype=np.float32) / 1e6)
                    del buf[:33]
                else: buf.pop(0)
        else:
            process_data(np.random.randn(8) * 0.05); time.sleep(0.004)

def audio_callback(outdata, frames, time_info, status):
    d = viz_shared["pred"]
    v_norm = np.linalg.norm(d)
    pitch = 180.0 + np.clip(d[1] * 800, -100, 1500)
    # Звук затихает при остановке предсказания
    amp = np.clip(v_norm * 12.0, 0.0, 0.7)
    t = np.linspace(0, frames/44100, frames)
    sig = np.sin(2 * np.pi * pitch * t) * amp
    outdata[:] = np.repeat(np.tanh(sig[:, None]), 2, axis=1)

class Visualizer(gl.GLViewWidget):
    def __init__(self):
        super().__init__(); self.setMouseTracking(True)
        self.setCameraPosition(azimuth=-90, elevation=90, distance=40.0)
        self.user_line = gl.GLLinePlotItem(color=(1, 1, 0, 0.4), width=1, mode='line_strip')
        self.brain_line = gl.GLLinePlotItem(color=(0, 1, 1, 1.0), width=3, mode='line_strip')
        self.addItem(self.user_line); self.addItem(self.brain_line)
    def mousePressEvent(self, ev):
        global is_drawing; is_drawing = True; user_pts.clear()
    def mouseReleaseEvent(self, ev):
        global is_drawing; is_drawing = False
    def mouseMoveEvent(self, ev):
        if is_drawing:
            w, h = self.width(), self.height()
            x, y = (ev.position().x()/w - 0.5) * 80, (0.5 - ev.position().y()/h) * 80
            user_pts.append([x, y])
            self.user_line.setData(pos=np.column_stack([list(user_pts), np.zeros(len(user_pts))]))

class Win(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro-Darwinism: Perfect Real-time Sync")
        self.setGeometry(50, 50, 1600, 900)
        cw = QWidget(); self.setCentralWidget(cw); layout = pg.QtWidgets.QGridLayout(cw)
        self.view = Visualizer(); layout.addWidget(self.view, 0, 0)
        side = QVBoxLayout(); self.mode_label = QLabel("Mode: INFERENCE")
        self.mode_label.setStyleSheet("font-size: 24px; color: green;")
        self.loss_p = pg.PlotWidget(title="Backbone Loss"); self.loss_curve = self.loss_p.plot(pen='y')
        for w in [self.mode_label, self.loss_p]: side.addWidget(w)
        layout.addLayout(side, 0, 1)
        layout.setColumnStretch(0, 4); layout.setColumnStretch(1, 1)
        self.data_loss = deque(maxlen=100)

def update_ui():
    with viz_lock: d = viz_shared.copy()
    win.data_loss.append(d["loss"]); win.loss_curve.setData(list(win.data_loss))
    win.mode_label.setText(f"Mode: {'LEARNING' if d['is_learning'] else 'INFERENCE'}")
    win.mode_label.setStyleSheet(f"font-size: 24px; color: {'red' if d['is_learning'] else 'green'};")
    
    brain_pts.append(list(brain_pos))
    win.view.brain_line.setData(pos=np.column_stack([list(brain_pts), np.full(len(brain_pts), 0.02)]))
    
    if not is_drawing and len(user_pts) > 0:
        user_pts.popleft()
        win.view.user_line.setData(pos=np.column_stack([list(user_pts), np.zeros(len(user_pts))]))

if __name__ == '__main__':
    app = QApplication(sys.argv); win = Win(); win.show()
    threading.Thread(target=serial_worker, daemon=True).start()
    threading.Thread(target=training_loop, daemon=True).start()
    with sd.OutputStream(channels=2, callback=audio_callback, samplerate=44100):
        timer = QTimer(); timer.timeout.connect(update_ui); timer.start(30)
        sys.exit(app.exec())
