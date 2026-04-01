import sys
import asyncio
import threading
import numpy as np
import pyqtgraph as pg
import torch
import sounddevice as sd
import random
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QLabel, QGraphicsEllipseItem, QGraphicsRectItem)
from PyQt6.QtCore import pyqtSignal, QObject, QTimer, Qt
from bleak import BleakScanner, BleakClient

# ==========================================
# 1. КОНФИГУРАЦИЯ И ГЕОМЕТРИЯ (CUDA)
# ==========================================
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ANGLES = np.deg2rad([-72, -36, 36, 72, 108, 144, -144, -108])
PINS_POS = torch.tensor(np.column_stack([np.cos(ANGLES), np.sin(ANGLES)]), 
                        dtype=torch.float32, device=DEVICE)

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()
CMD_UUID     = "c0de0001-36e1-4688-b7f5-ea07361b26a8".lower()

CHANNELS, EEG_FS, AUDIO_FS = 8, 250, 44100
BUFFER_SIZE = 250 
UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6

ctrl = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'attack': False, 'interact': False, 'floor': 1}

# ==========================================
# 2. АУДИО-ФИДБЕК (СИНХРОНИЗАЦИЯ С МОЗГОМ)
# ==========================================
class NeuroSound:
    def __init__(self):
        self.phase = 0.0
        self.stream = sd.OutputStream(channels=2, samplerate=AUDIO_FS, callback=self.callback)
        self.stream.start()

    def callback(self, outdata, frames, time_info, status):
        t = (self.phase + np.arange(frames)) / AUDIO_FS
        self.phase += frames
        r = np.sqrt(ctrl['x']**2 + ctrl['y']**2)
        vol = np.clip(r * 0.35, 0, 0.6)
        freq = 250.0 * (2.2 ** ctrl['y'])
        wave = np.sin(2 * np.pi * freq * t)
        if ctrl['attack']: wave += np.random.normal(0, 0.4, frames)
        if ctrl['interact']: wave *= (2.0 + np.sin(2 * np.pi * 150 * t))
        pan_l = np.clip(1.0 - ctrl['x'], 0.0, 1.0)
        pan_r = np.clip(1.0 + ctrl['x'], 0.0, 1.0)
        outdata[:, 0] = np.tanh(wave * pan_l * vol)
        outdata[:, 1] = np.tanh(wave * pan_r * vol)

snd = NeuroSound()

# ==========================================
# 3. ФИЗИЧЕСКИЙ ДВИЖОК ЛАБИРИНТА
# ==========================================
class MazeGame(pg.GraphicsLayoutWidget):
    def __init__(self, parent):
        super().__init__()
        self.parent_win = parent
        self.plot = self.addPlot()
        self.plot.setAspectLocked(True)
        self.maze_items = []
        self.cell_size = 2.0  # Размер одного блока стены
        self.player_radius = 0.4
        
        self.init_level()
        
        # Игрок (Голубая сфера)
        self.player = QGraphicsEllipseItem(-self.player_radius, -self.player_radius, 
                                           self.player_radius*2, self.player_radius*2)
        self.player.setBrush(pg.mkBrush('#0ff'))
        self.player.setPen(pg.mkPen('#fff', width=1))
        self.player.setZValue(100)
        self.plot.addItem(self.player)
        
        self.timer = QTimer(); self.timer.timeout.connect(self.update_game); self.timer.start(16)

    def generate_maze(self, dim):
        # Алгоритм Recursive Backtracker
        maze = np.ones((dim, dim), dtype=int)
        def walk(x, y):
            maze[y, x] = 0
            dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]
            random.shuffle(dirs)
            for dx, dy in dirs:
                nx, ny = x + dx*2, y + dy*2
                if 0 <= nx < dim and 0 <= ny < dim and maze[ny, nx] == 1:
                    maze[y+dy, x+dx] = 0
                    walk(nx, ny)
        walk(1, 1)
        maze[dim-2, dim-2] = 2 # Выход
        return maze

    def init_level(self):
        for item in self.maze_items: self.plot.removeItem(item)
        self.maze_items = []
        
        # Скейлинг сложности
        self.dim = 7 + (ctrl['floor'] * 2)
        self.maze_map = self.generate_maze(self.dim)
        
        # Отрисовка сплошных стен
        # Сдвигаем всё так, чтобы центр лабиринта был в 0,0
        self.offset = (self.dim * self.cell_size) / 2
        
        for r in range(self.dim):
            for c in range(self.dim):
                val = self.maze_map[r, c]
                x = c * self.cell_size - self.offset
                y = -(r * self.cell_size - self.offset)
                
                if val == 1: # Стена
                    w = QGraphicsRectItem(x, y - self.cell_size, self.cell_size, self.cell_size)
                    w.setBrush(pg.mkBrush(40, 42, 50))
                    w.setPen(pg.mkPen(80, 80, 90))
                    self.plot.addItem(w); self.maze_items.append(w)
                elif val == 2: # Портал
                    self.portal = QGraphicsEllipseItem(x + 0.2, y - self.cell_size + 0.2, 1.6, 1.6)
                    self.portal.setBrush(pg.mkBrush(0, 255, 100, 150))
                    self.portal.setPen(pg.mkPen('#0f0', width=2))
                    self.plot.addItem(self.portal); self.maze_items.append(self.portal)
                    self.exit_coord = (x + 1.0, y - 1.0)
        
        # Спавн игрока в (1,1)
        self.px = 1 * self.cell_size - self.offset + 1.0
        self.py = -(1 * self.cell_size - self.offset + 1.0)
        
        self.plot.setXRange(-self.offset - 2, self.offset + 2)
        self.plot.setYRange(-self.offset - 2, self.offset + 2)

    def check_collision(self, x, y):
        # Перевод мировых координат в индексы сетки
        grid_x = int((x + self.offset) / self.cell_size)
        grid_y = int((self.offset - y) / self.cell_size)
        
        if 0 <= grid_x < self.dim and 0 <= grid_y < self.dim:
            return self.maze_map[grid_y, grid_x] == 1
        return True

    def update_game(self):
        dt_speed = 0.15
        vx = ctrl['x'] * dt_speed
        vy = ctrl['y'] * dt_speed
        
        # Раздельная проверка осей для "скольжения" вдоль стен
        # Проверяем X
        if not self.check_collision(self.px + vx + np.sign(vx)*self.player_radius, self.py):
            self.px += vx
            
        # Проверяем Y
        if not self.check_collision(self.px, self.py + vy + np.sign(vy)*self.player_radius):
            self.py += vy

        self.player.setPos(self.px, self.py)
        
        # Проверка портала
        dist = np.sqrt((self.px - self.exit_coord[0])**2 + (self.py - self.exit_coord[1])**2)
        if dist < 1.2:
            self.parent_win.lbl_status.setText("🌀 ПОРТАЛ ОТКРЫТ! СЖИМАЙ ФОКУС (Z > 1.8)")
            if ctrl['interact']:
                ctrl['floor'] += 1
                self.init_level()
        else:
            self.parent_win.lbl_status.setText(f"🏰 ЭТАЖ {ctrl['floor']} | КОНТРОЛЬ: X:{ctrl['x']:.1f} Y:{ctrl['y']:.1f}")

# ==========================================
# 4. DSP И BLE (БЕЗ ИЗМЕНЕНИЙ)
# ==========================================
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Neuro Dungeon Solid Physics")
        self.resize(1000, 1050)
        self.setStyleSheet("background-color: #050505; color: white;")
        self.data_gpu = torch.zeros((8, BUFFER_SIZE), dtype=torch.float32, device=DEVICE)
        
        cw = QWidget(); self.setCentralWidget(cw); layout = QVBoxLayout(cw)
        self.lbl_status = QLabel("ПОИСК ДЕВАЙСА..."); self.lbl_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_status.setStyleSheet("color:#0f0; font-family:monospace; font-size:20px; background:#111;")
        layout.addWidget(self.lbl_status)
        
        self.game = MazeGame(self); layout.addWidget(self.game)
        
        self.lbl_btns = QLabel("Z-FOCUS: 0.00")
        self.lbl_btns.setStyleSheet("font-family:monospace; font-size:22px; color:#f0f;")
        self.lbl_btns.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.lbl_btns)

        self.ble = BleWorker()
        self.ble.data_received.connect(self.on_data)
        threading.Thread(target=self.ble.start_ble, daemon=True).start()
        self.timer = QTimer(); self.timer.timeout.connect(self.dsp); self.timer.start(40)

    def on_data(self, data):
        if len(data) < 33: return
        sample = np.zeros(8)
        for i in range(8):
            val = int.from_bytes(data[2+i*3:5+i*3], 'big', signed=True)
            sample[i] = val * UV_SCALE
        self.data_gpu = torch.roll(self.data_gpu, -1, dims=1)
        self.data_gpu[:, -1] = torch.tensor(sample, dtype=torch.float32, device=DEVICE)

    def dsp(self):
        with torch.no_grad():
            centered = self.data_gpu - torch.mean(self.data_gpu, dim=0, keepdim=True)
            f = torch.fft.rfft(centered, dim=1)
            freqs = torch.fft.rfftfreq(BUFFER_SIZE, d=1/EEG_FS, device=DEVICE)
            intensities = torch.mean(torch.abs(f[:, (freqs>=8) & (freqs<=25)]), dim=1)
            m_val, max_val = torch.mean(intensities), torch.max(intensities)
            focus = (max_val / (m_val + 1e-6)).item()
            
            if focus > 1.15:
                weights = intensities / (torch.sum(intensities) + 1e-6)
                cx = torch.sum(weights * PINS_POS[:, 0]).item()
                cy = torch.sum(weights * PINS_POS[:, 1]).item()
                prev_x, prev_y = ctrl['x'], ctrl['y']
                ctrl['x'] = np.clip(cx * 15.0, -1, 1)
                ctrl['y'] = np.clip(cy * 15.0, -1, 1)
                vel = np.sqrt((ctrl['x'] - prev_x)**2 + (ctrl['y'] - prev_y)**2)
                ctrl['attack'] = (vel > 0.45)
                ctrl['interact'] = (focus > 1.8 and vel < 0.1)
            else:
                ctrl['x'] *= 0.5; ctrl['y'] *= 0.5; ctrl['attack'] = False; ctrl['interact'] = False
            self.lbl_btns.setText(f"FOCUS (Z): {focus:.2f} | {'💥 ATK' if ctrl['attack'] else '---'}")

class BleWorker(QObject):
    data_received = pyqtSignal(bytearray)
    def start_ble(self): asyncio.run(self.run())
    async def run(self):
        device = await BleakScanner.find_device_by_filter(lambda d, ad: SERVICE_UUID in ad.service_uuids)
        if not device: return
        async with BleakClient(device) as client:
            pga = 2
            val = (pga << 12) | (pga << 8) | (pga << 4) | pga
            await client.write_gatt_char(CMD_UUID, bytes([0x04, (val>>8)&0xFF, val&0xFF]), response=False)
            await client.write_gatt_char(CMD_UUID, bytes([0x05, (val>>8)&0xFF, val&0xFF]), response=False)
            await client.start_notify(DATA_UUID, lambda s, d: self.data_received.emit(d))
            while True: await asyncio.sleep(1)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow(); win.show()
    sys.exit(app.exec())
