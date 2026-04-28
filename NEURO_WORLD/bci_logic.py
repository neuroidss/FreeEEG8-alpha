import numpy as np
import time, threading, asyncio
from bleak import BleakScanner, BleakClient
from scipy.signal import butter, lfilter, hilbert

# === CONFIG ===
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()
CMD_UUID     = "c0de0001-36e1-4688-b7f5-ea07361b26a8".lower()

UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6 
EEG_FS = 250.0

RADIUS = 10.0
ANGLES = np.array([-72, -36, 36, 72, 108, 144, -144, -108]) * np.pi / 180
ELECTRODES = np.stack([np.cos(ANGLES) * RADIUS, np.sin(ANGLES) * RADIUS], axis=1)
PAIRS =[(i, j) for i in range(8) for j in range(i + 1, 8)]

class CrystalBCI:
    def __init__(self):
        self.buffer = np.zeros((8, 500), dtype=np.float32)
        self.lock = threading.Lock()
        self.b_band, self.a_band = butter(3, [18, 36], btype='band', fs=EEG_FS)
        
        self.is_connected = False
        self.simulation_mode = True 
        
        self.vx, self.vy, self.tq = 0.0, 0.0, 0.0
        self.persistence = 0.0
        
        # 🔥 НОВЫЙ МАССИВ ДЛЯ ТОПОЛОГИИ МОЗГА (28 связей)
        self.ciplv_vec = np.zeros(28, dtype=np.float32)
        
        self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
        self.thread.start()

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_manager())

    async def ble_manager(self):
        def match_device(device, adv):
            if adv.service_uuids and SERVICE_UUID in [u.lower() for u in adv.service_uuids]: return True
            if device.name and ("FreeEEG" in device.name or "Crystal" in device.name): return True
            return False

        while True:
            try:
                device = await BleakScanner.find_device_by_filter(match_device, timeout=3.0)
                if device:
                    async with BleakClient(device, timeout=10.0) as client:
                        self.is_connected = True
                        self.simulation_mode = False 
                        await client.write_gatt_char(CMD_UUID, bytes([0x04, 0x22, 0x22]), response=False)
                        await asyncio.sleep(0.1)
                        await client.write_gatt_char(CMD_UUID, bytes([0x05, 0x22, 0x22]), response=False)
                        await client.start_notify(DATA_UUID, self._notification_handler)
                        while client.is_connected:
                            await asyncio.sleep(1)
            except: pass
            self.is_connected = False
            self.simulation_mode = True 
            await asyncio.sleep(1)

    def _notification_handler(self, sender, data):
        if len(data) >= 33 and data[0] == 0xA0 and data[32] == 0xC0:
            sample = np.zeros(8, dtype=np.float32)
            for i in range(8):
                start = 2 + i * 3
                val = (data[start] << 16) | (data[start+1] << 8) | data[start+2]
                if val & 0x800000: val -= 0x1000000
                sample[i] = val * UV_SCALE
            with self.lock:
                self.buffer = np.roll(self.buffer, -1, axis=1)
                self.buffer[:, -1] = sample
            self.process_raw()

    def process_raw(self):
        with self.lock:
            data = self.buffer.copy()
            
        band_data = lfilter(self.b_band, self.a_band, data, axis=1)
        ana = hilbert(band_data, axis=1)[:, -40:]
        
        tvx, tvy, ttq = 0.0, 0.0, 0.0
        raw_vec = np.zeros(28)
        
        for idx, (i, j) in enumerate(PAIRS):
            val = np.imag(np.mean(ana[i] * np.conj(ana[j])))
            dx, dy = ELECTRODES[j,0]-ELECTRODES[i,0], ELECTRODES[j,1]-ELECTRODES[i,1]
            
            tvx += val * dx
            tvy += val * dy
            ttq += (val * (ELECTRODES[i,0]*dy - ELECTRODES[i,1]*dx)) / (RADIUS * 10)
            
            # Взвешиваем связь на расстояние (дальние связи влияют сильнее)
            dist = np.sqrt(dx**2 + dy**2)
            raw_vec[idx] = val * (dist / (RADIUS*2))
        
        smooth, gain = 0.975, 0.05 * 1.5
        self.vx = self.vx * smooth + (tvx * 0.1) * gain * (1 - smooth)
        self.vy = self.vy * smooth + (tvy * 0.1) * gain * (1 - smooth)
        self.tq = self.tq * smooth + (ttq * 0.2) * gain * (1 - smooth)
        
        # Сглаживаем 28-мерный вектор
        self.ciplv_vec = self.ciplv_vec * smooth + (raw_vec * 0.5) * (1 - smooth)
        
        mag = np.sqrt(self.vx**2 + self.vy**2)
        if mag > 1.0: self.vx /= mag; self.vy /= mag
        self.tq = np.clip(self.tq, -1.0, 1.0)
        
        if mag > 0.05: self.persistence = min(1.0, self.persistence + 0.05)
        else: self.persistence *= 0.95

    def update_sim(self):
        if not self.is_connected:
            t = time.time()
            with self.lock:
                self.buffer = np.roll(self.buffer, -1, axis=1)
                for i in range(8):
                    rot_x = ELECTRODES[i,0]*np.cos(t*0.5) - ELECTRODES[i,1]*np.sin(t*0.5)
                    wave = np.sin(2*np.pi*25*t + rot_x*0.3)*15
                    self.buffer[i, -1] = wave + np.random.normal(0, 2)
            self.process_raw()
