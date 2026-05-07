import numpy as np
import time, threading, asyncio, traceback

BCI_LIBRARIES_AVAILABLE = True
try:
    from bleak import BleakScanner, BleakClient
    from scipy.signal import butter, lfilter, hilbert
except ImportError:
    BCI_LIBRARIES_AVAILABLE = False

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b".lower()
DATA_UUID    = "beb5483e-36e1-4688-b7f5-ea07361b26a8".lower()
CMD_UUID     = "c0de0001-36e1-4688-b7f5-ea07361b26a8".lower()

UV_SCALE = (1.2 / 4.0 / 8388607.0) * 1e6 
EEG_FS = 250.0
RADIUS = 10.0
ANGLES = np.array([-72, -36, 36, 72, 108, 144, -144, -108]) * np.pi / 180
ELECTRODES = np.stack([np.cos(ANGLES) * RADIUS, np.sin(ANGLES) * RADIUS], axis=1)
PAIRS =[(i, j) for i in range(8) for j in range(i + 1, 8)]

class FreeEEG:
    def __init__(self):
        self.buffer = np.zeros((8, 500), dtype=np.float32)
        self.lock = threading.Lock()
        if BCI_LIBRARIES_AVAILABLE:
            self.b_band, self.a_band = butter(3, [18, 36], btype='band', fs=EEG_FS)
        self.is_connected = False
        self.vx, self.vy, self.tq = 0.0, 0.0, 0.0
        self.persistence = 0.0
        self.ciplv_vec = np.zeros(28, dtype=np.float32)
        
        if BCI_LIBRARIES_AVAILABLE:
            self.thread = threading.Thread(target=self._run_async_loop, daemon=True)
            self.thread.start()

    @staticmethod
    def is_available(): return BCI_LIBRARIES_AVAILABLE

    def _run_async_loop(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        loop.run_until_complete(self.ble_manager())

    async def ble_manager(self):
        def match_device(device, adv):
            return adv.service_uuids and SERVICE_UUID in [u.lower() for u in adv.service_uuids]

        while True:
            try:
                device = await BleakScanner.find_device_by_filter(match_device, timeout=5.0)
                if device:
                    print(f"[BCI] Found device: {device.address}, connecting...")
                    async with BleakClient(device, timeout=15.0) as client:
                        self.is_connected = True
                        print("[BCI] ✅ Connected!")
                        await client.write_gatt_char(CMD_UUID, bytes([0x04, 0x22, 0x22]), response=False)
                        await asyncio.sleep(0.1)
                        await client.write_gatt_char(CMD_UUID, bytes([0x05, 0x22, 0x22]), response=False)
                        await client.start_notify(DATA_UUID, self._notification_handler)
                        while client.is_connected:
                            await asyncio.sleep(1)
            except Exception as e:
                print(f"[BCI] Connection error: {e}")
            
            self.is_connected = False
            # ПРИНУДИТЕЛЬНО ОБНУЛЯЕМ ПРИ ПОТЕРЕ
            self.ciplv_vec.fill(0)
            self.persistence = 0
            await asyncio.sleep(2)

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
        if not self.is_connected or not BCI_LIBRARIES_AVAILABLE: return
        with self.lock: data = self.buffer.copy()
        band_data = lfilter(self.b_band, self.a_band, data, axis=1)
        ana = hilbert(band_data, axis=1)[:, -40:]
        raw_vec = np.zeros(28)
        for idx, (i, j) in enumerate(PAIRS):
            val = np.imag(np.mean(ana[i] * np.conj(ana[j])))
            raw_vec[idx] = val * (np.sqrt((ELECTRODES[j,0]-ELECTRODES[i,0])**2 + (ELECTRODES[j,1]-ELECTRODES[i,1])**2) / (RADIUS*2))
        
        smooth = 0.95
        self.ciplv_vec = self.ciplv_vec * smooth + (raw_vec * 0.5) * (1 - smooth)
        self.persistence = min(1.0, np.linalg.norm(self.ciplv_vec) * 10.0)

    def update_sim(self):
        # Если не подключен - ничего не делаем, ciplv_vec остается нулевым (зануляется в ble_manager)
        if self.is_connected:
            self.process_raw()
