#include <Arduino.h>
#include <NimBLEDevice.h>
#include "USB.h"
#include "USBHIDGamepad.h"
#include <algorithm>

// ===================== НАСТРОЙКИ =====================
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

#define CH_COUNT 8
#define BUF_SIZE 256

const float UV_SCALE = (1.2f / 4.0f / 8388607.0f) * 1000000.0f;
const float RADIUS = 10.0f;
const float ANGLES[8] = {-72, -36, 36, 72, 108, 144, -144, -108};

float EX[8], EY[8];

// === ПАРАМЕТРЫ СТАБИЛЬНОСТИ (подкручивай здесь) ===
float SKILL_LEVEL = 0.05f;     // 0.08–0.12 — самый комфортный диапазон
float STICK_GAIN   = 3.0f;     // 2.0 = очень чувствительно, 3.0 = спокойнее

// =====================================================

volatile int eeg_head = 0;
float eeg[CH_COUNT][BUF_SIZE];

float re[CH_COUNT][BUF_SIZE];
float im[CH_COUNT][BUF_SIZE];
float centered[CH_COUNT][BUF_SIZE];

float norm_re[CH_COUNT][37];
float norm_im[CH_COUNT][37];

float cX = 0, cY = 0, cTQ = 0, persistence = 0;
float lastTX = 0, lastTY = 0;

volatile bool dataReady = false;

USBHIDGamepad Gamepad;

unsigned long lastDebug = 0;
int processCounter = 0;

// ===================== БЫСТРЫЙ FFT =====================
void fft(float* vR, float* vI) {
    int n = BUF_SIZE;
    for (int i = 0, j = 0; i < n; i++) {
        if (j > i) { std::swap(vR[i], vR[j]); std::swap(vI[i], vI[j]); }
        int m = n >> 1; while (m >= 1 && j >= m) { j -= m; m >>= 1; } j += m;
    }
    for (int s = 2; s <= n; s <<= 1) {
        int m = s >> 1; float t = -2.0f * PI / s;
        float wr = cosf(t), wi = sinf(t);
        for (int b = 0; b < n; b += s) {
            float ar = 1, ai = 0;
            for (int j = 0; j < m; j++) {
                int u = b + j, v = u + m;
                float tr = ar * vR[v] - ai * vI[v];
                float ti = ar * vI[v] + ai * vR[v];
                vR[v] = vR[u] - tr; vI[v] = vI[u] - ti;
                vR[u] += tr; vI[u] += ti;
                float nr = ar * wr - ai * wi; ai = ar * wi + ai * wr; ar = nr;
            }
        }
    }
}

// ===================== ciPLV =====================
float get_ciPLV(int a, int b) {
    float sr = 0, si = 0;
    for (int k = 18; k <= 36; k++) {
        float uAr = norm_re[a][k], uAi = norm_im[a][k];
        float uBr = norm_re[b][k], uBi = norm_im[b][k];
        sr += (uAr * uBr + uAi * uBi);
        si += (uAi * uBr - uAr * uBi);
    }
    float mRe = sr / 19.0f;
    float mIm = si / 19.0f;
    float den = sqrtf(fmaxf(0.0f, 1.0f - mRe * mRe));
    return (den < 0.001f) ? 0.0f : mIm / den;
}

// ===================== ОСНОВНАЯ НЕЙРО-ОБРАБОТКА =====================
void processNeuro() {
    int current_head = eeg_head;

    // 1. CAR + FFT
    for (int c = 0; c < CH_COUNT; c++) {
        float sum = 0;
        for (int t = 0; t < BUF_SIZE; t++) {
            int idx = (current_head + t) % BUF_SIZE;
            centered[c][t] = eeg[c][idx];
            sum += centered[c][t];
        }
        float avg = sum / BUF_SIZE;
        for (int t = 0; t < BUF_SIZE; t++) {
            re[c][t] = centered[c][t] - avg;
            im[c][t] = 0.0f;
        }
        fft(re[c], im[c]);

        // Notch 50/100 Hz
        for (int k : {51, 102}) {
            for (int i = -1; i <= 1; i++) {
                if (k + i >= 0 && k + i < BUF_SIZE) re[c][k + i] = im[c][k + i] = 0.0f;
            }
        }
    }

    // 2. Нормализация
    for (int c = 0; c < CH_COUNT; c++) {
        for (int k = 18; k <= 36; k++) {
            float mag = sqrtf(re[c][k]*re[c][k] + im[c][k]*im[c][k]) + 1e-6f;
            norm_re[c][k] = re[c][k] / mag;
            norm_im[c][k] = im[c][k] / mag;
        }
    }

    // 3. Tensor + ciPLV
    float tx = 0, ty = 0, tq = 0, div = 0;
    for (int i = 0; i < CH_COUNT; i++) {
        for (int j = i + 1; j < CH_COUNT; j++) {
            float v = get_ciPLV(i, j);
            float dx = EX[j] - EX[i], dy = EY[j] - EY[i];
            tx += v * dx; ty += v * dy;
            tq += (v * (EX[i] * dy - EY[i] * dx)) * 0.01f;
            div += v * (dx * EX[i] + dy * EY[i]) / RADIUS;
        }
    }

    // 4. Persistence (медленнее затухает)
    float mag = sqrtf(tx*tx + ty*ty) + 1e-6f;
    float cosTheta = (tx*lastTX + ty*lastTY) / (mag * sqrtf(lastTX*lastTX + lastTY*lastTY) + 1e-6f);
    if (mag > 0.05f && cosTheta > 0.8f) persistence = fminf(1.0f, persistence + 0.05f);
    else persistence *= 0.97f;
    lastTX = tx; lastTY = ty;

    // 5. Физика — максимально плавная
    float sm = 0.985f - (SKILL_LEVEL * 0.05f);
    cX = cX * sm + tx * (SKILL_LEVEL * 1.5f) * (1.0f - sm);
    cY = cY * sm + ty * (SKILL_LEVEL * 1.5f) * (1.0f - sm);
    cTQ = cTQ * sm + tq * (SKILL_LEVEL * 0.75f) * (1.0f - sm);

    float b = 1.0f + persistence * 6.0f;

    int8_t lx = constrain((cX * b / STICK_GAIN) * 127.0f, -127, 127);
    int8_t ly = constrain((-cY * b / STICK_GAIN) * 127.0f, -127, 127);
    int8_t rtq_val = constrain((cTQ * b / STICK_GAIN) * 127.0f, -127, 127);

    Gamepad.leftStick(lx, ly);
    Gamepad.rightStick(rtq_val, 0);

    bool atk   = (div < -1.8f) && (persistence > 0.3f);
    bool inter = (div >  1.8f) && (persistence > 0.3f);

    if (atk)   Gamepad.pressButton(2);   else Gamepad.releaseButton(2);
    if (inter) Gamepad.pressButton(1);   else Gamepad.releaseButton(1);

    // === ДЕБАГ (раз в секунду) ===
    processCounter++;
    if (millis() - lastDebug > 1000) {
        Serial.printf("cX=%.2f  cY=%.2f  cTQ=%.2f  pers=%.2f  div=%.2f  SKILL=%.2f\n",
                      cX, cY, cTQ, persistence, div, SKILL_LEVEL);
        lastDebug = millis();
        processCounter = 0;
    }

    yield();
}

// ===================== BLE CALLBACK =====================
void notifyCallback(NimBLERemoteCharacteristic* pRemoteCharacteristic, uint8_t* pData, size_t length, bool isNotify) {
    if (length >= 33 && pData[0] == 0xA0) {
        for (int i = 0; i < CH_COUNT; i++) {
            int s = 2 + i * 3;
            int32_t v = (pData[s] << 16) | (pData[s + 1] << 8) | pData[s + 2];
            if (v & 0x800000) v -= 0x1000000;
            eeg[i][eeg_head] = (float)v * UV_SCALE;
        }
        eeg_head = (eeg_head + 1) % BUF_SIZE;
        dataReady = true;
    }
}

NimBLEClient* pClient = nullptr;
NimBLEAdvertisedDevice* targetDevice = nullptr;
bool doConnect = false;

class ClientCallbacks : public NimBLEClientCallbacks {
    void onConnect(NimBLEClient* pClient) override { 
        pClient->updatePhy(BLE_HCI_LE_PHY_2M_PREF_MASK, BLE_HCI_LE_PHY_2M_PREF_MASK);
    }
    void onDisconnect(NimBLEClient* pClient, int reason) override { 
        NimBLEDevice::getScan()->start(0, false);
    }
};

class ScanCallbacks : public NimBLEScanCallbacks {
    void onResult(const NimBLEAdvertisedDevice* advDevice) override {
        if (advDevice->isAdvertisingService(NimBLEUUID(SERVICE_UUID))) {
            targetDevice = new NimBLEAdvertisedDevice(*advDevice);
            doConnect = true;
            NimBLEDevice::getScan()->stop();
        }
    }
};

// ===================== SETUP & LOOP =====================
void setup() {
    Serial.begin(921600);

    for (int i = 0; i < 8; i++) {
        float a = ANGLES[i] * PI / 180.0f;
        EX[i] = cosf(a) * RADIUS;
        EY[i] = sinf(a) * RADIUS;
    }

    Gamepad.begin();
    USB.VID(0x045E);
    USB.PID(0x028E);
    USB.begin();

    // Маленький клик при старте (как в Python)
    Gamepad.pressButton(0);
    delay(80);
    Gamepad.releaseButton(0);

    NimBLEDevice::init("FreeEEG_Dongle");
    NimBLEDevice::setPower(ESP_PWR_LVL_P3);
    NimBLEDevice::setMTU(256);

    NimBLEScan* pScan = NimBLEDevice::getScan();
    pScan->setScanCallbacks(new ScanCallbacks());
    pScan->start(0, false);
}

void loop() {
    if (doConnect) {
        doConnect = false;
        pClient = NimBLEDevice::createClient();
        pClient->setClientCallbacks(new ClientCallbacks(), false);
        if (pClient->connect(targetDevice)) {
            NimBLERemoteService* pSvc = pClient->getService(SERVICE_UUID);
            if (pSvc) {
                NimBLERemoteCharacteristic* pChr = pSvc->getCharacteristic(CHARACTERISTIC_UUID);
                if (pChr) pChr->subscribe(true, notifyCallback);
            }
        }
        delete targetDevice;
        targetDevice = nullptr;
    }

    if (dataReady) {
        dataReady = false;
        static int throttle = 0;
        if (++throttle >= 4) {        // ~62 Гц — плавнее чем раньше
            processNeuro();
            throttle = 0;
        }
    }
}