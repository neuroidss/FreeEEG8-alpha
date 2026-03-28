#include <Arduino.h>
#include <SPI.h>
#include "ADS131M08.h"
#include <NimBLEDevice.h>

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define DATA_CHAR_UUID      "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define CMD_CHAR_UUID       "c0de0001-36e1-4688-b7f5-ea07361b26a8" // Характеристика прозрачного моста

// Пины
#define PIN_CS     D3  
#define PIN_DRDY   D4  
#define PIN_RESET  D2  
#define PIN_SCLK   D8  
#define PIN_MOSI   D9  
#define PIN_MISO   D10 
#define PIN_CLKOUT D0

ADS131M08 adc;
NimBLECharacteristic* pDataCharacteristic = nullptr;
NimBLECharacteristic* pCmdCharacteristic = nullptr;

volatile bool deviceConnected = false;
volatile bool drdyTriggered = false;

// Стандартный пакет OpenBCI Cyton
uint8_t obciPacket[33];
uint8_t sampleCounter = 0;

// === ФЛАГИ ПРОЗРАЧНОГО МОСТА ===
volatile bool has_pending_read = false;
volatile bool has_pending_write = false;
volatile uint8_t  reg_addr = 0;
volatile uint16_t reg_val = 0;
volatile uint16_t reg_mask = 0;
volatile bool use_mask = false;

void IRAM_ATTR onDrdy() { drdyTriggered = true; }

// === КОЛЛБЕКИ КОМАНД С МОБИЛКИ ===
class CmdCallbacks : public NimBLECharacteristicCallbacks {
    // В NimBLE v2.0+ обязательно нужно передавать NimBLEConnInfo& connInfo
    void onWrite(NimBLECharacteristic* pChar, NimBLEConnInfo& connInfo) override {
        std::string rx = pChar->getValue();
        if (rx.length() == 1) {
            reg_addr = rx[0];
            has_pending_read = true;
        } else if (rx.length() == 3) {
            reg_addr = rx[0];
            reg_val = (rx[1] << 8) | rx[2];
            use_mask = false;
            has_pending_write = true;
        } else if (rx.length() == 5) {
            reg_addr = rx[0];
            reg_val = (rx[1] << 8) | rx[2];
            reg_mask = (rx[3] << 8) | rx[4];
            use_mask = true;
            has_pending_write = true;
        }
    }
};

class MyServerCallbacks : public NimBLEServerCallbacks {
    void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
        deviceConnected = true;
        // Максимально быстрый интервал (7.5 мс) для стриминга 500 SPS
        pServer->updateConnParams(connInfo.getConnHandle(), 6, 6, 0, 100);
    }
    void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
        deviceConnected = false;
        NimBLEDevice::startAdvertising();
    }
};

void setup() {
    Serial.begin(115200);

    adc.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS, PIN_DRDY, PIN_RESET);
    adc.reset();
    adc.setOsr(OSR_16384); // Дефолт: 250 SPS
    adc.writeRegisterMasked(0x08, 0x04, 0x000F); // DC Block > 1Hz

    // Инициализация пакета
    obciPacket[0] = 0xA0;
    obciPacket[32] = 0xC0;

    // BLE Инициализация
    NimBLEDevice::init("FreeEEG8");
    NimBLEDevice::setPower(ESP_PWR_LVL_P3); // +3 dBm
    
    NimBLEServer *pServer = NimBLEDevice::createServer();
    pServer->setCallbacks(new MyServerCallbacks());

    NimBLEService *pService = pServer->createService(SERVICE_UUID);
    
    pDataCharacteristic = pService->createCharacteristic(DATA_CHAR_UUID, NIMBLE_PROPERTY::NOTIFY);
    
    // Мост: Разрешаем WRITE (команды) и NOTIFY (ответы чтения регистров)
    pCmdCharacteristic = pService->createCharacteristic(CMD_CHAR_UUID, NIMBLE_PROPERTY::WRITE | NIMBLE_PROPERTY::NOTIFY);
    pCmdCharacteristic->setCallbacks(new CmdCallbacks());

    pService->start();

    NimBLEDevice::getAdvertising()->addServiceUUID(SERVICE_UUID);
    NimBLEDevice::getAdvertising()->start();

    pinMode(PIN_DRDY, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdy, FALLING);
}

void loop() {
    // 1. ОБРАБОТКА ЧТЕНИЯ РЕГИСТРА
    if (has_pending_read) {
        uint16_t val = adc.readRegister(reg_addr);
        uint8_t tx[3] = { reg_addr, (uint8_t)(val >> 8), (uint8_t)(val & 0xFF) };
        pCmdCharacteristic->notify(tx, 3);
        Serial.printf("Read Reg 0x%02X -> 0x%04X\n", reg_addr, val);
        has_pending_read = false;
    }

    // 2. ОБРАБОТКА ЗАПИСИ В РЕГИСТР
    if (has_pending_write) {
        if (use_mask) {
            adc.writeRegisterMasked(reg_addr, reg_val, reg_mask);
            Serial.printf("Write Masked: 0x%02X, Val 0x%04X, Mask 0x%04X\n", reg_addr, reg_val, reg_mask);
        } else {
            adc.writeRegister(reg_addr, reg_val);
            Serial.printf("Write Direct: 0x%02X, Val 0x%04X\n", reg_addr, reg_val);
        }
        has_pending_write = false;
    }

    // 3. ОТПРАВКА ДАННЫХ АЦП
    if (drdyTriggered) {
        drdyTriggered = false;
        AdcOutput raw = adc.readAdcRaw();

        obciPacket[1] = sampleCounter++;
        for (int i = 0; i < 8; i++) {
            int32_t v = raw.ch[i].i;
            obciPacket[2 + i*3 + 0] = (v >> 16) & 0xFF;
            obciPacket[2 + i*3 + 1] = (v >> 8) & 0xFF;
            obciPacket[2 + i*3 + 2] = v & 0xFF;
        }
        memset(&obciPacket[26], 0, 6);

        if (deviceConnected) {
            pDataCharacteristic->notify(obciPacket, 33);
        }
    }
}