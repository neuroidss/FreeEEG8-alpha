#include <Arduino.h>
#include <SPI.h>
#include "ADS131M08.h"
#include <NimBLEDevice.h>

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// Пины
#define PIN_CS    D0  
#define PIN_DRDY  D1  
#define PIN_RESET D2  
#define PIN_SCLK  D8  
#define PIN_MISO  D9  
#define PIN_MOSI  D10 
#define PIN_CLKOUT D3 

ADS131M08 adc;
NimBLECharacteristic* pDataCharacteristic = nullptr;
volatile bool deviceConnected = false;
volatile bool drdyTriggered = false;

// Стандартный пакет OpenBCI Cyton
uint8_t obciPacket[33];
uint8_t sampleCounter = 0;

void IRAM_ATTR onDrdy() { drdyTriggered = true; }

void setup() {
    Serial.begin(115200);
    
    // Клок АЦП
//    ledcAttach(PIN_CLKOUT, 8192000, 1);
//    ledcWrite(PIN_CLKOUT, 1);
//    delay(100);

    adc.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS, PIN_DRDY, PIN_RESET);
    adc.reset();
    adc.setOsr(OSR_16384); // 250 SPS - идеал по шуму
    adc.writeRegisterMasked(0x08, 0x04, 0x000F); // DC Block > 1Hz

    // Инициализация пакета
    obciPacket[0] = 0xA0;  // Header
    obciPacket[32] = 0xC0; // Footer

    // BLE
    NimBLEDevice::init("FreeEEG8");
    NimBLEDevice::setPower(ESP_PWR_LVL_N12); 
    NimBLEServer *pServer = NimBLEDevice::createServer();
    
    // Коллбэк для настройки интервала
    class MyServerCallbacks : public NimBLEServerCallbacks {
        void onConnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo) override {
            deviceConnected = true;
            // Устанавливаем минимальный интервал 7.5мс, чтобы пакеты не копились долго
            pServer->updateConnParams(connInfo.getConnHandle(), 6, 6, 0, 100);
        }
        void onDisconnect(NimBLEServer* pServer, NimBLEConnInfo& connInfo, int reason) override {
            deviceConnected = false;
            NimBLEDevice::startAdvertising();
        }
    };
    pServer->setCallbacks(new MyServerCallbacks());

    NimBLEService *pService = pServer->createService(SERVICE_UUID);
    pDataCharacteristic = pService->createCharacteristic(CHARACTERISTIC_UUID, NIMBLE_PROPERTY::NOTIFY);
    pService->start();

    NimBLEDevice::getAdvertising()->addServiceUUID(SERVICE_UUID);
    NimBLEDevice::getAdvertising()->start();

    pinMode(PIN_DRDY, INPUT_PULLUP);
    attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdy, FALLING);
}

void loop() {
    if (drdyTriggered) {
        drdyTriggered = false;
        AdcOutput raw = adc.readAdcRaw();

        // Формируем пакет OpenBCI
        obciPacket[1] = sampleCounter++; // ID 0-255

        for (int i = 0; i < 8; i++) {
            int32_t v = raw.ch[i].i;
            obciPacket[2 + i*3 + 0] = (v >> 16) & 0xFF;
            obciPacket[2 + i*3 + 1] = (v >> 8) & 0xFF;
            obciPacket[2 + i*3 + 2] = v & 0xFF;
        }
        
        // Очищаем Aux
        memset(&obciPacket[26], 0, 6);

        if (deviceConnected) {
            // ОТПРАВЛЯЕМ МГНОВЕННО (33 байта)
            pDataCharacteristic->notify(obciPacket, 33);
        }
    }
}