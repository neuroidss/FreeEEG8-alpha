#include <Arduino.h>
#include <NimBLEDevice.h>

#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define LED_PIN 21

NimBLEClient* pClient = nullptr;
NimBLEAdvertisedDevice* targetDevice = nullptr;
bool doConnect = false;

// Коллбэк уведомлений: просто пересылаем всё в Serial
void notifyCallback(NimBLERemoteCharacteristic* pRemoteCharacteristic, uint8_t* pData, size_t length, bool isNotify) {
    // Пишем в USB все пришедшие байты. 
    // Если стек сгруппировал 2-3 пакета по 33 байта, Serial.write отправит их все разом.
    Serial.write(pData, length);
}

class ClientCallbacks : public NimBLEClientCallbacks {
    void onConnect(NimBLEClient* pClient) override { digitalWrite(LED_PIN, LOW); }
    void onDisconnect(NimBLEClient* pClient, int reason) override { 
        digitalWrite(LED_PIN, HIGH); 
        NimBLEDevice::getScan()->start(0, false);
    }
};

class ScanCallbacks : public NimBLEScanCallbacks {
    void onResult(const NimBLEAdvertisedDevice* advertisedDevice) override {
        if (advertisedDevice->isAdvertisingService(NimBLEUUID(SERVICE_UUID))) {
            targetDevice = new NimBLEAdvertisedDevice(*advertisedDevice);
            doConnect = true;
            NimBLEDevice::getScan()->stop();
        }
    }
};

void setup() {
    Serial.begin(921600);
    pinMode(LED_PIN, OUTPUT);
    digitalWrite(LED_PIN, HIGH);

    NimBLEDevice::init("FreeEEG_Dongle");
    NimBLEDevice::setMTU(256); // Позволит принимать несколько пакетов OpenBCI за один такт радио
    
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
                if (pChr && pChr->canNotify()) pChr->subscribe(true, notifyCallback);
            }
        }
        delete targetDevice;
        targetDevice = nullptr;
    }
}