#include <SPI.h>
#include <Arduino.h>
#include <esp_now.h>
#include <WiFi.h>

// ==========================================
// НАСТРОЙКА ПИНОВ (XIAO ESP32-C3)
// ==========================================
#define PIN_CS    D0
#define PIN_DRDY  D1
#define PIN_RESET D2
#define PIN_SCLK  D8
#define PIN_MISO  D9
#define PIN_MOSI  D10
#define PIN_CLK_OUT D3

// ==========================================
// КОМАНДЫ И РЕГИСТРЫ ADS131M08
// ==========================================
#define OPCODE_RREG     0xA000
#define OPCODE_WREG     0x6000
#define OPCODE_UNLOCK   0x0655
#define REG_CLOCK       0x03

#define SPI_SPEED       1000000 
const int CHANNEL_COUNT = 8;
volatile bool drdyTriggered = false;

// ==========================================
// СТРУКТУРА ДАННЫХ ESP-NOW
// ==========================================
typedef struct struct_message {
  int32_t channels[8];
} struct_message;

struct_message eegData;
esp_now_peer_info_t peerInfo;
uint8_t broadcastAddress[] = {0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF};

unsigned long lastPrintTime = 0;

void IRAM_ATTR onDrdy() {
  drdyTriggered = true;
}

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 2000); 
  
  Serial.println("Starting C3 Sender...");

  pinMode(PIN_CS, OUTPUT);
  pinMode(PIN_RESET, OUTPUT);
  pinMode(PIN_DRDY, INPUT_PULLUP);
  pinMode(PIN_CLK_OUT, OUTPUT);

  digitalWrite(PIN_CS, HIGH);
  digitalWrite(PIN_RESET, HIGH);
  digitalWrite(PIN_CLK_OUT, LOW);

  // 1. Запуск клока для АЦП (8.192 МГц)
  ledcAttach(PIN_CLK_OUT, 8192000, 1); 
  ledcWrite(PIN_CLK_OUT, 1);
  Serial.println("CLKIN started at 8.192 MHz");
  delay(50);

  // 2. Инициализация ESP-NOW
  WiFi.mode(WIFI_STA);
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  memcpy(peerInfo.peer_addr, broadcastAddress, 6);
  peerInfo.channel = 0;  
  peerInfo.encrypt = false;
  esp_now_add_peer(&peerInfo);

  // 3. Настройка SPI и АЦП
  SPI.begin(PIN_SCLK, PIN_MISO, PIN_MOSI, PIN_CS);
  Serial.println("Resetting ADS131M08...");
  hardwareReset();

  // Разблокируем регистры
  sendCommand(OPCODE_UNLOCK);

  // Настраиваем CLOCK регистр на OSR = 16384 (250 SPS)
  writeRegister(REG_CLOCK, 0x0F1E);
  Serial.println("ADC Configured for 250 SPS. Waiting for DRDY...");

  attachInterrupt(digitalPinToInterrupt(PIN_DRDY), onDrdy, FALLING);
}

void loop() {
  if (drdyTriggered) {
    drdyTriggered = false;
    readAndSendADC();
  }
}

// --- ФУНКЦИИ ---

void hardwareReset() {
  digitalWrite(PIN_RESET, LOW);
  delayMicroseconds(100);
  digitalWrite(PIN_RESET, HIGH);
  delay(10);
}

void sendCommand(uint16_t cmd) {
  SPI.beginTransaction(SPISettings(SPI_SPEED, MSBFIRST, SPI_MODE1));
  digitalWrite(PIN_CS, LOW);
  SPI.transfer16(cmd);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

void writeRegister(uint8_t address, uint16_t value) {
  uint16_t cmd = OPCODE_WREG | (address << 7) | 0;
  SPI.beginTransaction(SPISettings(SPI_SPEED, MSBFIRST, SPI_MODE1));
  digitalWrite(PIN_CS, LOW);
  SPI.transfer16(cmd);
  SPI.transfer16(value);
  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();
}

void readAndSendADC() {
  SPI.beginTransaction(SPISettings(SPI_SPEED, MSBFIRST, SPI_MODE1));
  digitalWrite(PIN_CS, LOW);

  // Читаем статус
  uint16_t status = SPI.transfer16(0x0000);
  
  // Читаем 8 каналов
  for (int i = 0; i < CHANNEL_COUNT; i++) {
    uint8_t b1 = SPI.transfer(0x00);
    uint8_t b2 = SPI.transfer(0x00);
    uint8_t b3 = SPI.transfer(0x00);

    int32_t val = (b1 << 16) | (b2 << 8) | b3;
    if (val & 0x800000) val |= 0xFF000000; // Расширение знака
    
    eegData.channels[i] = val; // Сохраняем в структуру
  }

  digitalWrite(PIN_CS, HIGH);
  SPI.endTransaction();

  // Отправляем пакет по воздуху
  esp_now_send(broadcastAddress, (uint8_t *) &eegData, sizeof(eegData));

  // Раз в секунду печатаем, чтобы видеть, что не зависло
  if (millis() - lastPrintTime > 1000) {
    Serial.print("Status: "); Serial.print(status, HEX);
    Serial.print("\tCh0: "); Serial.print(eegData.channels[0]);
    Serial.print("\tCh7: "); Serial.println(eegData.channels[7]);
    lastPrintTime = millis();
  }
}