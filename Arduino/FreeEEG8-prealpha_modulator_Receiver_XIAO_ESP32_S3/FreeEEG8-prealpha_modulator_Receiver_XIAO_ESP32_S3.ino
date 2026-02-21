#include <Arduino.h>
#include <driver/i2s.h>
#include <esp_now.h>
#include <WiFi.h>
#include <esp_wifi.h>

// ==========================================
// ПИНЫ АУДИО (XIAO ESP32-S3)
// ==========================================
#define I2S_BCLK       D5
#define I2S_LRCK       D4
#define I2S_MCLK       D8
#define I2S_DIN        D3
#define I2S_DOUT       D2

#define SAMPLE_RATE    48000
#define BUFFER_SIZE    64

// ==========================================
// ЭЭГ ДАННЫЕ
// ==========================================
typedef struct struct_message {
  int32_t channels[8];
} struct_message;

struct_message receivedData;

// Переменные для связи ядер (volatile)
volatile int32_t raw_eeg_ch0 = 0; 
volatile bool is_connected = false;
volatile unsigned long last_packet_time = 0;

// Плавный коэффициент для обработки звука (чтобы не было щелчков)
float smoothed_modulator = 0.0;

// Колбэк ESP-NOW
void OnDataRecv(const esp_now_recv_info_t *recv_info, const uint8_t *incomingData, int len) {
  if (len == sizeof(receivedData)) {
    memcpy(&receivedData, incomingData, sizeof(receivedData));
    
    // Берем данные с 0-го канала
    raw_eeg_ch0 = receivedData.channels[0]; 
    last_packet_time = millis();
    is_connected = true;
  }
}

void setup() {
  Serial.begin(115200);

  // 1. НАСТРОЙКА WI-FI И ESP-NOW (Фиксируем канал 1)
  WiFi.mode(WIFI_STA);
  esp_wifi_set_channel(1, WIFI_SECOND_CHAN_NONE);
  
  if (esp_now_init() != ESP_OK) {
    Serial.println("Error initializing ESP-NOW");
    return;
  }
  esp_now_register_recv_cb(OnDataRecv);

  // 2. НАСТРОЙКА I2S (АУДИО)
  i2s_config_t i2s_config = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_RX),
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
    .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S,
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, 
    .dma_buf_count = 4,
    .dma_buf_len = BUFFER_SIZE,
    .use_apll = true,
    .tx_desc_auto_clear = true
  };

  i2s_pin_config_t pin_config = {
    .mck_io_num = I2S_MCLK,
    .bck_io_num = I2S_BCLK,
    .ws_io_num = I2S_LRCK,
    .data_out_num = I2S_DOUT,
    .data_in_num = I2S_DIN
  };

  i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pin_config);

  Serial.println("Neuro-Pedal Ready! I2S and ESP-NOW running.");
}

// --- Глобальные переменные для фильтрации ЭЭГ ---
float dc_blocked_eeg = 0.0;
float prev_raw_eeg = 0.0;
float eeg_envelope = 0.0;

void loop() {
  int32_t sample_buffer[BUFFER_SIZE * 2];
  size_t bytes_read = 0;
  size_t bytes_written = 0;

  // Если пакеты перестали приходить
  if (millis() - last_packet_time > 200) {
    is_connected = false;
  }

  // 1. Читаем аудио с гитары
  i2s_read(I2S_NUM_0, sample_buffer, sizeof(sample_buffer), &bytes_read, portMAX_DELAY);
  int samples_count = bytes_read / 4;

  // 2. DSP ДЛЯ ЭЭГ (Выполняем один раз на аудио-буфер)
  float target_modulator = 1.0; 

  if (is_connected) {
    // А. Убираем постоянный ток (DC Offset) - простейший High-Pass фильтр
    float current_raw = (float)raw_eeg_ch0;
    dc_blocked_eeg = current_raw - prev_raw_eeg + 0.99 * dc_blocked_eeg;
    prev_raw_eeg = current_raw;

    // Б. Извлекаем огибающую (Envelope Follower) - сглаживаем скачки
    // 0.05 - это "скорость атаки/спада". Меньше число = более плавное дыхание.
    eeg_envelope += (abs(dc_blocked_eeg) - eeg_envelope) * 0.05;

    // В. Нормализуем в диапазон 0.0 ... 1.0. 
    // Делим на 1_000_000 (подберите это число! Если наводка слабая, уменьшите до 100_000)
    target_modulator = eeg_envelope / 1000000.0; 
    
    // Ограничитель
    if (target_modulator > 1.0) target_modulator = 1.0;
    
    // Минимальная громкость 10%, чтобы звук не пропадал
    target_modulator = 0.1 + (target_modulator * 0.9);
  }

  // 3. ОБРАБОТКА ЗВУКА С ПЛАВНЫМ СГЛАЖИВАНИЕМ
  for (int i = 0; i < samples_count; i++) {
    // Интерполируем коэффициент для каждого сэмпла звука (чтобы не было щелчков)
    smoothed_modulator += (target_modulator - smoothed_modulator) * 0.001;

    // Применяем громкость
    sample_buffer[i] = (int32_t)(sample_buffer[i] * smoothed_modulator);
  }

  // 4. Пишем аудио в ЦАП
  i2s_write(I2S_NUM_0, sample_buffer, bytes_read, &bytes_written, portMAX_DELAY);
}