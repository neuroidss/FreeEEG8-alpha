#include <Arduino.h>
#include <driver/i2s.h>
#include <NimBLEDevice.h>
#include <math.h>
#include "esp_dsp.h" // Библиотека для аппаратного FFT

// ==========================================
// ПИНЫ АУДИО (Твои пины для XIAO ESP32-S3)
// ==========================================
#define I2S_BCLK       D5
#define I2S_LRCK       D4
#define I2S_MCLK       D8 // Опционально, если кодек требует Master Clock
#define I2S_DIN        D3 // Вход с АЦП (Гитара)
#define I2S_DOUT       D2 // Выход на ЦАП (На комбик)

#define SAMPLE_RATE    44100
#define BUFFER_SIZE    128 // Уменьшаем для меньшей задержки

// ==========================================
// BLE НАСТРОЙКИ
// ==========================================
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// === ГЛОБАЛЬНЫЕ ПЕРЕМЕННЫЕ (для обмена между ядрами) ===
volatile bool is_connected = false;
volatile unsigned long last_packet_time = 0;

// Структура для нейро-параметров (заполняется на Ядре 0, читается на Ядре 1)
struct NeuroParams {
    float deltaAmp, thetaAmp, alphaAmp, gammaAmp;
    float leadX, leadY;
    // Упрощенные фазы для триггеров (0.0 -> 1.0)
    float deltaPhase, thetaPhase; 
};
volatile NeuroParams neuroParams;

// Структура для параметров инструмента (заполняется на Ядре 1, читается на Ядре 1)
struct InstrumentParams {
    float rhythm;   // Сила удара (0.0 - 1.0)
    float tonality; // Высота тона/спектр (0.0 - 1.0)
};
volatile InstrumentParams instrumentParams;

// === ГЕОМЕТРИЯ СЕНСОРА (Твоя, шаг 36 градусов) ===
const float COS_TABLE[8] = { 0.809, 0.309, -0.309, -0.809, -0.809, -0.309, 0.309, 0.809 };
const float SIN_TABLE[8] = { 0.588, 0.951, 0.951, 0.588, -0.588, -0.951, -0.951, -0.588 };

// ==========================================
// DSP: ЭЭГ ОБРАБОТКА (IIR-ФИЛЬТРЫ) - ЯДРО 0
// ==========================================
// Biquad фильтры (pre-calculated for 250Hz sample rate)
float b_delta[3] = {0.0025, 0, -0.0025}; float a_delta[3] = {1, -1.99, 0.99};
float b_theta[3] = {0.014, 0, -0.014};  float a_theta[3] = {1, -1.97, 0.97};
float b_alpha[3] = {0.02, 0, -0.02};    float a_alpha[3] = {1, -1.90, 0.95};
float b_gamma[3] = {0.1, 0, -0.1};      float a_gamma[3] = {1, -1.17, 0.80};

// Состояния фильтров
float z_delta[8][2] = {0}, z_theta[8][2] = {0}, z_alpha[8][2] = {0}, z_gamma[8][2] = {0};
float prev_delta_val = 0, prev_theta_val = 0;

void processEEGPacket(const uint8_t* pkt) {
    float raw[8], mean = 0;
    for(int i=0; i<8; i++) {
        int32_t v = (pkt[2+i*3]<<16) | (pkt[3+i*3]<<8) | pkt[4+i*3];
        if(v & 0x800000) v -= 0x1000000;
        raw[i] = (float)v * ((1.2 / 32.0 / 8388607.0) * 1e6);
        mean += raw[i];
    }
    mean /= 8.0;

    float global_signal = 0;
    float vx = 0, vy = 0;
    for(int i=0; i<8; i++) {
        global_signal += raw[i];
        const float centered = raw[i] - mean;
        vx += centered * COS_TABLE[i];
        vy += centered * SIN_TABLE[i];
    }
    global_signal /= 8.0;
    
    // Фильтрация и вычисление амплитуд
    float delta_out=0, theta_out=0, alpha_out=0, gamma_out=0;
    
    // IIR Biquad filter implementation (Direct Form I)
    auto filter_channel = [&](float input, float b[], float a[], float z[]) {
        float out = b[0] * input + z[0];
        z[0] = b[1] * input - a[1] * out + z[1];
        z[1] = b[2] * input - a[2] * out;
        return out;
    };
    
    // Delta and Theta use global signal
    delta_out = filter_channel(global_signal, b_delta, a_delta, z_delta[0]);
    theta_out = filter_channel(global_signal, b_theta, a_theta, z_theta[0]);

    // Alpha and Gamma average over channels
    float alpha_sum = 0, gamma_sum = 0;
    for(int i=0; i<8; i++) {
        alpha_sum += abs(filter_channel(raw[i] - mean, b_alpha, a_alpha, z_alpha[i]));
        gamma_sum += abs(filter_channel(raw[i] - mean, b_gamma, a_gamma, z_gamma[i]));
    }
    alpha_out = alpha_sum / 8.0;
    gamma_out = gamma_sum / 8.0;

    // Сглаживание амплитуд (Envelope Follower)
    neuroParams.deltaAmp = neuroParams.deltaAmp * 0.9 + abs(delta_out) * 0.1 * 5.0;
    neuroParams.thetaAmp = neuroParams.thetaAmp * 0.9 + abs(theta_out) * 0.1 * 3.0;
    neuroParams.alphaAmp = neuroParams.alphaAmp * 0.9 + alpha_out * 0.1 * 2.0;
    neuroParams.gammaAmp = neuroParams.gammaAmp * 0.9 + gamma_out * 0.1 * 1.5;

    // Сглаживание вектора
    neuroParams.leadX = neuroParams.leadX * 0.95 + vx * 0.05;
    neuroParams.leadY = neuroParams.leadY * 0.95 + vy * 0.05;
    
    // Упрощенные триггеры фазы через пересечение нуля
    if(delta_out > 0 && prev_delta_val <= 0) neuroParams.deltaPhase = 1.0; else neuroParams.deltaPhase *= 0.9;
    if(theta_out > 1.57 && prev_theta_val <= 1.57) neuroParams.thetaPhase = 1.0; else neuroParams.thetaPhase *= 0.9;
    prev_delta_val = delta_out;
    prev_theta_val = theta_out;
}

// ==========================================
// BLE КОЛЛБЭКИ И ЗАДАЧА (ЯДРО 0)
// ==========================================
void notifyCallback(NimBLERemoteCharacteristic* pChr, uint8_t* pData, size_t length, bool isNotify) {
    // Просто перекидываем данные в буфер
    // Парсинг будет происходить в основном цикле задачи, чтобы не блокировать BLE
}

class ClientCallbacks : public NimBLEClientCallbacks {
    void onConnect(NimBLEClient* pClient) { is_connected = true; }
    void onDisconnect(NimBLEClient* pClient, int reason) { 
        is_connected = false;
        NimBLEDevice::getScan()->start(0, false);
    }
};

class ScanCallbacks : public NimBLEScanCallbacks {
    void onResult(const NimBLEAdvertisedDevice* advertisedDevice) {
        if (advertisedDevice->isAdvertisingService(NimBLEUUID(SERVICE_UUID))) {
            NimBLEDevice::getScan()->stop();
            NimBLEDevice::createClient()->connect(advertisedDevice);
        }
    }
};

void bleTask(void *pvParameters) {
    NimBLEDevice::init("ESP32-NeuroSynth");
    NimBLEClient* pClient = NimBLEDevice::createClient();
    pClient->setClientCallbacks(new ClientCallbacks());

    NimBLEScan* pScan = NimBLEDevice::getScan();
    pScan->setScanCallbacks(new ScanCallbacks());
    pScan->setActiveScan(true);
    pScan->start(0, false);

    while(1) {
        if(is_connected) {
             // Здесь можно добавить логику переподключения или проверки связи
            if(millis() - last_packet_time > 1000) {
                is_connected = false;
                pClient->disconnect();
            }
        }
        vTaskDelay(100 / portTICK_PERIOD_MS);
    }
}

// ==========================================
// SETUP
// ==========================================
void setup() {
    Serial.begin(115200);
    
    // Запускаем BLE на Ядре 0
    xTaskCreatePinnedToCore(bleTask, "BLE", 4096, NULL, 5, NULL, 0);

    // Конфигурация I2S
    i2s_config_t i2s_config = {
        .mode = (i2s_mode_t)(I2S_MODE_MASTER | I2S_MODE_TX | I2S_MODE_RX),
        .sample_rate = SAMPLE_RATE,
        .bits_per_sample = I2S_BITS_PER_SAMPLE_32BIT,
        .channel_format = I2S_CHANNEL_FMT_RIGHT_LEFT,
        .communication_format = I2S_COMM_FORMAT_STAND_I2S,
        .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1, 
        .dma_buf_count = 8,
        .dma_buf_len = BUFFER_SIZE,
        .use_apll = true
    };
    i2s_pin_config_t pin_config = {
        .mck_io_num = I2S_MCLK, .bck_io_num = I2S_BCLK, .ws_io_num = I2S_LRCK,
        .data_out_num = I2S_DOUT, .data_in_num = I2S_DIN
    };
    i2s_driver_install(I2S_NUM_0, &i2s_config, 0, NULL);
    i2s_set_pin(I2S_NUM_0, &pin_config);
    
    // Инициализация ESP-DSP для FFT
    esp_err_t err = dsps_fft2r_init_fc32(NULL, CONFIG_DSP_MAX_FFT_SIZE);
    if (err != ESP_OK) {
        Serial.println("Not possible to initialize FFT. Check CONFIG_DSP_MAX_FFT_SIZE");
    }
}


// ==========================================
// LOOP (АУДИО ОБРАБОТКА - ЯДРО 1)
// ==========================================
void loop() {
    // Буферы для I2S и FFT
    // ВАЖНО: Для FFT буфер должен быть в 2 раза больше (Real + Imaginary)
    static int32_t i2s_buffer[BUFFER_SIZE * 2];
    static float fft_input[BUFFER_SIZE * 2]; // Увеличено в 2 раза
    static float window[BUFFER_SIZE];
    static bool window_inited = false;
    
    if (!window_inited) {
        dsps_wind_hann_f32(window, BUFFER_SIZE);
        window_inited = true;
    }

    size_t bytes_read, bytes_written;

    // Читаем из АЦП (гитара)
    i2s_read(I2S_NUM_0, i2s_buffer, sizeof(i2s_buffer), &bytes_read, portMAX_DELAY);
    
    // --- Анализ инструмента на ESP-DSP ---
    for(int i=0; i < BUFFER_SIZE; i++) {
        // Заполняем вещественную часть сигналом гитары, умноженным на окно
        float sample = (float)i2s_buffer[i*2] / 2147483648.0f;
        fft_input[i*2] = sample * window[i]; 
        fft_input[i*2 + 1] = 0; // Мнимая часть = 0
    }

    // Выполняем FFT (Complex Forward FFT)
    dsps_fft2r_fc32(fft_input, BUFFER_SIZE);
    dsps_bit_rev_fc32(fft_input, BUFFER_SIZE);
    
    // ИСПРАВЛЕННАЯ ФУНКЦИЯ: превращаем результат в спектр вещественных чисел
    // Если dsps_cplx2re_fc32 не видна, используем версию предложенную компилятором
    dsps_cplx2reC_fc32(fft_input, BUFFER_SIZE); 

    float energy = 0, centroid = 0, total_mag = 0;
    // После cplx2re первые BUFFER_SIZE/2 элементов содержат амплитуды частот
    for (int i = 0; i < BUFFER_SIZE/2; i++) {
        // Извлекаем магнитуду из упакованного комплексного результата
        float re = fft_input[i*2];
        float im = fft_input[i*2 + 1];
        float mag = sqrtf(re*re + im*im);
        
        energy += mag;
        if (mag > 0.001) { // Порог шума
            float freq = i * (float)SAMPLE_RATE / BUFFER_SIZE;
            centroid += freq * mag;
            total_mag += mag;
        }
    }
    
    if (total_mag > 0) centroid /= total_mag;
    energy /= (BUFFER_SIZE/2);

    // Сглаживаем и передаем в глобальные переменные
    instrumentParams.rhythm = instrumentParams.rhythm * 0.8 + fmin(energy * 15.0, 1.0) * 0.2;
    instrumentParams.tonality = instrumentParams.tonality * 0.9 + fmin(centroid / 1500.0, 1.0) * 0.1;
    
    // --- Синтез звука ---
    static float timer_kick=10, timer_bass=10;
    static float phase_lead_c=0, phase_lead_m=0;
    
    for (int i=0; i < BUFFER_SIZE; i++) {
        float mix_l = 0, mix_r = 0;
        float dt = 1.0 / SAMPLE_RATE;

        // KICK
        if (neuroParams.deltaPhase > 0.5 && timer_kick > 1.0) timer_kick = 0;
        if (timer_kick < 0.38) {
            float T = timer_kick;
            float decay = 16.0 - (instrumentParams.rhythm * 6.0);
            mix_l += sinf(6.2832 * (48*T - 3.5*expf(-55*T))) * expf(-decay*T) * fmax(0.6, fmin(neuroParams.deltaAmp * 2.2, 1.6)) * 1.4;
            timer_kick += dt;
        }

        // BASS
        if (neuroParams.thetaPhase > 0.5 && timer_bass > 1.0) timer_bass = 0;
        if (timer_bass < 0.48) {
            float T = timer_bass;
            float osc = 2 * ((fmod(T, 1.0)) - 0.5) * expf(-13 * T) * fmax(0.35, fmin(neuroParams.thetaAmp * 1.9, 1.0));
            float sat = 2.2 + neuroParams.thetaAmp * 5.0 + (instrumentParams.rhythm * 3.0);
            mix_l += tanhf(osc * sat) * 1.15;
            timer_bass += dt;
        }
        
        // ALPHA LEAD
        float lead_vol = fmax(0, fmin((neuroParams.alphaAmp - 0.5) * 6.0, 0.72));
        if (lead_vol > 0.01) {
            float slide_mod = instrumentParams.tonality * 100.0;
            float c_freq = (150 + neuroParams.leadX * 360) + slide_mod;
            phase_lead_c += c_freq * dt;
            phase_lead_m += (c_freq * 0.5) * dt;
            float fm_depth = 3.5 + (instrumentParams.rhythm * 2.5);
            mix_r += sinf(6.2832*phase_lead_c + fm_depth * sinf(6.2832*phase_lead_m)) * lead_vol * 0.58;
        }

        i2s_buffer[i*2]   = (int32_t)(tanhf(mix_l) * 2147483647.0f);
        i2s_buffer[i*2+1] = (int32_t)(tanhf(mix_r) * 2147483647.0f);
    }
    
    // Пишем в ЦАП (на комбик)
    i2s_write(I2S_NUM_0, i2s_buffer, bytes_read, &bytes_written, portMAX_DELAY);
}