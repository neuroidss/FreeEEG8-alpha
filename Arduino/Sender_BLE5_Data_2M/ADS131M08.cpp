#include "Arduino.h"
#include "ADS131M08.h"

// Define SPISettings here so it's consistent.
// 1MHz is a safe starting point, MSBFIRST is standard, SPI_MODE1 (CPOL=0, CPHA=1) as per datasheet 8.5.1 Interface
#define settings SPISettings(1000000, MSBFIRST, SPI_MODE1) 

ADS131M08::ADS131M08() : csPin(0), drdyPin(0), clkPin(0), misoPin(0), mosiPin(0), resetPin(0)
{
  for( uint16_t i = 0U; i < 8; i++){
    fullScale.ch[i].f = 1.2; // Default to +-1.2V, user can change with setFullScale
    pgaGain[i] = ADS131M08_PgaGain::PGA_1;
    resultFloat.ch[i].f = 0.0;
    resultRaw.ch[i].u[0] = 0U;
    resultRaw.ch[i].u[1] = 0U;
  }
  
}

// Helper for 24-bit SPI transfer
uint32_t ADS131M08::transfer24(uint32_t data) {
    uint8_t b1 = spi.transfer((data >> 16) & 0xFF);
    uint8_t b2 = spi.transfer((data >> 8) & 0xFF);
    uint8_t b3 = spi.transfer(data & 0xFF);
    return ((uint32_t)b1 << 16) | ((uint32_t)b2 << 8) | b3;
}


uint8_t ADS131M08::writeRegister(uint8_t address, uint16_t value)
{
  // The ADS131M08 expects 24-bit words by default.
  // Commands are 16-bit, so they are sent as `command << 8`.
  // Register values are 16-bit, also sent as `value << 8`.
  
  uint32_t cmd_word = ((uint32_t)CMD_WRITE_REG | (address << 7)) << 8; // 16-bit command + 7-bit address + 0 for N regs (default to 0 for single write)
  uint32_t data_word = ((uint32_t)value) << 8; // 16-bit value + 8 zeros (padding to 24 bits)
  
  // Start transaction
  spi.beginTransaction(settings);
  digitalWrite(csPin, LOW);
  delayMicroseconds(1); // tCSSC min 16ns

  // Send WRITE_REG command
  transfer24(cmd_word);

  // Send register value
  transfer24(data_word);

  // Send 8 dummy words (0x000000) for the rest of a 10-word frame as per datasheet 8.5.1.7
  for(int i = 0; i < 8; i++) { 
      transfer24(0x000000); // Send dummy 24-bit word
  }

  delayMicroseconds(1); // tCSH min 15ns (for 2.7-3.6V)
  digitalWrite(csPin, HIGH);
  spi.endTransaction();
  
  return 1; // Indicate one register was conceptually written
}

void ADS131M08::writeRegisterMasked(uint8_t address, uint16_t value, uint16_t mask)
{
  // Read current content
  uint16_t register_contents = readRegister(address);

  // Clear bits to be modified using NOT mask, then OR with new value
  register_contents = (register_contents & ~mask) | (value & mask); // Apply mask to value as well, just in case

  // Write back the modified content
  writeRegister(address, register_contents);
}

uint16_t ADS131M08::readRegister(uint8_t address)
{
  // For a single RREG command, the device responds with the register contents
  // in the *next* frame. (Datasheet 8.5.1.10.7.1 Reading a Single Register)
  
  uint32_t cmd_word = ((uint32_t)CMD_READ_REG | (address << 7)) << 8; // 16-bit command + 7-bit address + 0 for N regs (default to 0 for single read)
  uint32_t dummy_word = 0x000000;
  
  // --- First transaction: Send RREG command ---
  spi.beginTransaction(settings);
  digitalWrite(csPin, LOW);
  delayMicroseconds(1); // tCSSC

  transfer24(cmd_word); // Send RREG command. The response will be in the next frame.
  for (int i = 0; i < 9; i++) { // Fill the rest of the 10-word command frame with dummy data
      transfer24(dummy_word);
  }
  
  delayMicroseconds(1); // tCSH
  digitalWrite(csPin, HIGH);
  spi.endTransaction();

  // Wait for the device to process and prepare the response (typically 50us - 100us)
  // Datasheet 8.5.1.10.7.1 implies data is available in the *next* frame.
  delayMicroseconds(50); 

  // --- Second transaction: Read the response frame ---
  spi.beginTransaction(settings);
  digitalWrite(csPin, LOW);
  delayMicroseconds(1); // tCSSC

  uint32_t received_data_word = transfer24(dummy_word); // First word of the response frame is the register data

  for (int i = 0; i < 9; i++) { // Send 9 dummy words to complete the 10-word response frame.
      transfer24(dummy_word);
  }

  delayMicroseconds(1); // tCSH
  digitalWrite(csPin, HIGH);
  spi.endTransaction();

  // The register value is 16-bit, residing in the MSB of the 24-bit word.
  return (uint16_t)(received_data_word >> 8);
}

void ADS131M08::begin(uint8_t clk_pin, uint8_t miso_pin, uint8_t mosi_pin, uint8_t cs_pin, uint8_t drdy_pin, uint8_t reset_pin)
{
  // Set pins up
  this->csPin = cs_pin;
  this->drdyPin = drdy_pin;
  this->clkPin = clk_pin;
  this->misoPin = miso_pin;
  this->mosiPin = mosi_pin;
  this->resetPin = reset_pin;

  // For ESP32, use spi.begin(sclk, miso, mosi, ss) directly.
  // Pass -1 for SS (cs_pin) as the library handles it manually.
  // This explicitly sets the pins for the default SPI bus (VSPI on ESP32 by default for SPIClass).
  // The SPIClass instance is already created as a member 'spi'.
  this->spi.begin(clk_pin, miso_pin, mosi_pin, -1); 

  // beginTransaction will be called before each specific SPI operation.
  // No need to call it here globally.

  // Configure chip select as an output and ensure it's HIGH (inactive)
  pinMode(this->csPin, OUTPUT);
  digitalWrite(this->csPin, HIGH);
  
  // Configure RESET as an output
  pinMode(this->resetPin, OUTPUT);
  
  // Configure DRDY as an input
  pinMode(this->drdyPin, INPUT);
}

int8_t ADS131M08::isDataReadySoft(byte channel)
{
  // This function reads the STATUS register and checks the DRDY bit for the specified channel.
  // The ADS131M08_PgaGain enum is not used here.
  uint16_t status_reg = readRegister(REG_STATUS);
  switch(channel) {
    case 0: return (status_reg & REGMASK_STATUS_DRDY0) ? 1 : 0;
    case 1: return (status_reg & REGMASK_STATUS_DRDY1) ? 1 : 0;
    case 2: return (status_reg & REGMASK_STATUS_DRDY2) ? 1 : 0;
    case 3: return (status_reg & REGMASK_STATUS_DRDY3) ? 1 : 0;
    case 4: return (status_reg & REGMASK_STATUS_DRDY4) ? 1 : 0;
    case 5: return (status_reg & REGMASK_STATUS_DRDY5) ? 1 : 0;
    case 6: return (status_reg & REGMASK_STATUS_DRDY6) ? 1 : 0;
    case 7: return (status_reg & REGMASK_STATUS_DRDY7) ? 1 : 0;
    default: return -1; // Invalid channel
  }
}

bool ADS131M08::isResetStatus(void)
{
  return (readRegister(REG_STATUS) & REGMASK_STATUS_RESET);
}

bool ADS131M08::isLockSPI(void)
{
  return (readRegister(REG_STATUS) & REGMASK_STATUS_LOCK);
}

bool ADS131M08::setDrdyFormat(uint8_t drdyFormat)
{
  if (drdyFormat > 1)
  {
    return false;
  }
  else
  {
    writeRegisterMasked(REG_MODE, drdyFormat, REGMASK_MODE_DRDY_FMT);
    return true;
  }
}

bool ADS131M08::setDrdyStateWhenUnavailable(uint8_t drdyState)
{
  if (drdyState > 1)
  {
    return false;
  }
  else
  {
    writeRegisterMasked(REG_MODE, (drdyState == DRDY_STATE_HI_Z) ? REGMASK_MODE_DRDY_HiZ : 0, REGMASK_MODE_DRDY_HiZ);
    return true;
  }
}

bool ADS131M08::setPowerMode(uint8_t powerMode)
{
  if (powerMode > 3) // PWR bits are 0-3
  {
    return false;
  }
  else
  {
    writeRegisterMasked(REG_CLOCK, powerMode, REGMASK_CLOCK_PWR);
    return true;
  }
}

bool ADS131M08::setOsr(uint16_t osr)
{
  if (osr > 7) // OSR bits are 0-7
  {
    return false;
  }
  else
  {
    writeRegisterMasked(REG_CLOCK, osr << 2 , REGMASK_CLOCK_OSR);
    return true;
  }
}

void ADS131M08::setFullScale(uint8_t channel, float scale)
{
  if (channel > 7) {
    return;
  }

  this->fullScale.ch[channel].f = scale;
  
}

float ADS131M08::getFullScale(uint8_t channel)
{
  if (channel > 7) {
    return 0.0;
  }

  return this->fullScale.ch[channel].f;
  
}

void ADS131M08::reset()
{
  digitalWrite(this->resetPin, LOW);
  delay(10); // Minimum 10ms for hardware reset (tW(RSL) in datasheet 6.6)
  digitalWrite(this->resetPin, HIGH);
  delay(15); // Wait tPOR + tREGACQ after SYNC/RESET goes high, approx 250us + 5us (datasheet 6.7, 8.4.1.2)
             // Using 15ms here, similar to your sketch's longer delay for stability.
  delay(1000); // Additional delay as in your .ino for stability
}

bool ADS131M08::setChannelEnable(uint8_t channel, uint16_t enable)
{
  if (channel > 7)
  {
    return false;
  }
  // The CHx_EN bits are from bit 8 (CH0_EN) to bit 15 (CH7_EN) in REG_CLOCK
  uint16_t channel_bit_shift = 8 + channel; 
  uint16_t value_to_write = (enable & 0x01) << channel_bit_shift; // Ensure 'enable' is 0 or 1
  uint16_t mask = 1 << channel_bit_shift; // Mask for the specific channel enable bit

  uint8_t reg_address = REG_CLOCK; // FIXED: Declare reg_address here

  writeRegisterMasked(reg_address, value_to_write, mask);
  return true;
}

bool ADS131M08::setChannelPGA(uint8_t channel, ADS131M08_PgaGain pga)
{ uint16_t pgaCode = (uint16_t) pga;

  if (channel > 7)
  {
    return false;
  }
  
  uint8_t reg_address;
  uint16_t value_to_write;
  uint16_t mask;

  if (channel <= 3) {
      reg_address = REG_GAIN1;
      // PGA gain bits are 3 bits wide (0-7), shifted by 4*channel
      value_to_write = (pgaCode & 0x07) << (channel * 4); 
      mask = (uint16_t)0x0007 << (channel * 4); // Corresponding mask
  } else { // channels 4-7 use REG_GAIN2
      reg_address = REG_GAIN2;
      value_to_write = (pgaCode & 0x07) << ((channel - 4) * 4); // Adjust channel index for GAIN2 register
      mask = (uint16_t)0x0007 << ((channel - 4) * 4);
  }

  writeRegisterMasked(reg_address, value_to_write, mask);
  this->pgaGain[channel] = pga;
  return true;
}

ADS131M08_PgaGain ADS131M08::getChannelPGA(uint8_t channel)
{
  if(channel > 7)
  {
    return ADS131M08_PgaGain::PGA_INVALID;
  }
  return this->pgaGain[channel];
}

void ADS131M08::setGlobalChop(uint16_t global_chop)
{
  // GC_EN is bit 8 in REG_CFG
  writeRegisterMasked(REG_CFG, (global_chop & 0x01) << 8, REGMASK_CFG_GC_EN);
}

void ADS131M08::setGlobalChopDelay(uint16_t delay)
{
  // GC_DLY[3:0] are bits 12:9 in REG_CFG
  writeRegisterMasked(REG_CFG, (delay & 0x0F) << 9, REGMASK_CFG_GC_DLY);
}

bool ADS131M08::setInputChannelSelection(uint8_t channel, uint8_t input)
{
  if (channel > 7)
  {
    return false;
  }
  // The CHx_CFG registers are spaced by 5 addresses (e.g., REG_CH0_CFG=0x09, REG_CH1_CFG=0x0E)
  uint8_t reg_address = REG_CH0_CFG + (channel * 5); 
  // MUX bits are 1:0 in CHx_CFG registers
  writeRegisterMasked(reg_address, (input & 0x03), REGMASK_CHX_CFG_MUX);
  return true;
}

bool ADS131M08::setChannelOffsetCalibration(uint8_t channel, int32_t offset)
{
  if (channel > 7)
  {
    return false;
  }
  // CHx_OCAL_MSB and CHx_OCAL_LSB registers are spaced by 5 addresses from the base CH0_OCAL_MSB/LSB
  uint8_t reg_ocal_msb = REG_CH0_OCAL_MSB + (channel * 5);
  uint8_t reg_ocal_lsb = REG_CH0_OCAL_LSB + (channel * 5);

  // OCAL_MSB is bits 23:8 of the 24-bit offset, OCAL_LSB is bits 7:0.
  // The registers store 16 bits (MSB) and 8 bits (LSB) respectively.
  // MSB register (16 bits): stores bits 23:8 of the offset value
  writeRegisterMasked(reg_ocal_msb, (offset >> 8) & 0xFFFF, 0xFFFF); 
  // LSB register (bits 15:8): stores bits 7:0 of the offset value (shifted by 8 to occupy MSB of the 16-bit register)
  writeRegisterMasked(reg_ocal_lsb, (offset & 0xFF) << 8, REGMASK_CHX_OCAL0_LSB); 
  return true;
}

bool ADS131M08::setChannelGainCalibration(uint8_t channel, uint32_t gain)
{
  if (channel > 7)
  {
    return false;
  }
  // CHx_GCAL_MSB and CHx_GCAL_LSB registers are spaced by 5 addresses from the base CH0_GCAL_MSB/LSB
  uint8_t reg_gcal_msb = REG_CH0_GCAL_MSB + (channel * 5);
  uint8_t reg_gcal_lsb = REG_CH0_GCAL_LSB + (channel * 5); // FIXED: Corrected from REG_CH0_OCAL_LSB

  // GCAL_MSB is bits 23:8 of the 24-bit gain, GCAL_LSB is bits 7:0.
  // The registers store 16 bits (MSB) and 8 bits (LSB) respectively.
  // MSB register (16 bits): stores bits 23:8 of the gain value
  writeRegisterMasked(reg_gcal_msb, (gain >> 8) & 0xFFFF, 0xFFFF); 
  // LSB register (bits 15:8): stores bits 7:0 of the gain value (shifted by 8 to occupy MSB of the 16-bit register)
  writeRegisterMasked(reg_gcal_lsb, (gain & 0xFF) << 8, REGMASK_CHX_GCAL0_LSB); 
  return true;
}

bool ADS131M08::isDataReady()
{
  // This checks the physical DRDY pin state directly.
  return (digitalRead(drdyPin) == LOW); // DRDY is active LOW
}

uint16_t ADS131M08::getId()
{
  return readRegister(REG_ID);
}

uint16_t ADS131M08::getModeReg()
{
  return readRegister(REG_MODE);
}

uint16_t ADS131M08::getClockReg()
{
  return readRegister(REG_CLOCK);
}

uint16_t ADS131M08::getCfgReg()
{
  return readRegister(REG_CFG);
}

AdcOutput ADS131M08::readAdcRaw(void)
{
  // This function reads 10 24-bit words as a full frame in continuous conversion mode:
  // Status (1 word), Channel 0-7 data (8 words), CRC (1 word) - if CRC is enabled in MODE.
  // If CRC is not enabled (default), the last word will be 0x000000.
  // (Datasheet 8.5.1.9 ADC Conversion Data and Figure 8-18 Typical Communication Frame)

  // A full frame consists of 10 24-bit words, which is 30 bytes (10 * 3 bytes).
  const size_t frame_bytes = 10 * 3; 
  // Added __attribute__((aligned(4))) for DMA buffer alignment
  __attribute__((aligned(4))) uint8_t tx_buffer[frame_bytes] = {0}; // Transmit dummy bytes (all zeros)
  __attribute__((aligned(4))) uint8_t rx_buffer[frame_bytes];       // Buffer to receive data

  spi.beginTransaction(settings);
  digitalWrite(csPin, LOW);
  delayMicroseconds(1); // tCSSC

#ifdef USE_DMA_FOR_ADC_READ
  delayMicroseconds(1); // Added small delay before DMA transfer
  spi.transferBytes(tx_buffer, rx_buffer, frame_bytes);
#else
  // Original byte-by-byte transfer for compatibility or if DMA is not desired/available
  for (size_t i = 0; i < frame_bytes; i++) {
      rx_buffer[i] = spi.transfer(tx_buffer[i]);
  }
#endif

  delayMicroseconds(1); // tCSH
  digitalWrite(csPin, HIGH);
  spi.endTransaction();

  // Parse received data from rx_buffer
  // First word (bytes 0, 1, 2) is Status
  uint32_t received_word;
  received_word = ((uint32_t)rx_buffer[0] << 16) | ((uint32_t)rx_buffer[1] << 8) | rx_buffer[2];
  this->resultRaw.status = (uint16_t)(received_word >> 8); // Status is 16-bit, residing in MSB of 24-bit word

  // Next 8 words (bytes 3 to 26) are Channel data (8 channels * 3 bytes/channel)
  for(int i = 0; i < 8; i++)
  {
    size_t byte_offset = 3 + (i * 3); // Start of current channel's 3 bytes
    received_word = ((uint32_t)rx_buffer[byte_offset] << 16) | 
                    ((uint32_t)rx_buffer[byte_offset + 1] << 8) | 
                    rx_buffer[byte_offset + 2];
    
    // ADC data is 24-bit, stored as two's complement.
    // We need to sign-extend it correctly from 24-bit to 32-bit signed int.
    int32_t val = (int32_t)received_word;
    if (val & 0x800000) { // If 24th bit (MSB of 24-bit number) is set
      val |= 0xFF000000; // Sign extend to 32-bit by filling upper 8 bits with 1s
    } else {
      val &= 0x00FFFFFF; // Ensure upper bits are clear if positive
    }
    this->resultRaw.ch[i].i = val;
  }
  
  // The last word (bytes 27, 28, 29) is CRC or dummy data. We don't store it here.

  return this->resultRaw;
}

float ADS131M08::scaleResult(uint8_t num)
{
  if( num >= 8) {
    return 0.0;
  }
  
  float pga_val;
  switch(this->pgaGain[num]) {
      case ADS131M08_PgaGain::PGA_1: pga_val = 1.0; break;
      case ADS131M08_PgaGain::PGA_2: pga_val = 2.0; break;
      case ADS131M08_PgaGain::PGA_4: pga_val = 4.0; break;
      case ADS131M08_PgaGain::PGA_8: pga_val = 8.0; break;
      case ADS131M08_PgaGain::PGA_16: pga_val = 16.0; break;
      case ADS131M08_PgaGain::PGA_32: pga_val = 32.0; break;
      case ADS131M08_PgaGain::PGA_64: pga_val = 64.0; break;
      case ADS131M08_PgaGain::PGA_128: pga_val = 128.0; break;
      default: pga_val = 1.0; break; // Default to 1 if invalid PGA (shouldn't happen with proper init)
  }

  const float VREF = 1.2; // Internal reference voltage
  const float MAX_RAW_COUNT_POS = 8388607.0; // 2^23 - 1 (Max positive 24-bit value for 2's complement)

  // Calculate LSB value in Volts based on VREF and PGA gain
  // FSR = +/- VREF / PGA (for bipolar mode)
  // LSB = (FSR / 2) / (2^23 - 1)  = (VREF / PGA) / (2^23 - 1)
  float lsb_value_volts = (VREF / pga_val) / MAX_RAW_COUNT_POS;
  
  // Scale the raw integer result to a float voltage
  // Apply fullScale factor if set by the user, otherwise it's 1.2 (default)
  return this->resultFloat.ch[num].f = (float)this->resultRaw.ch[num].i * lsb_value_volts;
}

AdcOutput ADS131M08::scaleResult(void)
{
  // update status 
  this->resultFloat.status = this->resultRaw.status;
  // Scale all channels
  for(int i = 0; i<8; i++)
  {
    this->scaleResult(i);
  }

  return this->resultFloat;
}

AdcOutput ADS131M08::readAdcFloat(void)
{
  this->readAdcRaw();
  return this->scaleResult();
}
