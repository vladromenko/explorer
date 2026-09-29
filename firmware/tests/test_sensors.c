#include "board.h"
#include "adc.h"
#include "stm32h7xx_ll_adc.h"
#include "app_icm20948.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

SPI_HandleTypeDef hspi2;
ADC_HandleTypeDef hadc1;
volatile uint32_t board_rx_overruns[PORT_COUNT],board_uart_errors[PORT_COUNT];
static uint64_t now,available_at=650000,mag_sample_due;
static uint8_t registers[4][128],selected_bank,mag_mode,mag_register,mag_count;
static bool spi_failure,mag_present,mag_overflow,imu_quiet,adc_failure,gyro_wrong;
static bool mag_drdy;
static unsigned starts[2],emitted_lidar[2],imu_valid,mag_valid,battery_valid,imu_error,mag_error;
static unsigned resets,master_resets,mag_resets,calibrations;
static uint8_t diagnostics[64],rxbytes[2][4096];
static size_t rxused[2],rxread[2];
static unsigned chip_selected;
uint64_t board_time_us(void) { return now; }
void HAL_Delay(uint32_t ms) { assert(ms<=1); now+=(uint64_t)ms*1000; }
void HAL_GPIO_WritePin(void *port,uint16_t pin,unsigned state) {
    assert(port==GPIOB && pin==GPIO_PIN_12); chip_selected=state==GPIO_PIN_RESET;
}
static void reset_registers(void) {
    memset(registers,0,sizeof(registers)); registers[0][B0_WHO_AM_I]=ICM20948_ID;
    registers[0][B0_PWR_MGMT_1]=0x41; selected_bank=0;
}
static uint8_t external_register(uint8_t reg) {
    if(reg==MAG_WIA2) { return mag_present?AK09916_ID:0xff; }
    if(reg==MAG_CNTL2) { return mag_mode; }
    if(reg==MAG_ST1) { return mag_drdy?1:0; }
    if(reg>=MAG_HXL && reg<=MAG_HZH) { mag_drdy=false; return (uint8_t)(reg-0x10); }
    if(reg==MAG_ST2) { mag_drdy=false; return mag_overflow?8:0; }
    return 0;
}
HAL_StatusTypeDef HAL_SPI_TransmitReceive(SPI_HandleTypeDef *h,uint8_t *tx,uint8_t *rx,uint16_t n,uint32_t timeout) {
    assert(h==&hspi2 && chip_selected && timeout==2 && n<=25);
    if(spi_failure || now<available_at) { now+=2000; return HAL_TIMEOUT; }
    now+=100;
    if(mag_mode==8 && now>=mag_sample_due) { mag_drdy=true; mag_sample_due=now+10000; }
    memset(rx,0,n);
    if(tx[0]==REG_BANK_SEL && n==2) { selected_bank=tx[1]>>4; assert(selected_bank<4); }
    else if(tx[0]&0x80) {
        uint8_t reg=tx[0]&0x7f;
        for(unsigned i=1;i<n;++i) {
            uint8_t at=(uint8_t)(reg+i-1);
            if(selected_bank==0 && at==B0_INT_STATUS_1) { rx[i]=imu_quiet?0:1; }
            else if(selected_bank==0 && at>=B0_ACCEL_XOUT_H && at<=B0_TEMP_OUT_L) { rx[i]=(uint8_t)(at+10); }
            else if(selected_bank==0 && at==B0_I2C_MST_STATUS) { rx[i]=mag_present?0:1; }
            else if(selected_bank==0 && at>=B0_EXT_SLV_SENS_DATA_00 && at<B0_EXT_SLV_SENS_DATA_00+9) {
                assert(mag_count>at-B0_EXT_SLV_SENS_DATA_00);
                /* A repeated ST1..ST2 transfer loses ready status. Tests reject
                 * that former sequence rather than making its mock succeed. */
                assert(!(mag_register==MAG_ST1 && mag_count>1));
                rx[i]=external_register((uint8_t)(mag_register+at-B0_EXT_SLV_SENS_DATA_00));
            } else if(selected_bank==2 && at==B2_GYRO_CONFIG_1 && gyro_wrong) { rx[i]=0; }
            else { rx[i]=registers[selected_bank][at]; }
        }
    } else {
        assert(n==2); registers[selected_bank][tx[0]]=tx[1];
        if(selected_bank==0 && tx[0]==B0_PWR_MGMT_1 && (tx[1]&0x80)) { ++resets; reset_registers(); }
        if(selected_bank==0 && tx[0]==B0_USER_CTRL && (tx[1]&2)) { ++master_resets; }
        if(selected_bank==3 && tx[0]==B3_I2C_SLV0_CTRL && (tx[1]&0x80)) {
            mag_register=registers[3][B3_I2C_SLV0_REG]; mag_count=tx[1]&15;
            if(!(registers[3][B3_I2C_SLV0_ADDR]&0x80) && mag_present) {
                if(mag_register==MAG_CNTL3) { ++mag_resets; mag_mode=0; }
                if(mag_register==MAG_CNTL2) { mag_mode=registers[3][B3_I2C_SLV0_DO]; }
            }
        }
    }
    return HAL_OK;
}
uint32_t LL_ADC_IsEnabled(ADC_TypeDef *instance) { assert(instance==hadc1.Instance); return 0; }
uint32_t LL_ADC_IsCalibrationOnGoing(ADC_TypeDef *instance) {
    assert(instance==hadc1.Instance); return calibrations && now<1000000;
}
void LL_ADC_StartCalibration(ADC_TypeDef *instance,uint32_t mode,uint32_t ended) {
    assert(instance==hadc1.Instance && mode==ADC_CALIB_OFFSET_LINEARITY && ended==ADC_SINGLE_ENDED);
    ++calibrations;
}
HAL_StatusTypeDef HAL_ADC_Start(ADC_HandleTypeDef *h) { assert(h==&hadc1); return HAL_OK; }
HAL_StatusTypeDef HAL_ADC_PollForConversion(ADC_HandleTypeDef *h,uint32_t timeout) {
    assert(h==&hadc1 && timeout==2); if(adc_failure) { now+=2000; return HAL_TIMEOUT; } return HAL_OK;
}
uint32_t HAL_ADC_GetValue(ADC_HandleTypeDef *h) { assert(h==&hadc1); return 3400; }
HAL_StatusTypeDef HAL_ADC_Stop(ADC_HandleTypeDef *h) { assert(h==&hadc1); return HAL_OK; }
bool board_receive(unsigned port,uint8_t *value) {
    assert(port==PORT_LIDAR0 || port==PORT_LIDAR1); unsigned i=port-PORT_LIDAR0;
    if(rxread[i]<rxused[i]) { *value=rxbytes[i][rxread[i]++]; return true; }
    rxread[i]=rxused[i]=0; return false;
}
bool board_send(unsigned port,const uint8_t *p,uint16_t length) {
    assert(port==PORT_LIDAR0 || port==PORT_LIDAR1);
    assert(length==2 && p[0]==0xa5 && p[1]==0x60); ++starts[port-PORT_LIDAR0]; return true;
}
bool board_emit(uint8_t type,const uint8_t *p,size_t length) {
    if(type==EW_IMU) {
        assert(length==44 && ew_u64(p)<=now && ew_u64(p+36)<=now);
        if(p[9]==0) { assert(now-ew_u64(p+36)<100000); }
        if(p[8]==0) { assert(!spi_failure && !imu_quiet && !gyro_wrong); ++imu_valid; }
        else { ++imu_error; for(unsigned i=10;i<24;++i) { assert(p[i]==0); } }
        if(p[9]==0) {
            assert(mag_present && !mag_overflow && mag_mode==8 && p[24]==1);
            for(unsigned i=1;i<=6;++i) { assert(p[24+i]==i); } ++mag_valid;
        } else { ++mag_error; }
    } else if(type==EW_BATTERY) {
        assert(length==15); if(p[8]==0) {
            assert(!adc_failure && now>=1000000 && ew_u16(p+9)==3400);
            assert(fabs(ew_f32(p+11)-3400*(3.30/4096)*4.03)<.00001); ++battery_valid;
        }
    } else if(type==72) { assert(length==64); memcpy(diagnostics,p,64); }
    else {
        assert(type==EW_LIDAR0 || type==EW_LIDAR1); assert(length==25); ++emitted_lidar[type-EW_LIDAR0];
    }
    return true;
}
static void advance(unsigned milliseconds) {
    uint64_t end=now+(uint64_t)milliseconds*1000;
    while(now<end) {
        now+=1000; uint64_t before=now; board_sensor_poll();
        /* Nominal bounded SPI execution plus one fault must leave main alive. */
        assert(now-before<15000);
    }
}
static void packet(unsigned index,bool valid) {
    uint8_t p[13]={0xaa,0x55,0,1,1,0,1,0,0,0,20,0x90,1};
    uint16_t checksum=(uint16_t)(ew_u16(p)^ew_u16(p+2)^ew_u16(p+4)^ew_u16(p+6)^p[10]^ew_u16(p+11));
    ew_put16(p+8,valid?checksum:(uint16_t)(checksum^1));
    assert(rxused[index]+sizeof(p)<sizeof(rxbytes[index]));
    memcpy(rxbytes[index]+rxused[index],p,sizeof(p)); rxused[index]+=sizeof(p);
}
int main(void) {
    reset_registers(); board_sensor_init(); assert(now==0);
    /* Late power: first identity access fails. The MCU remains responsive and
     * discovers the later sensor without requiring a physical reset. */
    advance(500); assert(imu_valid==0 && battery_valid==0 && starts[0]==1 && starts[1]==1);
    assert((hadc1.State&HAL_ADC_STATE_ERROR_INTERNAL)!=0 && calibrations==1);
    advance(1500); assert(imu_valid>0 && imu_error>0 && mag_valid==0);
    assert(starts[0]>=2 && starts[1]>=2 && battery_valid>0 && calibrations==1);
    /* Garbage UART bytes must not suppress a lidar start retry. */
    unsigned before_start=starts[0]; rxbytes[0][rxused[0]++]=0x17; advance(1100);
    assert(starts[0]>before_start);
    mag_present=true; advance(2100); assert(mag_valid>0 && master_resets>=2 && mag_resets>0);
    assert(diagnostics[13]==1 && diagnostics[14]==1 && diagnostics[15]==1);
    unsigned previous_imu=imu_valid,previous_mag=mag_valid;
    spi_failure=true; advance(100); assert(imu_valid==previous_imu && mag_valid==previous_mag);
    spi_failure=false; advance(2200); assert(imu_valid>previous_imu && mag_valid>previous_mag);
    /* A silent but electrically responding IMU is retried; no stale sample is
     * republished with a new time merely because SPI still acknowledges. */
    imu_quiet=true; previous_imu=imu_valid; unsigned old_resets=resets;
    advance(1800); assert(imu_valid==previous_imu && resets>old_resets);
    imu_quiet=false; advance(1000); assert(imu_valid>previous_imu);
    mag_overflow=true; previous_mag=mag_valid; advance(600); assert(mag_valid==previous_mag);
    mag_overflow=false; advance(2200); assert(mag_valid>previous_mag);
    adc_failure=true; unsigned previous_battery=battery_valid; advance(1100); assert(battery_valid==previous_battery);
    adc_failure=false; advance(600); assert(battery_valid>previous_battery);
    packet(0,true); packet(1,true); advance(5);
    assert(emitted_lidar[0]==1 && emitted_lidar[1]==1);
    unsigned healthy_start0=starts[0],healthy_start1=starts[1];
    for(unsigned i=0;i<12;++i) { packet(0,true); packet(1,true); advance(250); }
    assert(starts[0]==healthy_start0 && starts[1]==healthy_start1);
    for(unsigned i=0;i<13;++i) { packet(0,false); packet(1,true); advance(250); }
    assert(starts[0]>healthy_start0 && starts[1]==healthy_start1);
    assert(emitted_lidar[0]==13 && emitted_lidar[1]==26);
    /* Drop a partial packet after overrun, then recover on a clean packet. */
    rxbytes[0][rxused[0]++]=0xaa; advance(1); ++board_rx_overruns[PORT_LIDAR0];
    packet(0,true); advance(5); assert(emitted_lidar[0]==14);
    /* Bad configuration readback must never become a valid scaled sample. */
    spi_failure=true; advance(100); spi_failure=false; gyro_wrong=true; previous_imu=imu_valid;
    advance(2200); assert(imu_valid==previous_imu);
    gyro_wrong=false; advance(2200); assert(imu_valid>previous_imu);
    puts("real sensors: delayed power, IMU/mag retries, status, ADC faults, both lidar checksums/recovery: PASS");
}
