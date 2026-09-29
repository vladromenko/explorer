#include "board.h"
#include "spi.h"
#include "adc.h"
#include "app_icm20948.h"
#include <string.h>

/* Register values/scales from Yahboom Read_IMU, transport failure is propagated.
 * Unlike that example, never loop forever on WHO_AM_I, never erase gravity by
 * assuming the boot orientation, and never return stale data as a new sample.
 */
static bool imu_ready,mag_ready,battery_ready;
static uint64_t imu_due,battery_due;
static uint8_t imu_id,gyro_readback,accel_readback,imu_stage,spi_status;
static uint32_t lidar_rx_bytes[2],lidar_valid[2],lidar_start_attempts[2];
static uint64_t diagnostics_due,lidar_start_due;
static bool transaction(uint8_t *send,uint8_t *receive,uint16_t n) {
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_12,GPIO_PIN_RESET);
    HAL_StatusTypeDef r=HAL_SPI_TransmitReceive(&hspi2,send,receive,n,2);
    spi_status=(uint8_t)r;
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_12,GPIO_PIN_SET); return r==HAL_OK;
}
static bool bank(uint8_t value) {
    uint8_t tx[2]={REG_BANK_SEL,value},rx[2]; return transaction(tx,rx,2);
}
static bool write_reg(uint8_t b,uint8_t reg,uint8_t value) {
    uint8_t tx[2]={reg,value},rx[2]; return bank(b) && transaction(tx,rx,2);
}
static bool read_reg(uint8_t b,uint8_t reg,uint8_t *out,uint16_t n) {
    if(n>24) { return false; }
    uint8_t tx[25]={0},rx[25]={0}; tx[0]=reg|0x80;
    bool ok=bank(b) && transaction(tx,rx,(uint16_t)(n+1));
    if(ok) { memcpy(out,rx+1,n); } return ok;
}
static bool mag_read(uint8_t reg,uint8_t *out,uint8_t n) {
    bool ok=write_reg(ub_3,B3_I2C_SLV0_ADDR,0x80|MAG_SLAVE_ADDR) &&
        write_reg(ub_3,B3_I2C_SLV0_REG,reg) && write_reg(ub_3,B3_I2C_SLV0_CTRL,(uint8_t)(0x80|n));
    if(ok) { HAL_Delay(1); ok=read_reg(ub_0,B0_EXT_SLV_SENS_DATA_00,out,n); }
    return ok;
}
static bool mag_write(uint8_t reg,uint8_t value) {
    return write_reg(ub_3,B3_I2C_SLV0_ADDR,MAG_SLAVE_ADDR) &&
        write_reg(ub_3,B3_I2C_SLV0_REG,reg) && write_reg(ub_3,B3_I2C_SLV0_DO,value) &&
        write_reg(ub_3,B3_I2C_SLV0_CTRL,0x81);
}
void board_sensor_init(void) {
    /* Yahboom starts its lidar task after 200 ms; allow external sensors to
     * finish power-on before the first request as well as after an MCU reset. */
    HAL_Delay(200);
    uint8_t id=0;
    imu_ready=read_reg(ub_0,B0_WHO_AM_I,&id,1) && id==ICM20948_ID;
    imu_id=id; imu_stage=1;
    if(imu_ready) {
        imu_ready=write_reg(ub_0,B0_PWR_MGMT_1,0x80); HAL_Delay(100);
        imu_ready=imu_ready && write_reg(ub_0,B0_PWR_MGMT_1,1);
        /* Same wake stabilization interval as Yahboom ICM20948_wakeup(). */
        HAL_Delay(100); imu_stage=2;
        imu_ready=imu_ready && write_reg(ub_0,B0_PWR_MGMT_2,0) && write_reg(ub_0,B0_USER_CTRL,0x30) &&
            write_reg(ub_2,B2_ODR_ALIGN_EN,1) && write_reg(ub_2,B2_GYRO_CONFIG_1,6) &&
            write_reg(ub_2,B2_ACCEL_CONFIG,6) && write_reg(ub_2,B2_GYRO_SMPLRT_DIV,0) &&
            write_reg(ub_2,B2_ACCEL_SMPLRT_DIV_1,0) && write_reg(ub_2,B2_ACCEL_SMPLRT_DIV_2,0) &&
            write_reg(ub_0,B0_INT_ENABLE_1,1);
        /* Match vendor bypass/zero-divisor mode. The host publication rate is
         * 100 Hz; it must not be described as the internal sensor sample rate. */
        HAL_Delay(100);
        uint8_t gyro_cfg=0,accel_cfg=0;
        imu_ready=imu_ready && read_reg(ub_2,B2_GYRO_CONFIG_1,&gyro_cfg,1) && gyro_cfg==6 &&
            read_reg(ub_2,B2_ACCEL_CONFIG,&accel_cfg,1) && accel_cfg==6;
        gyro_readback=gyro_cfg; accel_readback=accel_cfg; imu_stage=imu_ready?4:3;
        bool master=imu_ready && write_reg(ub_3,B3_I2C_MST_CTRL,7);
        for(unsigned attempt=0;master && !mag_ready && attempt<5;++attempt) {
            mag_ready=mag_read(MAG_WIA2,&id,1) && id==AK09916_ID;
            if(!mag_ready) { HAL_Delay(2); }
        }
        if(mag_ready) {
            mag_ready=mag_write(MAG_CNTL2,continuous_measurement_100hz);
            /* Finish the asynchronous mode write before replacing SLV0 with a
             * read transaction. Motors are still disabled during this init. */
            HAL_Delay(10);
        }
    }
    battery_ready=HAL_ADCEx_Calibration_Start(&hadc1,ADC_CALIB_OFFSET_LINEARITY,ADC_SINGLE_ENDED)==HAL_OK;
    const uint8_t start[2]={0xa5,0x60};
    (void)board_send(PORT_LIDAR0,start,2); (void)board_send(PORT_LIDAR1,start,2);
    lidar_start_attempts[0]=1; lidar_start_attempts[1]=1;
    lidar_start_due=board_time_us()+1000000;
}
typedef struct {
    uint8_t data[775]; uint16_t used,want; uint64_t started;
    uint32_t errors,overflow_seen,uart_errors_seen;
} lidar_parser_t;
static lidar_parser_t lidars[2];
static void lidar_poll(unsigned index) {
    lidar_parser_t *p=&lidars[index]; uint8_t b;
    unsigned port=PORT_LIDAR0+index;
    if(board_rx_overruns[port]!=p->overflow_seen || board_uart_errors[port]!=p->uart_errors_seen) {
        p->used=0; ++p->errors;
        p->overflow_seen=board_rx_overruns[port]; p->uart_errors_seen=board_uart_errors[port];
    }
    for(unsigned budget=0;budget<512 && board_receive(port,&b);++budget) {
        ++lidar_rx_bytes[index];
        if(p->used && board_time_us()-p->started>100000) { p->used=0; ++p->errors; }
        if(p->used==0) {
            if(b==0xaa) { p->data[p->used++]=b; p->started=board_time_us(); }
        } else if(p->used==1 && b!=0x55) {
            p->used=b==0xaa?1:0; p->started=board_time_us();
        } else {
            p->data[p->used++]=b;
            if(p->used==4) { p->want=(uint16_t)(10u+3u*b); if(b==0) { p->used=0; ++p->errors; } }
            if(p->used>=10 && p->used==p->want) {
                uint16_t sum=(uint16_t)(ew_u16(p->data)^ew_u16(p->data+2)^ew_u16(p->data+4)^ew_u16(p->data+6));
                for(unsigned i=10;i<p->want;i+=3) { sum^=(uint16_t)(p->data[i]^ew_u16(p->data+i+1)); }
                if(sum==ew_u16(p->data+8)) {
                    ++lidar_valid[index];
                    uint8_t payload[791]; ew_put64(payload,p->started); ew_put32(payload+8,p->errors);
                    memcpy(payload+12,p->data,p->want);
                    (void)board_emit((uint8_t)(EW_LIDAR0+index),payload,p->want+12u);
                } else { ++p->errors; }
                p->used=0;
            }
        }
    }
}
void board_sensor_poll(void) {
    lidar_poll(0); lidar_poll(1);
    uint64_t now=board_time_us();
    if(now>=lidar_start_due) {
        const uint8_t start[2]={0xa5,0x60};
        lidar_start_due=now+1000000;
        for(unsigned i=0;i<2;++i) {
            if(lidar_rx_bytes[i]==0 && lidar_start_attempts[i]<3 &&
               board_send(PORT_LIDAR0+i,start,2)) { ++lidar_start_attempts[i]; }
        }
    }
    if(now>=diagnostics_due) {
        uint8_t p[64]={0}; ew_put64(p,now);
        p[8]=imu_stage; p[9]=imu_id; p[10]=gyro_readback; p[11]=accel_readback;
        p[12]=spi_status; p[13]=imu_ready; p[14]=mag_ready; p[15]=battery_ready;
        for(unsigned i=0;i<2;++i) {
            uint8_t *d=p+16+24*i;
            ew_put32(d,lidar_rx_bytes[i]); ew_put32(d+4,lidar_valid[i]);
            ew_put32(d+8,lidars[i].errors); ew_put32(d+12,board_rx_overruns[PORT_LIDAR0+i]);
            ew_put32(d+16,board_uart_errors[PORT_LIDAR0+i]); ew_put32(d+20,lidar_start_attempts[i]);
        }
        (void)board_emit(72,p,sizeof(p)); diagnostics_due=now+1000000;
    }
    if(now>=imu_due) {
        imu_due=now+10000;
        uint8_t payload[44]={0},raw[14],mag[9],ready=0;
        ew_put64(payload,now); payload[8]=imu_ready?1:2;
        bool ok=imu_ready && read_reg(ub_0,B0_INT_STATUS_1,&ready,1);
        if(ok && (ready&1u)) {
            ok=read_reg(ub_0,B0_ACCEL_XOUT_H,raw,14);
            if(ok) { payload[8]=0; memcpy(payload+10,raw,14); }
        } else if(!ok) { payload[8]=2; }
        payload[9]=1;
        if(mag_ready && mag_read(MAG_ST1,mag,9)) {
            if((mag[0]&1u) && !(mag[8]&8u)) { payload[9]=0; memcpy(payload+24,mag,9); }
        }
        ew_put64(payload+36,board_time_us());
        (void)board_emit(EW_IMU,payload,sizeof(payload));
    }
    if(now>=battery_due) {
        battery_due=now+500000;
        uint8_t payload[15]={0}; ew_put64(payload,now);
        bool ok=battery_ready && HAL_ADC_Start(&hadc1)==HAL_OK && HAL_ADC_PollForConversion(&hadc1,2)==HAL_OK;
        payload[8]=ok?0:2;
        if(ok) {
            uint16_t raw=(uint16_t)HAL_ADC_GetValue(&hadc1); ew_put16(payload+9,raw);
            ew_putf32(payload+11,(float)raw*(3.30f/4096.0f)*4.03f);
        }
        (void)HAL_ADC_Stop(&hadc1);
        (void)board_emit(EW_BATTERY,payload,sizeof(payload));
    }
}
