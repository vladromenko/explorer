#include "board.h"
#include "spi.h"
#include "adc.h"
#include "stm32h7xx_ll_adc.h"
#include "app_icm20948.h"
#include <string.h>

/* Register values/scales from Yahboom Read_IMU, transport failure is propagated.
 * Unlike that example, never loop forever on WHO_AM_I, never erase gravity by
 * assuming the boot orientation, and never return stale data as a new sample.
 */
static bool imu_ready,mag_ready,battery_ready,battery_calibrating;
static uint64_t imu_due,battery_due;
static uint8_t imu_id,gyro_readback,accel_readback,imu_stage,spi_status;
static uint32_t lidar_rx_bytes[2],lidar_valid[2],lidar_start_attempts[2];
static uint64_t diagnostics_due,lidar_start_due,lidar_good_us[2];
static uint64_t init_due,imu_good_us,mag_init_due,mag_good_us,battery_calibration_due;
static uint8_t init_phase,config_index,mag_phase,imu_failures,mag_runtime;
static uint8_t mag_sample[9],mag_work[9];
static bool mag_new;
typedef struct {
    uint8_t reg,count,value,step,data[9]; bool read,active,done,ok;
    uint64_t due;
} mag_transfer_t;
static mag_transfer_t mag_transfer;
static const uint8_t imu_config[][3]={
    {ub_0,B0_PWR_MGMT_2,0},{ub_0,B0_USER_CTRL,0x10},
    {ub_2,B2_ODR_ALIGN_EN,1},{ub_2,B2_GYRO_CONFIG_1,6},
    {ub_2,B2_ACCEL_CONFIG,6},{ub_2,B2_GYRO_SMPLRT_DIV,0},
    {ub_2,B2_ACCEL_SMPLRT_DIV_1,0},{ub_2,B2_ACCEL_SMPLRT_DIV_2,0},
    {ub_0,B0_INT_ENABLE_1,1}
};
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
/* SLV0 repeats transfers. Never read ST1 through ST2 in one repeating
 * transfer: reading measurement data clears DRDY, so the next repetition can
 * overwrite a ready shadow with a not-ready status before the host sees it.
 * Use the vendor ST1 -> XYZ -> ST2 sequence; ST2 finishes the acquisition.
 * TDK DS-000189 v1.5 sections 8.11, 11.6, 13.2 and 13.4. */
static void mag_begin(uint8_t reg,uint8_t count,uint8_t value,bool read) {
    mag_transfer=(mag_transfer_t){.reg=reg,.count=count,.value=value,.read=read,.active=true};
}
static void mag_transfer_poll(uint64_t now) {
    mag_transfer_t *t=&mag_transfer;
    if(!t->active || now<t->due) { return; }
    bool ok=true; uint8_t status=0;
    /* One SPI register per main pass, including during recovery. This bounds
     * a missing peripheral to one 2 ms HAL timeout rather than blocking the
     * main loop for a chain of retries or long stabilization delays. */
    switch(t->step) {
    case 0: ok=write_reg(ub_3,B3_I2C_SLV0_CTRL,0); break;
    case 1: ok=read_reg(ub_0,B0_I2C_MST_STATUS,&status,1); break;
    case 2: ok=write_reg(ub_3,B3_I2C_SLV0_ADDR,(uint8_t)(MAG_SLAVE_ADDR|(t->read?0x80:0))); break;
    case 3: ok=write_reg(ub_3,B3_I2C_SLV0_REG,t->reg); break;
    case 4:
        if(!t->read) { ok=write_reg(ub_3,B3_I2C_SLV0_DO,t->value); }
        break;
    case 5:
        ok=write_reg(ub_3,B3_I2C_SLV0_CTRL,(uint8_t)(0x80|t->count));
        t->due=board_time_us()+1000; break;
    case 6: ok=read_reg(ub_0,B0_I2C_MST_STATUS,&status,1) && !(status&0x21u); break;
    case 7:
        if(t->read) { ok=read_reg(ub_0,B0_EXT_SLV_SENS_DATA_00,t->data,t->count); }
        t->done=true; t->active=false; t->ok=ok; break;
    default: ok=false; break;
    }
    if(ok) { ++t->step; }
    else { t->ok=false; t->done=true; t->active=false; }
}
static void imu_retry(uint64_t now) {
    imu_ready=false; mag_ready=false; init_phase=0; config_index=0; mag_phase=0;
    init_due=now+1000000; imu_failures=0; mag_transfer.active=false; mag_new=false;
}
static void imu_initialize(uint64_t now) {
    if(imu_ready || now<init_due) { return; }
    bool ok=true;
    switch(init_phase) {
    case 0:
        imu_id=0; gyro_readback=accel_readback=0; imu_stage=1;
        ok=read_reg(ub_0,B0_WHO_AM_I,&imu_id,1) && imu_id==ICM20948_ID;
        if(ok) { init_phase=1; }
        break;
    case 1:
        ok=write_reg(ub_0,B0_PWR_MGMT_1,0x80);
        init_phase=2; init_due=now+100000; break;
    case 2:
        ok=write_reg(ub_0,B0_PWR_MGMT_1,1);
        init_phase=3; init_due=now+100000; imu_stage=2; break;
    case 3:
        ok=write_reg(imu_config[config_index][0],imu_config[config_index][1],imu_config[config_index][2]);
        if(++config_index==sizeof(imu_config)/sizeof(imu_config[0])) {
            init_phase=4; init_due=now+100000;
        }
        break;
    case 4:
        ok=read_reg(ub_2,B2_GYRO_CONFIG_1,&gyro_readback,1) && gyro_readback==6;
        init_phase=5; break;
    case 5:
        ok=read_reg(ub_2,B2_ACCEL_CONFIG,&accel_readback,1) && accel_readback==6;
        if(ok) {
            imu_ready=true; imu_stage=4; imu_good_us=now;
            mag_phase=0; mag_init_due=now;
        }
        break;
    default: ok=false; break;
    }
    if(!ok) { imu_stage=3; imu_retry(now); }
}
static void mag_retry(uint64_t now) {
    mag_ready=false; mag_new=false; mag_phase=0; mag_runtime=0;
    mag_init_due=now+1000000; mag_transfer.active=false;
}
static void mag_initialize(uint64_t now) {
    if(!imu_ready || mag_ready || now<mag_init_due) { return; }
    bool ok=true,advance=true;
    switch(mag_phase) {
    case 0: /* Preserve SPI interface disable while resetting the I2C master. */
        ok=write_reg(ub_0,B0_USER_CTRL,0x12); mag_init_due=now+10000; break;
    case 1: ok=write_reg(ub_0,B0_USER_CTRL,0x30); mag_init_due=now+100000; break;
    case 2: ok=write_reg(ub_3,B3_I2C_MST_CTRL,7); break;
    case 3: mag_begin(MAG_WIA2,1,0,true); break;
    case 4:
        advance=mag_transfer.done;
        if(advance) { ok=mag_transfer.ok && mag_transfer.data[0]==AK09916_ID; }
        break;
    case 5: mag_begin(MAG_CNTL3,1,1,false); break;
    case 6:
        advance=mag_transfer.done;
        if(advance) { ok=mag_transfer.ok; mag_init_due=now+100000; }
        break;
    case 7: /* Stop repeated reset writes, then allow the device to settle. */
        ok=write_reg(ub_3,B3_I2C_SLV0_CTRL,0); mag_init_due=now+100000; break;
    case 8: mag_begin(MAG_CNTL2,1,continuous_measurement_100hz,false); break;
    case 9:
        advance=mag_transfer.done;
        if(advance) { ok=mag_transfer.ok; mag_init_due=now+10000; }
        break;
    case 10: mag_begin(MAG_CNTL2,1,0,true); break;
    case 11:
        advance=mag_transfer.done;
        if(advance) {
            ok=mag_transfer.ok && mag_transfer.data[0]==continuous_measurement_100hz;
            if(ok) { mag_ready=true; mag_good_us=now; mag_runtime=0; }
        }
        break;
    default: ok=false; break;
    }
    if(!ok) { mag_retry(now); }
    else if(advance) { ++mag_phase; }
}
static void mag_poll(uint64_t now) {
    if(!mag_ready) { return; }
    if(now-mag_good_us>500000) { mag_retry(now); return; }
    if(mag_runtime && !mag_transfer.done) { return; }
    if(mag_runtime && !mag_transfer.ok) { mag_retry(now); return; }
    switch(mag_runtime) {
    case 0: mag_begin(MAG_ST1,1,0,true); mag_runtime=1; break;
    case 1:
        if(mag_transfer.data[0]&1u) {
            mag_work[0]=mag_transfer.data[0]; mag_begin(MAG_HXL,6,0,true); mag_runtime=2;
        } else { mag_runtime=0; }
        break;
    case 2:
        memcpy(mag_work+1,mag_transfer.data,6); mag_work[7]=0; /* Reserved wire byte. */
        mag_begin(MAG_ST2,1,0,true); mag_runtime=3; break;
    case 3:
        mag_work[8]=mag_transfer.data[0]; mag_runtime=0;
        if(!(mag_work[8]&8u)) {
            memcpy(mag_sample,mag_work,sizeof(mag_sample)); mag_good_us=now; mag_new=true;
        }
        break;
    default: mag_retry(now); break;
    }
}
void board_sensor_init(void) {
    /* No sensor can keep the control loop from starting. Each init/recovery
     * step is bounded; the vendor power/reset/wake waits use clock deadlines. */
    imu_ready=mag_ready=battery_ready=false;
    init_phase=config_index=mag_phase=imu_failures=0; imu_stage=0;
    imu_due=battery_due=diagnostics_due=0;
    init_due=board_time_us()+200000; lidar_start_due=init_due;
    battery_calibrating=false; battery_calibration_due=0; mag_transfer=(mag_transfer_t){0}; mag_new=false; mag_runtime=0;
}
static void battery_calibration_poll(uint64_t now) {
    if(battery_ready) { return; }
    /* Same LL operation as ST HAL_ADCEx_Calibration_Start, without its
     * 633,600,000-iteration busy wait. ADC is disabled by the initial HAL init;
     * no conversion is allowed until the hardware clears ADCAL itself. */
    if(!battery_calibrating) {
        if(LL_ADC_IsEnabled(hadc1.Instance) || LL_ADC_IsCalibrationOnGoing(hadc1.Instance)) { return; }
        hadc1.State=(hadc1.State & ~(HAL_ADC_STATE_REG_BUSY|HAL_ADC_STATE_INJ_BUSY))|HAL_ADC_STATE_BUSY_INTERNAL;
        LL_ADC_StartCalibration(hadc1.Instance,ADC_CALIB_OFFSET_LINEARITY,ADC_SINGLE_ENDED);
        battery_calibrating=true; battery_calibration_due=now+100000;
    } else if(!LL_ADC_IsCalibrationOnGoing(hadc1.Instance)) {
        hadc1.State=(hadc1.State & ~(HAL_ADC_STATE_BUSY_INTERNAL|HAL_ADC_STATE_ERROR_INTERNAL))|HAL_ADC_STATE_READY;
        battery_ready=true; battery_calibrating=false;
    } else if(now>=battery_calibration_due) {
        hadc1.State=(hadc1.State & ~HAL_ADC_STATE_BUSY_INTERNAL)|HAL_ADC_STATE_ERROR_INTERNAL;
        /* Keep polling the existing operation. Never restart a calibration
         * that may still be active or label a timed-out result as voltage. */
    }
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
                    ++lidar_valid[index]; lidar_good_us[index]=board_time_us();
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
    imu_initialize(now); mag_transfer_poll(now); mag_initialize(now); mag_poll(now);
    battery_calibration_poll(now);
    if(now>=lidar_start_due) {
        const uint8_t start[2]={0xa5,0x60};
        lidar_start_due=now+1000000;
        for(unsigned i=0;i<2;++i) {
            /* Garbage bytes do not prove a running scanner. Keep a bounded
             * once-per-second start retry until checksummed packets arrive,
             * and recover after two seconds without a valid packet. */
            if((lidar_valid[i]==0 || now-lidar_good_us[i]>2000000) &&
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
        uint8_t payload[44]={0},raw[14],ready=0;
        ew_put64(payload,now); payload[8]=imu_ready?1:2;
        bool ok=imu_ready && read_reg(ub_0,B0_INT_STATUS_1,&ready,1);
        if(ok && (ready&1u)) {
            ok=read_reg(ub_0,B0_ACCEL_XOUT_H,raw,14);
            if(ok) { payload[8]=0; memcpy(payload+10,raw,14); imu_good_us=now; }
            else { payload[8]=2; }
        } else if(!ok) { payload[8]=2; }
        if(imu_ready) {
            if(!ok) { ++imu_failures; } else { imu_failures=0; }
            if(imu_failures>=3 || now-imu_good_us>500000) { imu_retry(now); }
        }
        payload[9]=mag_ready?1:2;
        if(mag_ready && mag_new && now-mag_good_us<100000) {
            payload[9]=0; memcpy(payload+24,mag_sample,9);
            ew_put64(payload+36,mag_good_us); mag_new=false;
        } else { ew_put64(payload+36,now); }
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
