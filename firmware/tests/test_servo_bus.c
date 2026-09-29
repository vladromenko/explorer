#include "servo_bus.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

UART_HandleTypeDef board_uart[PORT_COUNT];
volatile uint32_t board_rx_overruns[PORT_COUNT],board_uart_errors[PORT_COUNT];
static uint64_t now,busy_until,write_settled_at;
static uint8_t receive_bytes[8];
static unsigned read_index,read_length,missing_joint,queries[6],sync_count,partial_count;
static unsigned transient_joint,transient_remaining,device_fault_joint;
static uint8_t last_sync[38];
static uint16_t last_sync_length;
uint64_t board_time_us(void) { return now; }
bool board_receive(unsigned port,uint8_t *value) {
    assert(port==PORT_ARM);
    if(now>=busy_until && read_index<read_length) { *value=receive_bytes[read_index++]; return true; }
    return false;
}
bool board_send(unsigned port,const uint8_t *bytes,uint16_t length) {
    assert(port==PORT_ARM && board_uart[port].gState==HAL_UART_STATE_READY);
    board_uart[port].gState=1; busy_until=now+87*length;
    assert(bytes[0]==255 && bytes[1]==255);
    if(bytes[2]==254) {
        write_settled_at=busy_until+5000;
        assert(length>=13 && length<=38 && length==bytes[3]+4);
        uint8_t sum=0; for(unsigned i=2;i<length-1u;++i) { sum=(uint8_t)(sum+bytes[i]); }
        assert((uint8_t)~sum==bytes[length-1]);
        ++sync_count; if(length<38) { ++partial_count; }
        memcpy(last_sync,bytes,length); last_sync_length=length;
        for(unsigned at=7;at<length-1u;at+=5) {
            assert(bytes[at+1]==(2000>>8) && bytes[at+2]==(2000&255));
            assert(bytes[at+3]==0 && bytes[at+4]==0);
            if(length<38) { assert(bytes[at]!=missing_joint); }
        }
    } else {
        /* The UART becoming idle is not the same as a servo having processed
         * a broadcast. Emulate the observed post-write missing response. */
        assert(now>=write_settled_at);
        assert(length==8 && bytes[2]>=1 && bytes[2]<=6 && bytes[4]==2);
        unsigned id=bytes[2]; ++queries[id-1]; read_index=read_length=0;
        bool transient=id==transient_joint && transient_remaining>0;
        if(transient) { --transient_remaining; }
        if(id!=missing_joint && !transient) {
            uint8_t response[8]={255,245,(uint8_t)id,4,0,2000>>8,2000&255,0};
            if(id==device_fault_joint) { response[4]=1; }
            uint8_t sum=0; for(unsigned i=2;i<7;++i) { sum=(uint8_t)(sum+response[i]); }
            response[7]=(uint8_t)~sum; memcpy(receive_bytes,response,8); read_length=8;
        }
    }
    return true;
}
bool board_emit(uint8_t type,const uint8_t *payload,size_t length) {
    assert(type==EW_SERVO && length==30 && payload[8]>=1 && payload[8]<=6); return true;
}
static void advance(ec_controller_t *c,unsigned milliseconds) {
    for(unsigned i=0;i<milliseconds;++i) {
        now+=1000;
        if(now>=busy_until) { board_uart[PORT_ARM].gState=HAL_UART_STATE_READY; }
        servo_bus_poll(c);
    }
}
int main(void) {
    ec_controller_t c={0}; advance(&c,60);
    assert(sync_count==0); /* Startup only reads; no HOME, torque or target write. */
    for(unsigned i=0;i<6;++i) { assert(queries[i]>0 && servo_measurements[i].valid); }
    unsigned before_transient=queries[5];
    uint64_t before_stamp=servo_measurements[5].time;
    transient_joint=6; transient_remaining=1;
    for(unsigned step=0;step<40 && transient_remaining>0;++step) { advance(&c,1); }
    /* Retrying must not turn an old acquisition into a new measurement. */
    assert(transient_remaining==0 && queries[5]==before_transient+1);
    assert(servo_measurements[5].time==before_stamp);
    assert(servo_measurements[5].valid);
    advance(&c,60);
    assert(transient_remaining==0 && queries[5]>=before_transient+2);
    assert(servo_measurements[5].valid && servo_measurements[5].time>before_stamp);
    device_fault_joint=6; advance(&c,60);
    assert(!servo_measurements[5].valid && servo_measurements[5].error==EX_DEVICE_ERROR);
    device_fault_joint=0; advance(&c,60);
    assert(servo_measurements[5].valid);
    c.arm_cancel=true; c.arm_cancel_generation=1; advance(&c,10);
    assert(servo_cancel_completed_generation==1 && last_sync_length==38);
    c.arm_cancel=false; missing_joint=2; advance(&c,120);
    assert(!servo_measurements[1].valid);
    unsigned before_queries=queries[1];
    c.arm_cancel=true; c.arm_cancel_generation=2; advance(&c,100);
    assert(partial_count>0 && servo_cancel_completed_generation==1);
    assert(queries[1]>before_queries); /* Partial holds do not starve recovery reads. */
    missing_joint=0; advance(&c,100);
    assert(servo_cancel_completed_generation==2 && last_sync_length==38);
    c.arm_cancel=false; c.arm_enabled=true; c.arm_pending=true;
    c.arm_expires_us=now+3000000;
    for(unsigned i=0;i<6;++i) {
        c.arm_raw[i]=2000;
        c.calibration[i]=(ex_joint_calibration_t){96,4000,900,3100,.001,-2.,-1.1,1.1,true};
    }
    unsigned before_stream=sync_count;
    for(unsigned cycle=0;cycle<100;++cycle) {
        ++c.arm_generation; advance(&c,20);
        for(unsigned i=0;i<6;++i) {
            assert(servo_measurements[i].valid);
            assert(now-servo_measurements[i].time<250000);
        }
    }
    assert(sync_count-before_stream>=90);
    puts("real servo dispatcher: read-only startup, checksums, partial hold, continued polling, fresh cancel generation: PASS");
}
