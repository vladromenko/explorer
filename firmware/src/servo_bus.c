#include "servo_bus.h"
#include <string.h>

/* Manufacturer Subscriber_uart_servo/APP/app_uart_servo.h says FF F5.
 * This choice is fixed, not autodetected from a possible FF FF request echo.
 * Raw replies are emitted for physical confirmation before motion acceptance.
 */
#define REPLY_HEADER2 0xf5
#define REPLY_US UINT64_C(20000)
#define READ_RETRIES 2u
servo_measurement_t servo_measurements[6];
volatile uint32_t servo_measure_generation;
uint64_t servo_sent_generation;
volatile uint32_t servo_cancel_completed_generation;
static unsigned joint;
static uint8_t reply[8],used;
static uint64_t deadline,next_request,next_write,next_hold;
static bool waiting;
static unsigned read_retries;
/* Broadcast writes have no reply. The next read must wait for BOTH wire
 * completion and a processing interval; UART READY alone is insufficient.
 * 5 ms is a conservative bench candidate, not a physically accepted timing. */
static bool write_settling;
static uint64_t write_quiet_until;
static uint32_t last_uart_errors,last_overruns;

static void report(ex_result_t result,uint16_t raw,uint8_t device) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    servo_measurement_t *m=&servo_measurements[joint];
    m->error=result; m->device_error=device; m->valid=result==EX_OK;
    if(m->valid) { m->raw=raw; m->time=board_time_us(); }
    ++servo_measure_generation;
    __set_PRIMASK(saved);
    uint8_t payload[30]={0};
    ew_put64(payload,board_time_us()); payload[8]=(uint8_t)(joint+1);
    payload[9]=(uint8_t)result; payload[10]=device; ew_put16(payload+11,raw);
    ew_put64(payload+13,m->time); memcpy(payload+21,reply,8); payload[29]=REPLY_HEADER2;
    (void)board_emit(EW_SERVO,payload,sizeof(payload));
    waiting=false; used=0; read_retries=0;
    joint=(joint+1u)%6u; next_request=board_time_us()+1000;
}
static void retry_read_or_report(ex_result_t error) {
    /* A transient bus error does not destroy a recent valid measurement.
     * Its acquisition time remains unchanged, so ec_tick still enforces the
     * independent 250 ms feedback deadline. Three failed requests end this
     * joint's turn; missing hardware cannot starve the other joints forever.
     * Device-reported faults never enter this retry path. */
    if(read_retries<READ_RETRIES) {
        ++read_retries; waiting=false; used=0;
        next_request=board_time_us()+2000;
    } else {
        report(error,0,0);
    }
}
static bool hold_measured(void) {
    uint8_t frame[EX_SYNC_BYTES]={0xff,0xff,0xfe,0x22,0x83,0x2a,4};
    const uint64_t now=board_time_us();
    unsigned count=0;
    for(unsigned i=0;i<6;++i) {
        const servo_measurement_t *m=&servo_measurements[i];
        if(m->valid && now>=m->time && now-m->time<250000 && m->raw>=96 && m->raw<=4000) {
            unsigned at=7+5*count; frame[at]=(uint8_t)(i+1);
            frame[at+1]=(uint8_t)(m->raw>>8); frame[at+2]=(uint8_t)m->raw;
            ++count;
        }
        /* Freeze at the measured raw position, including outside a soft limit.
         * This is not a newly planned position and does not widen move limits. */
    }
    if(!count) { return false; }
    unsigned length=8+5*count; frame[3]=(uint8_t)(length-4);
    uint8_t sum=0; for(unsigned i=2;i<length-1;++i) { sum=(uint8_t)(sum+frame[i]); }
    frame[length-1]=(uint8_t)~sum;
    bool sent=board_send(PORT_ARM,frame,(uint16_t)length);
    /* A missing joint cannot prevent holding the fresh joints. Completion
     * still requires all six; partial transmission never clears cancellation. */
    if(sent) { write_settling=true; write_quiet_until=0; }
    return sent && count==6;
}
void servo_bus_poll(const ec_controller_t *s) {
    uint64_t now=board_time_us(); uint8_t b;
    if(write_settling) {
        if(board_uart[PORT_ARM].gState!=HAL_UART_STATE_READY) { return; }
        if(!write_quiet_until) { write_quiet_until=now+5000; }
        if(now<write_quiet_until) { return; }
        write_settling=false;
    }
    if(last_uart_errors!=board_uart_errors[PORT_ARM] || last_overruns!=board_rx_overruns[PORT_ARM]) {
        last_uart_errors=board_uart_errors[PORT_ARM]; last_overruns=board_rx_overruns[PORT_ARM];
        if(waiting) { retry_read_or_report(EX_FRAME); }
    }
    for(unsigned budget=0;budget<128 && board_receive(PORT_ARM,&b);++budget) {
        if(waiting) {
            if(used==0) { if(b==0xff) { reply[used++]=b; } }
            else if(used==1) {
                if(b==REPLY_HEADER2) { reply[used++]=b; }
                else { used=b==0xff?1:0; }
            } else {
                reply[used++]=b;
                if(used==8) {
                    uint16_t raw=0; uint8_t device=0;
                    ex_result_t r=ex_servo_decode_position(reply,8,(uint8_t)(joint+1),REPLY_HEADER2,&raw,&device);
                    if(r==EX_OK || r==EX_DEVICE_ERROR) { report(r,raw,device); }
                    else { used=0; }
                }
            }
        }
    }
    if(waiting && now>=deadline) { retry_read_or_report(EX_STALE); }
    if(!waiting && board_uart[PORT_ARM].gState==HAL_UART_STATE_READY) {
        if(s->arm_cancel && s->arm_cancel_generation!=servo_cancel_completed_generation && now>=next_hold) {
            if(hold_measured()) { servo_cancel_completed_generation=s->arm_cancel_generation; }
            next_hold=now+20000;
        } else if(s->arm_enabled && s->arm_pending && now<s->arm_expires_us &&
                  s->arm_generation!=servo_sent_generation && now>=next_write) {
            uint8_t frame[EX_SYNC_BYTES];
            ex_joint_calibration_t allowed[6]; memcpy(allowed,s->calibration,sizeof(allowed));
            if(s->arm_recovery) {
                /* Dispatcher has atomically validated monotonic progress inside
                 * the explicitly configured corridor; normal limits stay intact. */
                for(unsigned i=0;i<6;++i) {
                    uint16_t lo=0,hi=0;
                    bool in_work=ex_joint_work_bounds(&allowed[i],&lo,&hi)==EX_OK && s->arm_raw[i]>=lo && s->arm_raw[i]<=hi;
                    if(s->recovery_min[i] && !in_work) {
                        allowed[i].raw_command_min=s->recovery_min[i];
                        allowed[i].raw_command_max=s->recovery_max[i];
                        double a=allowed[i].radians_at_raw_zero+allowed[i].radians_per_tick*s->recovery_min[i];
                        double b=allowed[i].radians_at_raw_zero+allowed[i].radians_per_tick*s->recovery_max[i];
                        allowed[i].soft_min_rad=a<b?a:b; allowed[i].soft_max_rad=a>b?a:b;
                    }
                }
            }
            ex_result_t r=ex_servo_make_sync(s->arm_raw,allowed,0,0,0,frame);
            if(r==EX_OK && board_send(PORT_ARM,frame,sizeof(frame))) {
                uint32_t saved=__get_PRIMASK(); __disable_irq();
                servo_sent_generation=s->arm_generation; __set_PRIMASK(saved);
                next_write=now+20000;
                write_settling=true; write_quiet_until=0;
            }
        }
        if(!write_settling && board_uart[PORT_ARM].gState==HAL_UART_STATE_READY && now>=next_request) {
            uint8_t request[8];
            if(ex_servo_make_read_request((uint8_t)(joint+1),request)==EX_OK && board_send(PORT_ARM,request,8)) {
                waiting=true; used=0; memset(reply,0,sizeof(reply)); deadline=now+REPLY_US;
            }
        }
    }
}
