#include "servo_bus.h"
#include <string.h>

/* Manufacturer Subscriber_uart_servo/APP/app_uart_servo.h specifies FF F5.
 * Do not accept a possible FF FF request echo as measured position. */
#define REPLY_HEADER2 0xf5
#define REPLY_US UINT64_C(20000)
#define READ_RETRIES 2u
#define WRITE_TIMEOUT_US UINT64_C(20000)
servo_measurement_t servo_measurements[6];
volatile uint32_t servo_measure_generation;
uint64_t servo_sent_generation;
volatile uint32_t servo_cancel_completed_generation;
volatile uint32_t servo_write_fault_generation;
ex_result_t servo_last_write_error;
static ex_result_t last_build_error;
static unsigned joint,read_retries;
static uint8_t reply[8],used;
static uint64_t deadline,next_request,next_write,next_hold,read_acquired_us;
static bool waiting;
static uint32_t last_uart_errors,last_overruns,seen_cancel_generation;

enum write_kind { WRITE_NONE,WRITE_TARGET,WRITE_PARTIAL_HOLD,WRITE_FULL_HOLD };
static enum write_kind writing;
static uint64_t writing_generation,write_started,write_deadline,write_quiet_until;
static uint32_t writing_cancel_generation,writing_completion;
static bool write_failed;

void servo_bus_init(void) {
    memset(servo_measurements,0,sizeof(servo_measurements));
    servo_measure_generation=0; servo_sent_generation=0; servo_cancel_completed_generation=0;
    servo_write_fault_generation=0;
    servo_last_write_error=EX_OK; last_build_error=EX_OK;
    joint=0; read_retries=0; used=0; memset(reply,0,sizeof(reply));
    deadline=0; next_request=0; next_write=0; next_hold=0; read_acquired_us=0;
    waiting=false; last_uart_errors=board_uart_errors[PORT_ARM];
    last_overruns=board_rx_overruns[PORT_ARM]; seen_cancel_generation=0;
    writing=WRITE_NONE; writing_generation=0; write_started=0; write_deadline=0; write_quiet_until=0;
    writing_cancel_generation=0; writing_completion=0; write_failed=false;
}

static void report(ex_result_t result,uint16_t raw,uint8_t device) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    servo_measurement_t *m=&servo_measurements[joint];
    m->error=result; m->device_error=device; m->valid=result==EX_OK;
    /* Replies carry no acquisition timestamp or transaction ID. Use the
     * earliest request in this bounded retry group, never the late arrival.
     * A reply from an arbitrarily old cycle cannot be identified by this
     * vendor protocol; idle draining and deadlines do not invent such an ID. */
    if(m->valid) { m->raw=raw; m->time=read_acquired_us; }
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

static void write_fault(ex_result_t error) {
    /* Queue acceptance or UART READY after an abort is not wire completion.
     * Preserve the last raw/time as history but revoke its control validity. */
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    for(unsigned i=0;i<6;++i) {
        servo_measurements[i].valid=false;
        servo_measurements[i].error=error;
        servo_measurements[i].device_error=0;
    }
    ++servo_measure_generation;
    ++servo_write_fault_generation;
    servo_last_write_error=error;
    __set_PRIMASK(saved);
}

static void retry_read_or_report(ex_result_t error) {
    if(read_retries<READ_RETRIES) {
        ++read_retries; waiting=false; used=0;
        next_request=board_time_us()+2000;
    } else {
        report(error,0,0);
    }
}

static bool finish_write(uint64_t now) {
    if(writing!=WRITE_NONE) {
        uint32_t saved=__get_PRIMASK(); __disable_irq();
        const uint32_t completed=board_tx_completed[PORT_ARM];
        const uint64_t completed_us=board_tx_completed_us[PORT_ARM];
        const bool ready=board_uart[PORT_ARM].gState==HAL_UART_STATE_READY;
        __set_PRIMASK(saved);
        if(completed!=writing_completion) {
            /* A late main poll must not turn a timely TC into a timeout.
             * Conversely, a late TC is still a fault, even if no poll ran
             * at the deadline. Queue acceptance is never used as completion. */
            if(!write_failed && (completed!=writing_completion+1u ||
               completed_us<write_started || completed_us>=write_deadline || completed_us>now)) {
                write_failed=true; write_fault(EX_STALE);
            }
            if(!write_failed) {
                saved=__get_PRIMASK(); __disable_irq();
                if(writing==WRITE_TARGET) { servo_sent_generation=writing_generation; }
                if(writing==WRITE_FULL_HOLD) { servo_cancel_completed_generation=writing_cancel_generation; }
                __set_PRIMASK(saved);
            }
            writing=WRITE_NONE; write_quiet_until=completed_us+SERVO_WRITE_QUIET_US;
        } else if(now>=write_deadline && !write_failed) {
            write_failed=true; write_fault(EX_STALE);
        } else if(ready) {
            if(!write_failed) { write_fault(EX_FRAME); }
            writing=WRITE_NONE; write_quiet_until=now+SERVO_WRITE_QUIET_US;
        }
    }
    return writing==WRITE_NONE && now>=write_quiet_until;
}

static bool start_write(const ec_controller_t *s,const uint8_t *frame,uint16_t length,
                        enum write_kind kind) {
    const uint64_t now=board_time_us();
    const bool cancel=kind!=WRITE_TARGET;
    if(!cancel && (now>=s->arm_expires_us ||
       s->arm_expires_us-now<servo_write_budget_us(length))) { return false; }
    uint32_t completion=board_tx_completed[PORT_ARM];
    /* Main rechecks the LIVE controller under a brief critical section.
     * A copied snapshot alone cannot authorize actuator transmission. */
    if(!servo_bus_commit(s,frame,length,cancel)) { return false; }
    writing=kind; writing_generation=s->arm_generation;
    writing_cancel_generation=s->arm_cancel_generation; writing_completion=completion;
    write_started=now; write_deadline=now+WRITE_TIMEOUT_US; write_failed=false;
    return true;
}

static bool hold_measured(const ec_controller_t *s) {
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
    }
    if(!count) { return false; }
    unsigned length=8+5*count; frame[3]=(uint8_t)(length-4);
    uint8_t sum=0; for(unsigned i=2;i<length-1;++i) { sum=(uint8_t)(sum+frame[i]); }
    frame[length-1]=(uint8_t)~sum;
    return start_write(s,frame,(uint16_t)length,count==6?WRITE_FULL_HOLD:WRITE_PARTIAL_HOLD);
}

static void resync_reply(void) {
    unsigned start=8;
    for(unsigned i=1;i<7 && start==8;++i) {
        if(reply[i]==0xff && reply[i+1]==REPLY_HEADER2) { start=i; }
    }
    if(start<8) {
        used=(uint8_t)(8-start); memmove(reply,reply+start,used);
    } else {
        used=reply[7]==0xff?1:0;
        if(used) { reply[0]=0xff; }
    }
}

void servo_bus_poll(const ec_controller_t *s) {
    if(!s) { return; }
    uint64_t now=board_time_us(); uint8_t b;
    if(!finish_write(now)) { return; }
    if(s->arm_cancel && s->arm_cancel_generation!=seen_cancel_generation) {
        /* Cancellation does not wait through a missing joint's retries. Drop
         * only this logical read; the UART finishes any already-started bytes. */
        seen_cancel_generation=s->arm_cancel_generation;
        waiting=false; used=0; read_retries=0; next_hold=now; next_request=now+1000;
    }
    if(last_uart_errors!=board_uart_errors[PORT_ARM] || last_overruns!=board_rx_overruns[PORT_ARM]) {
        last_uart_errors=board_uart_errors[PORT_ARM]; last_overruns=board_rx_overruns[PORT_ARM];
        if(waiting) { retry_read_or_report(EX_FRAME); }
    }
    /* Expired buffered replies must be discarded before any decode occurs. */
    if(waiting && now>=deadline) { retry_read_or_report(EX_STALE); }
    unsigned budget=0;
    while(budget<128 && board_receive(PORT_ARM,&b)) {
        ++budget;
        if(waiting) {
            if(used==0) { if(b==0xff) { reply[used++]=b; } }
            else if(used==1) {
                if(b==REPLY_HEADER2) { reply[used++]=b; }
                else { used=b==0xff?1:0; }
            } else {
                reply[used++]=b;
                if(used==8) {
                    if(board_time_us()>=deadline) { retry_read_or_report(EX_STALE); }
                    else {
                        uint16_t raw=0; uint8_t device=0;
                        ex_result_t r=ex_servo_decode_position(reply,8,(uint8_t)(joint+1),REPLY_HEADER2,&raw,&device);
                        if(r==EX_OK && (raw<96 || raw>4000)) { r=EX_RANGE; }
                        if(r==EX_OK || r==EX_DEVICE_ERROR || r==EX_RANGE) { report(r,raw,device); }
                        else { resync_reply(); }
                    }
                }
            }
        }
    }
    if(!waiting && board_uart[PORT_ARM].gState==HAL_UART_STATE_READY) {
        if(s->arm_cancel && s->arm_cancel_generation!=servo_cancel_completed_generation && now>=next_hold) {
            if(hold_measured(s)) { next_hold=now+20000; }
        } else if(!s->arm_cancel && s->arm_enabled && s->arm_pending && now<s->arm_expires_us &&
                  s->arm_generation!=servo_sent_generation && now>=next_write) {
            uint8_t frame[EX_SYNC_BYTES];
            ex_joint_calibration_t allowed[6]; memcpy(allowed,s->calibration,sizeof(allowed));
            if(s->arm_recovery) {
                for(unsigned i=0;i<6;++i) {
                    uint16_t lo=0,hi=0;
                    bool in_work=ex_joint_work_bounds(&allowed[i],&lo,&hi)==EX_OK && s->arm_raw[i]>=lo && s->arm_raw[i]<=hi;
                    if(s->recovery_min[i] && !in_work) {
                        allowed[i].raw_command_min=s->recovery_min[i];
                        allowed[i].raw_command_max=s->recovery_max[i];
                        double a=allowed[i].radians_at_raw_zero+allowed[i].radians_per_tick*s->recovery_min[i];
                        double z=allowed[i].radians_at_raw_zero+allowed[i].radians_per_tick*s->recovery_max[i];
                        allowed[i].soft_min_rad=a<z?a:z; allowed[i].soft_max_rad=a>z?a:z;
                    }
                }
            }
            ex_result_t r=ex_servo_make_sync(s->arm_raw,allowed,0,0,0,frame);
            last_build_error=r;
            if(r==EX_OK && start_write(s,frame,sizeof(frame),WRITE_TARGET)) { next_write=now+20000; }
        }
        /* Drain fully before new reads so buffered old bytes cannot become
         * a new transaction merely because one poll's receive budget ended. */
        if(writing==WRITE_NONE && budget<128 && board_uart[PORT_ARM].gState==HAL_UART_STATE_READY && now>=next_request) {
            uint8_t request[8];
            if(ex_servo_make_read_request((uint8_t)(joint+1),request)==EX_OK && board_send(PORT_ARM,request,8)) {
                if(read_retries==0) { read_acquired_us=board_time_us(); }
                waiting=true; used=0; memset(reply,0,sizeof(reply)); deadline=board_time_us()+REPLY_US;
            }
        }
    }
}

void servo_bus_diagnostics(const ec_controller_t *s,uint8_t out[SERVO_DIAGNOSTIC_BYTES]) {
    /* Main and ISR-owned values must form one snapshot. This is telemetry,
     * not an acknowledgement of target-register contents or reached pose. */
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    memset(out,0,SERVO_DIAGNOSTIC_BYTES);
    ew_put64(out,board_time_us()); ew_put64(out+8,s->arm_generation);
    ew_put64(out+16,servo_sent_generation); ew_put64(out+24,s->arm_expires_us);
    ew_put64(out+32,write_started); ew_put64(out+40,board_tx_completed_us[PORT_ARM]);
    ew_put64(out+48,write_deadline); ew_put32(out+56,board_uart_errors[PORT_ARM]);
    ew_put32(out+60,board_rx_overruns[PORT_ARM]); ew_put32(out+64,servo_write_fault_generation);
    ew_put32(out+68,board_tx_completed[PORT_ARM]); out[72]=(uint8_t)s->arm_stop_reason;
    out[73]=(uint8_t)servo_last_write_error; out[74]=(uint8_t)last_build_error;
    out[75]=(uint8_t)writing; out[76]=(uint8_t)waiting; out[77]=(uint8_t)(joint+1u);
    out[78]=(uint8_t)read_retries;
    for(unsigned i=0;i<6;++i) {
        if(servo_measurements[i].valid) { out[79]|=(uint8_t)(1u<<i); }
        ew_put16(out+88+2*i,s->arm_raw[i]);
    }
    ew_put64(out+80,deadline);
    __set_PRIMASK(saved);
}
