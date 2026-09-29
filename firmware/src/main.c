#include "board.h"
#include "servo_bus.h"
#include "app_motor.h"
#include "tim.h"
#include "build_identity.h"
#include <math.h>
#include <string.h>

static ec_controller_t controller;
static ew_frame_t commands[4];
static volatile uint32_t command_write,command_read,command_drops,emergency;
static uint8_t transmit[16][EW_MAX_FRAME];
static uint16_t transmit_length[16];
static unsigned transmit_write,transmit_read;
static uint32_t transmit_drops;
static volatile uint32_t control_generation;
static volatile uint64_t main_alive_us,control_alive_us;
static uint16_t raw_encoder[4],previous_encoder[4];
static int32_t encoder_delta[4];
static double targets[4],measurements[4],outputs[4];
static ex_pid_t regulators[4];
static ex_twist_t actual_velocity;
static double common_scale;
static bool encoder_measurement_valid;
static uint32_t max_control_us;
static uint8_t results[8][20];
static volatile uint32_t result_write,result_read,result_drops;
static uint32_t sensor_generation_seen;
static uint32_t arm_write_fault_seen;
static uint64_t rgb_generation;

/* Logical FL FR RL RR -> vendor M1 M3 M2 M4. */
static const unsigned motor_id[4]={0,2,1,3};
static const int encoder_sign[4]={1,-1,1,-1};
static const ex_mecanum_config_t geometry={{.04,.04,.04,.04},.167,{18.75,18.75,18.75,18.75}};
/* Conversion of vendor incremental PID gains to continuous units, m/s.
 * These are initial vendor-equivalent gains, not a new physical calibration. */
static const ex_pid_config_t pid_config={800,6000,5,0,1000,1000,.03};

bool board_emit(uint8_t type,const uint8_t *payload,size_t length) {
    unsigned next=(transmit_write+1)%16;
    if(next==transmit_read) { ++transmit_drops; return false; }
    size_t n=ew_encode(type,payload,length,transmit[transmit_write]);
    if(!n) { return false; }
    transmit_length[transmit_write]=(uint16_t)n; transmit_write=next; return true;
}
static void snapshot(ec_controller_t *copy) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    *copy=controller; __set_PRIMASK(saved);
}
bool servo_bus_commit(const ec_controller_t *copy,const uint8_t *frame,uint16_t length,bool cancel) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    uint64_t now=board_time_us();
    /* Check validity through the estimated end of a 115200 8N1 frame. A
     * higher-priority stop during TX is handled by the next measured hold. */
    uint64_t end=now+servo_write_budget_us(length);
    bool allowed=ec_arm_commit_allowed(&controller,copy,end,cancel);
    if(!cancel && (emergency || arm_write_fault_seen!=servo_write_fault_generation ||
                  now<main_alive_us || now-main_alive_us>30000u ||
                  now<control_alive_us || now-control_alive_us>30000u)) { allowed=false; }
    bool sent=allowed && board_send(PORT_ARM,frame,length);
    __set_PRIMASK(saved); return sent;
}
static void result(const ew_frame_t *f,ex_result_t r,uint64_t now) {
    uint32_t next=(result_write+1u)%8u;
    if(next==result_read) { ++result_drops; }
    else {
        uint8_t *out=results[result_write]; ew_put64(out,now);
        ew_put64(out+8,f->length>=24?ew_u64(f->payload+16):0);
        out[16]=f->type; out[17]=(uint8_t)r;
        out[18]=(uint8_t)controller.base.mode; out[19]=(uint8_t)controller.base.fault;
        __DMB(); result_write=next;
    }
}
void board_control_interrupt(void) {
    uint64_t now=board_time_us();
    if(emergency || now-main_alive_us>30000) {
        ex_base_estop(&controller.base); command_read=command_write; emergency=0;
    }
    if(arm_write_fault_seen!=servo_write_fault_generation) {
        ec_cancel_arm(&controller,servo_last_write_error); arm_write_fault_seen=servo_write_fault_generation;
    }
    if(sensor_generation_seen!=servo_measure_generation) {
        /* Main is preempted here; the sample set is committed using the brief
         * critical section in the servo dispatcher. */
        for(unsigned i=0;i<6;++i) {
            servo_measurement_t *m=&servo_measurements[i];
            ec_measure(&controller,i,m->raw,m->valid,m->time);
        }
        sensor_generation_seen=servo_measure_generation;
    }
    if(servo_sent_generation==controller.arm_generation) { controller.arm_pending=false; }
    if(controller.arm_cancel && servo_cancel_completed_generation==controller.arm_cancel_generation) {
        controller.arm_cancel=false;
    }
    for(unsigned budget=0;budget<3 && command_read!=command_write;++budget) {
        ew_frame_t *f=&commands[command_read];
        result(f,ec_dispatch(&controller,f,now),now);
        __DMB(); command_read=(command_read+1u)%4u;
    }
    ex_twist_t requested={0}; (void)ec_tick(&controller,now,&requested);
    raw_encoder[0]=(uint16_t)TIM3->CNT; raw_encoder[1]=(uint16_t)TIM5->CNT;
    raw_encoder[2]=(uint16_t)TIM2->CNT; raw_encoder[3]=(uint16_t)TIM4->CNT;
    double dt=control_alive_us?(double)(now-control_alive_us)/1000000.0:.01;
    bool valid=dt>0 && dt<=.03;
    for(unsigned i=0;i<4;++i) {
        int32_t delta=0;
        if(ex_encoder_delta16(previous_encoder[i],raw_encoder[i],1000,&delta)!=EX_OK) { valid=false; }
        previous_encoder[i]=raw_encoder[i]; encoder_delta[i]=delta;
        if(valid) { measurements[i]=(double)(delta*encoder_sign[i])*6.283185307179586/2464.0/dt; }
    }
    if(!valid || ex_mecanum_inverse(&geometry,requested,targets,&common_scale)!=EX_OK) {
        ex_base_estop(&controller.base);
    }
    bool stop=controller.base.mode!=EX_BASE_ACTIVE ||
        (requested.x==0 && requested.y==0 && requested.yaw==0);
    if(!stop) {
        for(unsigned i=0;i<4;++i) {
            if(ex_pid_step(&regulators[i],&pid_config,targets[i]*.04,measurements[i]*.04,dt,&outputs[i])!=EX_OK) {
                stop=true; ex_base_estop(&controller.base);
            }
        }
    }
    if(stop) {
        Motor_Stop(MOTOR_BRAKE);
        for(unsigned i=0;i<4;++i) { ex_pid_reset(&regulators[i]); targets[i]=0; outputs[i]=0; }
    } else {
        for(unsigned i=0;i<4;++i) { Motor_Set_Pwm((uint8_t)motor_id[i],(int16_t)outputs[i]); }
    }
    if(valid) { (void)ex_mecanum_forward(&geometry,measurements,&actual_velocity); }
    encoder_measurement_valid=valid;
    control_alive_us=now; ++control_generation;
    uint32_t elapsed=(uint32_t)(board_time_us()-now);
    if(elapsed>max_control_us) { max_control_us=elapsed; }
}
static void status(void) {
    uint8_t p[216]={0}; uint32_t saved=__get_PRIMASK(); __disable_irq();
    ew_put64(p,control_alive_us); ew_put64(p+8,controller.base.boot_id);
    ew_put64(p+16,controller.base.session_id); ew_put64(p+24,controller.sequence);
    ew_put64(p+32,controller.base.expires_us); ew_put64(p+40,controller.base.finite_end_us);
    p[48]=(uint8_t)controller.base.mode; p[49]=(uint8_t)controller.base.fault;
    p[50]=controller.arm_enabled; p[51]=controller.arm_cancel;
    for(unsigned i=0;i<4;++i) {
        unsigned at=52+26*i; ew_put16(p+at,raw_encoder[i]); ew_put32(p+at+2,(uint32_t)encoder_delta[i]);
        ew_putf32(p+at+6,(float)targets[i]); ew_putf32(p+at+10,(float)measurements[i]);
        ew_putf32(p+at+14,(float)outputs[i]); ew_putf32(p+at+18,(float)regulators[i].integral);
        ew_put32(p+at+22,motor_id[i]+1);
    }
    ew_putf32(p+156,(float)actual_velocity.x); ew_putf32(p+160,(float)actual_velocity.y);
    ew_putf32(p+164,(float)actual_velocity.yaw); ew_put32(p+168,max_control_us);
    ew_put32(p+172,command_drops); ew_put32(p+176,result_drops);
    ew_put32(p+180,transmit_drops); ew_put32(p+184,controller.rejected);
    ew_put64(p+188,servo_sent_generation); ew_put64(p+196,controller.base.highest_session_id);
    ew_put32(p+204,board_rx_overruns[PORT_HOST]); ew_put32(p+208,board_uart_errors[PORT_HOST]);
    ew_put32(p+212,encoder_measurement_valid?1u:0u);
    __set_PRIMASK(saved); (void)board_emit(EW_STATUS,p,sizeof(p));
    static uint64_t next_arm_diagnostic;
    const uint64_t now=board_time_us();
    if(now>=next_arm_diagnostic) {
        uint8_t arm[SERVO_DIAGNOSTIC_BYTES];
        saved=__get_PRIMASK(); __disable_irq();
        servo_bus_diagnostics(&controller,arm);
        __set_PRIMASK(saved);
        (void)board_emit(EW_ARM_DIAGNOSTICS,arm,sizeof(arm));
        next_arm_diagnostic=now+100000;
    }
}
static void hello(const ew_frame_t *request) {
    if(request->length!=8) { return; }
    uint8_t p[88]={0}; memcpy(p,request->payload,8);
    ew_put64(p+8,board_time_us()); ew_put64(p+16,controller.base.boot_id);
    ew_put32(p+24,HAL_GetDEVID()); ew_put32(p+28,HAL_GetREVID());
    ew_put16(p+32,*(const volatile uint16_t*)FLASHSIZE_BASE);
    ew_put32(p+36,HAL_GetUIDw0()); ew_put32(p+40,HAL_GetUIDw1()); ew_put32(p+44,HAL_GetUIDw2());
    p[48]=EW_VERSION;
    memcpy(p+49,ex_source_digest,32); ew_put32(p+84,RCC->RSR);
    (void)board_emit(EW_IDENTITY,p,sizeof(p));
}
int main(void) {
    board_init();
    const uint64_t boot=board_boot_nonce();
    servo_bus_init();
    board_sensor_init();
    if(ec_init(&controller,boot,board_time_us())!=EX_OK) { Error_Handler(); }
    previous_encoder[0]=(uint16_t)TIM3->CNT; previous_encoder[1]=(uint16_t)TIM5->CNT;
    previous_encoder[2]=(uint16_t)TIM2->CNT; previous_encoder[3]=(uint16_t)TIM4->CNT;
    main_alive_us=board_time_us(); board_start_control();
    static ew_parser_t parser; static ew_frame_t incoming; static ec_controller_t copy;
    uint64_t next_status=0; uint32_t fed_generation=0,host_error=0,host_overrun=0;
    for(;;) {
        uint64_t now=board_time_us();
        uint32_t saved=__get_PRIMASK(); __disable_irq(); main_alive_us=now; __set_PRIMASK(saved);
        board_io_poll();
        if(host_error!=board_uart_errors[PORT_HOST] || host_overrun!=board_rx_overruns[PORT_HOST]) {
            emergency=1; host_error=board_uart_errors[PORT_HOST]; host_overrun=board_rx_overruns[PORT_HOST];
            memset(&parser,0,sizeof(parser));
        }
        uint8_t b;
        for(unsigned budget=0;budget<2048 && board_receive(PORT_HOST,&b);++budget) {
            if(ew_receive(&parser,b,&incoming)) {
                if(incoming.type==EW_HELLO) { hello(&incoming); }
                else if(incoming.type==EW_ESTOP && incoming.length==0) { emergency=1; }
                else {
                    uint32_t next=(command_write+1u)%4u;
                    if(next==command_read) { ++command_drops; emergency=1; }
                    else { commands[command_write]=incoming; __DMB(); command_write=next; }
                }
            }
        }
        snapshot(&copy); servo_bus_poll(&copy); board_sensor_poll();
        if(copy.rgb_pending && rgb_generation!=copy.rgb_generation) {
            if(board_rgb(copy.rgb[0],copy.rgb[1],copy.rgb[2])) { rgb_generation=copy.rgb_generation; }
        }
        board_beep(copy.beep);
        if(now>=next_status) { next_status=now+20000; status(); }
        if(result_read!=result_write) {
            if(board_emit(EW_RESULT,results[result_read],20)) { __DMB(); result_read=(result_read+1u)%8u; }
        }
        if(transmit_read!=transmit_write && board_send(PORT_HOST,transmit[transmit_read],transmit_length[transmit_read])) {
            transmit_read=(transmit_read+1u)%16u;
        }
        /* Feed only after a completed main pass and a NEW bounded control tick.
         * UART/DDS liveness alone cannot keep motors or the watchdog alive. */
        saved=__get_PRIMASK(); __disable_irq();
        uint32_t generation=control_generation; uint64_t control_time=control_alive_us;
        __set_PRIMASK(saved);
        if(generation!=fed_generation && board_time_us()-control_time<20000 && max_control_us<3000) {
            board_watchdog_feed(); fed_generation=generation;
        }
    }
}
