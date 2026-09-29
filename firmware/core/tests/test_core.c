#include "explorer_core.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

static unsigned long checks = 0;
static unsigned int scenarios = 0;
#define CHECK(x) do { ++checks; if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); exit(1); } } while (0)
#define NEAR(x,y,e) CHECK(fabs((x)-(y)) <= (e))
#define RUN(f) do { f(); ++scenarios; printf("PASS %02u %s\n", scenarios, #f); } while (0)

static const uint64_t NOW = UINT64_C(1000000);
static ex_base_t new_base(void) {
    ex_base_t b;
    ex_base_config_t c = {250000u, 20000u, 0.7, 0.7, 4.2}; /* TEST VALUES, not calibrated. */
    CHECK(ex_base_init(&b, &c, 17u, NOW) == EX_OK);
    return b;
}
static ex_base_command_t command(uint64_t now, uint64_t seq) {
    return (ex_base_command_t){17u, 1u, seq, now, now+200000u, {0.1,-0.1,0.2}, 0u};
}
static void advance(ex_base_t *b, uint64_t end) {
    ex_twist_t output;
    while (b->last_tick_us < end) {
        const uint64_t t = end - b->last_tick_us > 5000u ? b->last_tick_us+5000u : end;
        (void)ex_base_tick(b, t, &output);
    }
}
static void boot_disarmed_and_requires_new_session(void) {
    ex_base_t b = new_base(); ex_twist_t v={1,2,3};
    CHECK(ex_base_tick(&b, NOW, &v)==EX_DISARMED); CHECK(v.x==0 && v.y==0 && v.yaw==0);
    ex_base_command_t c=command(NOW,1);
    CHECK(ex_base_submit(&b,&c,NOW)==EX_DISARMED);
    CHECK(ex_base_begin(&b,18,1,NOW)==EX_SESSION);
    CHECK(ex_base_begin(&b,17,0,NOW)==EX_SESSION);
    CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    CHECK(ex_base_tick(&b,NOW,&v)==EX_OK); CHECK(v.x==0);
}
static void all_six_signed_directions(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    const ex_twist_t values[]={{.1,0,0},{-.1,0,0},{0,.1,0},{0,-.1,0},{0,0,.2},{0,0,-.2},{-.1,.1,-.2}};
    for (size_t i=0;i<sizeof(values)/sizeof(values[0]);++i) {
        ex_base_command_t c=command(NOW,(uint64_t)i+1u); c.velocity=values[i];
        CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
        ex_twist_t v; CHECK(ex_base_tick(&b,NOW,&v)==EX_OK);
        NEAR(v.x,values[i].x,0); NEAR(v.y,values[i].y,0); NEAR(v.yaw,values[i].yaw,0);
    }
}
static void local_expiry_zeroes_target_without_transport(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    advance(&b,NOW+195000u); CHECK(b.mode==EX_BASE_ACTIVE);
    ex_twist_t v; CHECK(ex_base_tick(&b,NOW+200000u,&v)==EX_EXPIRED);
    CHECK(v.x==0 && v.y==0 && v.yaw==0 && b.fault==EX_EXPIRED);
    c=command(NOW+200000u,2); CHECK(ex_base_submit(&b,&c,NOW+200000u)==EX_LATCHED);
}
static void late_callback_cannot_resurrect_expired_command(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    advance(&b,NOW+195000u);
    c=command(NOW+200000u,2);
    CHECK(ex_base_submit(&b,&c,NOW+200000u)==EX_EXPIRED);
    CHECK(b.mode==EX_BASE_FAULT && b.target.x==0);
}
static void finite_command_stops_before_lease_without_new_packet(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); c.finite_duration_us=25000;
    CHECK(ex_base_submit(&b,&c,NOW)==EX_OK); advance(&b,NOW+25000u);
    CHECK(b.target.x==0 && !b.command_present && b.mode==EX_BASE_ACTIVE);
    c=command(NOW+25000u,2); CHECK(ex_base_submit(&b,&c,NOW+25000u)==EX_OK);
}
static void hold_is_not_estop_and_does_not_renew_lease(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    const uint64_t expiry=b.expires_us;
    CHECK(ex_base_hold(&b,NOW)==EX_OK); CHECK(ex_base_hold(&b,NOW)==EX_OK);
    CHECK(b.mode==EX_BASE_ACTIVE && b.fault==EX_OK && b.expires_us==expiry);
    c=command(NOW,2); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
}
static void cancel_and_reconnect_reject_old_generations(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    CHECK(ex_base_cancel(&b,17,2,NOW)==EX_SESSION); CHECK(b.mode==EX_BASE_ACTIVE);
    CHECK(ex_base_cancel(&b,17,1,NOW)==EX_OK); CHECK(b.target.x==0);
    CHECK(ex_base_submit(&b,&c,NOW)==EX_DISARMED);
    CHECK(ex_base_begin(&b,17,1,NOW)==EX_SESSION);
    CHECK(ex_base_begin(&b,17,2,NOW)==EX_OK);
    CHECK(ex_base_submit(&b,&c,NOW)==EX_SESSION);
    c.session_id=2; CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    CHECK(ex_base_cancel(&b,17,1,NOW)==EX_SESSION);
}
static void duplicates_and_sequence_wrap_do_not_refresh(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,UINT64_MAX); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    CHECK(ex_base_submit(&b,&c,NOW)==EX_REPLAY);
    c.sequence=0; CHECK(ex_base_submit(&b,&c,NOW)==EX_REPLAY);
    c.sequence=1; CHECK(ex_base_submit(&b,&c,NOW)==EX_REPLAY);
    CHECK(b.expires_us==NOW+200000u);
}
static void malformed_commands_do_not_change_target_or_deadline(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    c.sequence=2; c.velocity.x=NAN; CHECK(ex_base_submit(&b,&c,NOW)==EX_VALUE);
    c.velocity.x=INFINITY; CHECK(ex_base_submit(&b,&c,NOW)==EX_VALUE);
    c.velocity.x=-100; CHECK(ex_base_submit(&b,&c,NOW)==EX_RANGE);
    c=command(NOW,2); c.issued_us=NOW+1; CHECK(ex_base_submit(&b,&c,NOW)==EX_EXPIRED);
    c=command(NOW,2); c.expires_us=NOW; CHECK(ex_base_submit(&b,&c,NOW)==EX_EXPIRED);
    c=command(NOW,2); c.expires_us=NOW+250001u; CHECK(ex_base_submit(&b,&c,NOW)==EX_EXPIRED);
    c=command(NOW,2); c.finite_duration_us=200001u; CHECK(ex_base_submit(&b,&c,NOW)==EX_RANGE);
    CHECK(b.last_sequence==1 && b.expires_us==NOW+200000u); NEAR(b.target.x,.1,0);
}
static void backward_clock_and_control_overrun_latch(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_command_t c=command(NOW,1); CHECK(ex_base_submit(&b,&c,NOW)==EX_OK);
    ex_twist_t v;
    CHECK(ex_base_tick(&b,NOW-1,&v)==EX_TIME); CHECK(v.x==0 && b.fault==EX_TIME);
    CHECK(ex_base_clear_fault(&b,NOW-1)==EX_TIME);
    CHECK(ex_base_clear_fault(&b,NOW)==EX_OK);
    CHECK(ex_base_begin(&b,17,1,NOW)==EX_SESSION);
    CHECK(ex_base_begin(&b,17,2,NOW)==EX_OK);
    CHECK(ex_base_tick(&b,NOW+20001u,&v)==EX_TICK_OVERRUN);
    CHECK(v.x==0 && b.mode==EX_BASE_FAULT);
}
static void estop_requires_clear_and_new_session(void) {
    ex_base_t b=new_base(); CHECK(ex_base_begin(&b,17,1,NOW)==EX_OK);
    ex_base_estop(&b); CHECK(b.target.x==0 && b.mode==EX_BASE_FAULT);
    CHECK(ex_base_begin(&b,17,2,NOW)==EX_LATCHED);
    CHECK(ex_base_clear_fault(&b,NOW)==EX_OK); CHECK(b.mode==EX_BASE_DISARMED);
    CHECK(ex_base_begin(&b,17,1,NOW)==EX_SESSION);
    CHECK(ex_base_begin(&b,17,2,NOW)==EX_OK);
}
static void invalid_configuration_and_clock_overflow(void) {
    ex_base_t b=new_base();
    ex_base_config_t c=b.config; c.max_tick_gap_us=c.max_command_us;
    CHECK(ex_base_init(&b,&c,17,NOW)==EX_RANGE); CHECK(b.mode==EX_BASE_FAULT);
    b=new_base(); CHECK(ex_base_begin(&b,17,1,UINT64_MAX-1u)==EX_TIME);
    CHECK(b.mode==EX_BASE_DISARMED);
}
static ex_mecanum_config_t geometry(void) {
    return (ex_mecanum_config_t){{.04,.041,.039,.04},.25,{12,13,11,12}}; /* Test fixture. */
}
static unsigned int rng=1234567u;
static double random_signed(void) {
    rng=1664525u*rng+1013904223u;
    return 2.0*(double)(rng & 65535u)/65535.0-1.0;
}
static void mecanum_inverse_forward_and_symmetric_saturation(void) {
    ex_mecanum_config_t g=geometry();
    for (unsigned int trial=0;trial<10000u;++trial) {
        const ex_twist_t v={random_signed(),random_signed(),3*random_signed()};
        const ex_twist_t neg={-v.x,-v.y,-v.yaw};
        double w[4],n[4],s,sn; ex_twist_t back;
        CHECK(ex_mecanum_inverse(&g,v,w,&s)==EX_OK);
        CHECK(ex_mecanum_inverse(&g,neg,n,&sn)==EX_OK); NEAR(s,sn,1e-12);
        for (size_t i=0;i<4;++i) { NEAR(w[i],-n[i],1e-10); CHECK(fabs(w[i])<=g.max_wheel_rad_s[i]+1e-10); }
        CHECK(ex_mecanum_forward(&g,w,&back)==EX_OK);
        NEAR(back.x,v.x*s,1e-10); NEAR(back.y,v.y*s,1e-10); NEAR(back.yaw,v.yaw*s,1e-10);
    }
}
static void mecanum_specific_axes_and_invalid_input(void) {
    ex_mecanum_config_t g=geometry(); double w[4],s;
    CHECK(ex_mecanum_inverse(&g,(ex_twist_t){0,.1,0},w,&s)==EX_OK);
    CHECK(w[0]<0 && w[1]>0 && w[2]>0 && w[3]<0);
    CHECK(ex_mecanum_inverse(&g,(ex_twist_t){0,0,.1},w,&s)==EX_OK);
    CHECK(w[0]<0 && w[1]>0 && w[2]<0 && w[3]>0);
    CHECK(ex_mecanum_inverse(&g,(ex_twist_t){NAN,0,0},w,&s)==EX_VALUE);
    g.radius_m[0]=0; CHECK(ex_mecanum_inverse(&g,(ex_twist_t){.1,0,0},w,&s)==EX_VALUE);
}
static void encoder_rollover_and_ambiguity(void) {
    int32_t delta=777;
    CHECK(ex_encoder_delta16(65530,5,100,&delta)==EX_OK); CHECK(delta==11);
    CHECK(ex_encoder_delta16(5,65530,100,&delta)==EX_OK); CHECK(delta==-11);
    CHECK(ex_encoder_delta16(0,32768,32767,&delta)==EX_RANGE);
    CHECK(ex_encoder_delta16(0,101,100,&delta)==EX_RANGE);
    CHECK(ex_encoder_delta16(0,100,100,&delta)==EX_OK); CHECK(delta==100);
}
static void pid_antiwindup_and_reversal(void) {
    ex_pid_t state={0}; const ex_pid_config_t c={1,5,0,0,1,.5,.1}; double out;
    for (unsigned int i=0;i<10000u;++i) { CHECK(ex_pid_step(&state,&c,100,0,.01,&out)==EX_OK); NEAR(out,1,0); }
    NEAR(state.integral,0,0);
    CHECK(ex_pid_step(&state,&c,-100,0,.01,&out)==EX_OK); NEAR(out,-1,0);
    CHECK(ex_pid_step(&state,&c,NAN,0,.01,&out)==EX_VALUE); NEAR(out,0,0);
    CHECK(ex_pid_step(&state,&c,0,0,.2,&out)==EX_VALUE);
    ex_pid_reset(&state); CHECK(!state.initialized && state.integral==0);
}
static ex_joint_calibration_t calibrated(void) {
    return (ex_joint_calibration_t){96,4000,900,3100,.001,-2.0,-1.1,1.1,true}; /* Not robot calibration. */
}
static void seal(uint8_t f[8]) {
    unsigned int sum=0; for (size_t i=2;i<7;++i) { sum+=f[i]; } f[7]=(uint8_t)(~sum & 255u);
}
static void servo_packet_checks_and_echo_rejection(void) {
    uint8_t f[8]={255,245,4,4,0,0x0e,0x32,0}; seal(f); uint16_t raw=777; uint8_t error;
    CHECK(ex_servo_decode_position(f,8,4,245,&raw,&error)==EX_OK); CHECK(raw==3634 && error==0);
    CHECK(ex_servo_decode_position(f,7,4,245,&raw,&error)==EX_FRAME);
    CHECK(ex_servo_decode_position(f,8,3,245,&raw,&error)==EX_FRAME);
    CHECK(ex_servo_decode_position(f,8,4,255,&raw,&error)==EX_FRAME);
    f[3]=5; seal(f); CHECK(ex_servo_decode_position(f,8,4,245,&raw,&error)==EX_FRAME);
    f[3]=4; f[4]=2; seal(f); raw=777;
    CHECK(ex_servo_decode_position(f,8,4,245,&raw,&error)==EX_DEVICE_ERROR); CHECK(raw==777 && error==2);
    CHECK(ex_servo_make_read_request(4,f)==EX_OK);
    CHECK(ex_servo_decode_position(f,8,4,255,&raw,&error)==EX_DEVICE_ERROR); CHECK(raw==777);
}
static void servo_single_bit_corruption_is_rejected(void) {
    uint8_t original[8]={255,245,4,4,0,7,208,0}; seal(original);
    for (unsigned int bit=0;bit<64u;++bit) {
        uint8_t f[8]; memcpy(f,original,8); f[bit/8u]^=(uint8_t)(1u<<(bit%8u));
        uint16_t raw=123; uint8_t error;
        CHECK(ex_servo_decode_position(f,8,4,245,&raw,&error)!=EX_OK); CHECK(raw==123);
    }
}
static void negative_joint_is_valid_when_inside_calibrated_limits(void) {
    ex_joint_calibration_t c=calibrated(); ex_joint_sample_t s;
    CHECK(ex_joint_observe(&c,1249,100,110,100,&s)==EX_OK);
    CHECK(s.raw_valid && s.position_valid && !s.outside_soft_limit); NEAR(s.position_rad,-.751,1e-12);
    uint16_t raw=0; CHECK(ex_joint_to_raw(&c,s.position_rad,&raw)==EX_OK); CHECK(raw==1249);
}
static void archived_negative_43_case_is_data_not_sentinel(void) {
    const double pi=3.14159265358979323846;
    ex_joint_calibration_t c={96,4000,900,3100,-pi/2200.0,3100.0*pi/2200.0,0,pi,true};
    ex_joint_sample_t s;
    CHECK(ex_joint_observe(&c,3634,100,110,100,&s)==EX_SOFT_LIMIT);
    CHECK(s.raw_valid && s.position_valid && s.outside_soft_limit);
    CHECK(s.position_rad*180.0/pi < -43 && s.position_rad*180.0/pi > -44);
    /* The former -1 sentinel can also arise from a geometrically valid read. */
    CHECK(ex_joint_observe(&c,3120,100,110,100,&s)==EX_SOFT_LIMIT);
    CHECK(s.position_valid && s.position_rad<0);
}
static void stale_unreadable_and_uncalibrated_are_distinct(void) {
    ex_joint_calibration_t c=calibrated(); ex_joint_sample_t s;
    CHECK(ex_joint_observe(&c,2000,100,200,100,&s)==EX_STALE); CHECK(s.raw_valid && !s.position_valid && s.acquired_us==100);
    CHECK(ex_joint_observe(&c,2000,201,200,100,&s)==EX_STALE);
    c.calibrated=false; CHECK(ex_joint_observe(&c,2000,100,110,100,&s)==EX_UNCALIBRATED);
    CHECK(s.raw_valid && !s.position_valid);
    CHECK(ex_joint_observe(&c,65535,100,110,100,&s)==EX_RANGE); CHECK(!s.raw_valid);
}
static void raw_roundtrip_for_every_commandable_tick(void) {
    ex_joint_calibration_t c=calibrated();
    for (uint32_t v=900;v<=3100;++v) {
        ex_joint_sample_t s; uint16_t raw=0;
        CHECK(ex_joint_observe(&c,(uint16_t)v,1,2,100,&s)==EX_OK);
        CHECK(ex_joint_to_raw(&c,s.position_rad,&raw)==EX_OK); CHECK(raw==v);
    }
}
static void group_command_is_atomic_and_never_substitutes_midpoint(void) {
    ex_joint_calibration_t c[6]; double q[6]={0,0,0,0,0,0}; uint16_t raw[6]={7,7,7,7,7,7};
    for (size_t i=0;i<6;++i) { c[i]=calibrated(); }
    q[3]=-4; CHECK(ex_arm_targets_to_raw(c,q,raw)==EX_SOFT_LIMIT);
    for (size_t i=0;i<6;++i) { CHECK(raw[i]==7); }
    q[3]=NAN; CHECK(ex_arm_targets_to_raw(c,q,raw)==EX_VALUE);
    q[3]=-.7; CHECK(ex_arm_targets_to_raw(c,q,raw)==EX_OK); CHECK(raw[3]==1300);
    uint8_t f[38]; CHECK(ex_servo_make_sync(raw,c,800,100,5000,f)==EX_OK);
    CHECK(f[0]==255 && f[1]==255 && f[2]==254 && f[3]==34 && f[4]==131);
    unsigned int sum=0; for (size_t i=2;i<37;++i) { sum+=f[i]; } CHECK(f[37]==(uint8_t)(~sum & 255u));
    uint8_t copy[38]; memcpy(copy,f,38); raw[3]=65535;
    CHECK(ex_servo_make_sync(raw,c,800,100,5000,f)==EX_RANGE); CHECK(memcmp(copy,f,38)==0);
    raw[3]=2000; CHECK(ex_servo_make_sync(raw,c,0,100,5000,f)==EX_RANGE); CHECK(memcmp(copy,f,38)==0);
}
static void profile_is_synchronized_bounded_and_monotone(void) {
    const double start[6]={0,-.7,.4,0,.2,0}; const double goal[6]={1.1,.2,-.4,-.75,.2,.6};
    ex_profile_limits_t lim[6];
    for (size_t i=0;i<6;++i) { lim[i]=(ex_profile_limits_t){-2,2,.8,1.7,3}; }
    ex_profile_t p; CHECK(ex_profile_plan(start,goal,lim,0,&p)==EX_OK);
    ex_profile_sample_t first,last,previous;
    CHECK(ex_profile_sample(&p,0,&first)==EX_OK); CHECK(ex_profile_sample(&p,p.duration_s,&last)==EX_OK);
    previous=first;
    for (size_t i=0;i<6;++i) {
        NEAR(first.position[i],start[i],1e-12); NEAR(last.position[i],goal[i],1e-12);
        NEAR(first.velocity[i],0,0); NEAR(first.acceleration[i],0,0);
        NEAR(last.velocity[i],0,0); NEAR(last.acceleration[i],0,0);
    }
    CHECK(last.finished && !first.finished);
    for (unsigned int k=0;k<=5000u;++k) {
        ex_profile_sample_t s; CHECK(ex_profile_sample(&p,p.duration_s*(double)k/5000.0,&s)==EX_OK);
        for (size_t i=0;i<6;++i) {
            CHECK(fabs(s.velocity[i])<=lim[i].velocity_max+1e-9);
            CHECK(fabs(s.acceleration[i])<=lim[i].acceleration_max+1e-9);
            CHECK(fabs(s.jerk[i])<=lim[i].jerk_max+1e-9);
            CHECK(s.position[i]>=lim[i].position_min && s.position[i]<=lim[i].position_max);
            if (goal[i]>=start[i]) { CHECK(s.position[i]>=previous.position[i]-1e-12); }
            else { CHECK(s.position[i]<=previous.position[i]+1e-12); }
        }
        previous=s;
    }
}
static void profile_honors_slower_request_and_rejects_bad_targets(void) {
    const double start[6]={0}; double goal[6]={1,1,1,1,1,1}; ex_profile_limits_t lim[6];
    for (size_t i=0;i<6;++i) { lim[i]=(ex_profile_limits_t){-2,2,2,4,20}; }
    ex_profile_t p; CHECK(ex_profile_plan(start,goal,lim,10,&p)==EX_OK); NEAR(p.duration_s,10,0);
    ex_profile_sample_t s; CHECK(ex_profile_sample(&p,11,&s)==EX_OK); CHECK(s.finished);
    CHECK(ex_profile_sample(&p,-.1,&s)==EX_TIME);
    CHECK(ex_profile_sample(&p,NAN,&s)==EX_TIME);
    goal[3]=-3; CHECK(ex_profile_plan(start,goal,lim,0,&p)==EX_SOFT_LIMIT);
    goal[3]=NAN; CHECK(ex_profile_plan(start,goal,lim,0,&p)==EX_VALUE);
    goal[3]=1; lim[2].velocity_max=0; CHECK(ex_profile_plan(start,goal,lim,0,&p)==EX_VALUE);
}
int main(void) {
    RUN(boot_disarmed_and_requires_new_session);
    RUN(all_six_signed_directions);
    RUN(local_expiry_zeroes_target_without_transport);
    RUN(late_callback_cannot_resurrect_expired_command);
    RUN(finite_command_stops_before_lease_without_new_packet);
    RUN(hold_is_not_estop_and_does_not_renew_lease);
    RUN(cancel_and_reconnect_reject_old_generations);
    RUN(duplicates_and_sequence_wrap_do_not_refresh);
    RUN(malformed_commands_do_not_change_target_or_deadline);
    RUN(backward_clock_and_control_overrun_latch);
    RUN(estop_requires_clear_and_new_session);
    RUN(invalid_configuration_and_clock_overflow);
    RUN(mecanum_inverse_forward_and_symmetric_saturation);
    RUN(mecanum_specific_axes_and_invalid_input);
    RUN(encoder_rollover_and_ambiguity);
    RUN(pid_antiwindup_and_reversal);
    RUN(servo_packet_checks_and_echo_rejection);
    RUN(servo_single_bit_corruption_is_rejected);
    RUN(negative_joint_is_valid_when_inside_calibrated_limits);
    RUN(archived_negative_43_case_is_data_not_sentinel);
    RUN(stale_unreadable_and_uncalibrated_are_distinct);
    RUN(raw_roundtrip_for_every_commandable_tick);
    RUN(group_command_is_atomic_and_never_substitutes_midpoint);
    RUN(profile_is_synchronized_bounded_and_monotone);
    RUN(profile_honors_slower_request_and_rejects_bad_targets);
    printf("SUMMARY: %u scenarios, %lu assertions, PASS. Host software only; no hardware tests.\n",scenarios,checks);
    return 0;
}
