#include "explorer_controller.h"
#include <math.h>
#include <string.h>

#define LEASE_US UINT64_C(250000)
#define FEEDBACK_US UINT64_C(250000)

ex_result_t ec_init(ec_controller_t *c,uint64_t boot,uint64_t now) {
    if(!c) { return EX_ARGUMENT; }
    memset(c,0,sizeof(*c));
    const ex_base_config_t cfg={LEASE_US,30000,.7,.7,4.2};
    return ex_base_init(&c->base,&cfg,boot,now);
}
static void cancel_arm(ec_controller_t *c, ex_result_t reason) {
    if(c->arm_enabled || c->arm_pending) {
        c->arm_cancel=true; ++c->arm_cancel_generation; c->arm_stop_reason=reason;
    }
    c->arm_enabled=false; c->arm_pending=false;
}
void ec_cancel_arm(ec_controller_t *c, ex_result_t reason) { if(c) { cancel_arm(c,reason); } }
static bool measured(const ec_controller_t *c,uint64_t now) {
    for(unsigned i=0;i<6;++i) {
        if(!c->measured_valid[i] || now<c->measured_us[i] ||
           now-c->measured_us[i]>FEEDBACK_US) { return false; }
    }
    return true;
}
static unsigned distance_from_work(uint16_t raw,const ex_joint_calibration_t *cal) {
    uint16_t lo=0,hi=0;
    if(ex_joint_work_bounds(cal,&lo,&hi)!=EX_OK) { return UINT16_MAX; }
    if(raw<lo) { return (unsigned)(lo-raw); }
    if(raw>hi) { return (unsigned)(raw-hi); }
    return 0;
}
static ex_result_t recovery_target(const ec_controller_t *c,const double q[6],uint16_t out[6]) {
    uint16_t proposed[6];
    for(unsigned i=0;i<6;++i) {
        const ex_joint_calibration_t *cal=&c->calibration[i];
        uint16_t lo=0,hi=0;
        if(ex_joint_work_bounds(cal,&lo,&hi)!=EX_OK) { return EX_UNCALIBRATED; }
        if(!isfinite(q[i]) || !cal->calibrated || !isfinite(cal->radians_per_tick) || cal->radians_per_tick==0) { return EX_VALUE; }
        double value=(q[i]-cal->radians_at_raw_zero)/cal->radians_per_tick;
        if(!isfinite(value) || value<96 || value>4000) { return EX_RANGE; }
        proposed[i]=(uint16_t)(value+.5);
        unsigned before=distance_from_work(c->arm_raw[i],cal),after=distance_from_work(proposed[i],cal);
        if(before==0) {
            /* Already-in-range joints stay at their starting target during a
             * recovery. A separate normal trajectory may move them afterwards. */
            if(proposed[i]!=c->arm_raw[i]) { return EX_SOFT_LIMIT; }
        } else {
            if(!c->recovery_min[i] || proposed[i]<c->recovery_min[i] || proposed[i]>c->recovery_max[i] || after>before) { return EX_SOFT_LIMIT; }
            int delta=(int)proposed[i]-(int)c->measured_raw[i];
            /* Bound lead during this exceptional path; no stop/dwell is added. */
            if(delta>30 || delta< -30) { return EX_RANGE; }
            if((c->arm_raw[i]<lo && proposed[i]>lo) ||
               (c->arm_raw[i]>hi && proposed[i]<hi)) { return EX_SOFT_LIMIT; }
        }
    }
    memcpy(out,proposed,sizeof(proposed)); return EX_OK;
}
static ex_result_t dispatch(ec_controller_t *c,const ew_frame_t *f,uint64_t now) {
    if(f->type==EW_ESTOP && f->length==0) {
        ex_base_estop(&c->base); cancel_arm(c,EX_LATCHED); return EX_OK;
    }
    if(f->length<40) { return EX_FRAME; }
    const uint8_t *p=f->payload;
    const uint64_t boot=ew_u64(p),session=ew_u64(p+8),seq=ew_u64(p+16);
    const uint64_t issued=ew_u64(p+24),expiry=ew_u64(p+32);
    if(boot!=c->base.boot_id || !session) { return EX_SESSION; }
    if(!seq) { return EX_REPLAY; }
    if(issued>now || expiry<=now || expiry<=issued || expiry-issued>LEASE_US) { return EX_EXPIRED; }
    if(f->type==EW_OPEN || f->type==EW_CLEAR) {
        if(f->length!=40 || session<=c->base.highest_session_id) { return EX_SESSION; }
        ex_result_t r;
        if(f->type==EW_CLEAR) {
            r=ex_base_clear_fault(&c->base,now);
            if(r==EX_OK) { c->base.highest_session_id=session; cancel_arm(c,EX_OK); }
        } else {
            if(c->arm_enabled || c->arm_pending || c->arm_cancel) { return EX_NOT_READY; }
            if(c->base.mode==EX_BASE_ACTIVE && !c->base.command_present) {
                r=ex_base_cancel(&c->base,boot,c->base.session_id,now);
                if(r!=EX_OK) { return r; }
            }
            r=ex_base_begin(&c->base,boot,session,now);
            if(r==EX_OK) { c->sequence=seq; }
        }
        return r;
    }
    if(session!=c->base.session_id) { return EX_SESSION; }
    if(seq<=c->sequence) { return EX_REPLAY; }
    if(c->base.mode==EX_BASE_FAULT) { return EX_LATCHED; }
    if(c->base.mode!=EX_BASE_ACTIVE) { return EX_DISARMED; }
    ex_result_t r=EX_FRAME;
    if(f->type==EW_BASE && f->length==60) {
        const ex_base_command_t cmd={boot,session,seq,issued,expiry,
            {ew_f32(p+40),ew_f32(p+44),ew_f32(p+48)},ew_u64(p+52)};
        r=ex_base_submit(&c->base,&cmd,now);
    } else if(f->type==EW_HOLD && f->length==40) {
        r=ex_base_hold(&c->base,now);
    } else if(f->type==EW_CANCEL && f->length==40) {
        r=ex_base_cancel(&c->base,boot,session,now); cancel_arm(c,EX_OK);
    } else if(f->type==EW_ARM_CANCEL && f->length==40) {
        cancel_arm(c,EX_OK); r=EX_OK;
    } else if((f->type==EW_ARM_ENABLE || f->type==EW_RECOVERY_ENABLE) && f->length==40) {
        if(c->arm_enabled || c->arm_pending || c->arm_cancel || !measured(c,now)) { return EX_NOT_READY; }
        bool recovery=f->type==EW_RECOVERY_ENABLE;
        for(unsigned i=0;i<6;++i) {
            ex_joint_sample_t sample;
            r=ex_joint_observe(&c->calibration[i],c->measured_raw[i],c->measured_us[i],now,FEEDBACK_US,&sample);
            if(!sample.position_valid) { return r; }
            if(sample.outside_soft_limit || distance_from_work(sample.raw,&c->calibration[i])!=0) {
                if(!recovery || !c->recovery_min[i] || sample.raw<c->recovery_min[i] || sample.raw>c->recovery_max[i]) { return EX_SOFT_LIMIT; }
            } else if(r!=EX_OK) { return r; }
        }
        memcpy(c->arm_raw,c->measured_raw,sizeof(c->arm_raw));
        c->arm_expires_us=expiry; c->arm_enabled=true;
        c->arm_recovery=recovery; c->arm_stop_reason=EX_OK;
        /* Align software target only. No boot/enable motion or torque write. */
        r=EX_OK;
    } else if(f->type==EW_CALIBRATION && f->length==232) {
        if(c->arm_enabled || c->arm_pending || c->arm_cancel) { return EX_NOT_READY; }
        ex_joint_calibration_t cal[6]; uint16_t recovery_lo[6],recovery_hi[6];
        for(unsigned i=0;i<6;++i) {
            const uint8_t *v=p+40+32*i;
            cal[i]=(ex_joint_calibration_t){ew_u16(v),ew_u16(v+2),ew_u16(v+4),ew_u16(v+6),
                ew_f32(v+8),ew_f32(v+12),ew_f32(v+16),ew_f32(v+20),true};
            /* Mechanical/raw envelope from the vendor remains immutable. */
            const uint16_t lo=i==4?380:900,hi=i==4?3700:3100;
            if(cal[i].raw_measure_min!=96 || cal[i].raw_measure_max!=4000 ||
               cal[i].raw_command_min<lo || cal[i].raw_command_max>hi ||
               ew_u32(v+28)!=0) { return EX_RANGE; }
            recovery_lo[i]=ew_u16(v+24); recovery_hi[i]=ew_u16(v+26);
            uint16_t work_lo=0,work_hi=0;
            r=ex_joint_work_bounds(&cal[i],&work_lo,&work_hi);
            if(r!=EX_OK) { return r; }
            /* Optional, separately accepted one-sided corridor. Never infer a
             * wider mechanical range merely because a reading is negative. */
            if(recovery_lo[i] || recovery_hi[i]) {
                if(recovery_lo[i]<96 || recovery_hi[i]>4000 || recovery_lo[i]>=recovery_hi[i] ||
                   recovery_hi[i]-recovery_lo[i]>1100 ||
                   !((recovery_hi[i]==work_lo && recovery_lo[i]<work_lo) ||
                     (recovery_lo[i]==work_hi && recovery_hi[i]>work_hi))) { return EX_RANGE; }
            }
            uint16_t test;
            const double middle=(cal[i].soft_min_rad+cal[i].soft_max_rad)*.5;
            r=ex_joint_to_raw(&cal[i],middle,&test);
            if(r!=EX_OK) { return r; }
        }
        memcpy(c->calibration,cal,sizeof(cal));
        memcpy(c->recovery_min,recovery_lo,sizeof(recovery_lo)); memcpy(c->recovery_max,recovery_hi,sizeof(recovery_hi)); r=EX_OK;
    } else if(f->type==EW_RGB && f->length==43) {
        memcpy(c->rgb,p+40,3); c->rgb_pending=true; ++c->rgb_generation; r=EX_OK;
    } else if(f->type==EW_BEEP && f->length==41 && p[40]<=1) {
        c->beep=p[40]!=0; r=EX_OK;
    } else if((f->type==EW_ARM || f->type==EW_ARM_RECOVER) && f->length==66) {
        if(!c->arm_enabled || c->arm_cancel || !measured(c,now)) { return EX_NOT_READY; }
        if(now>=c->arm_expires_us) { cancel_arm(c,EX_EXPIRED); return EX_EXPIRED; }
        const uint16_t runtime=ew_u16(p+64);
        /* Runtime 0 is vendor-defined immediate target; rate acceptance is separate. */
        if(runtime!=0) { return EX_RANGE; }
        double q[6]; uint16_t raw[6];
        for(unsigned i=0;i<6;++i) { q[i]=ew_f32(p+40+4*i); }
        if(c->arm_recovery!=(f->type==EW_ARM_RECOVER)) { return EX_NOT_READY; }
        r=c->arm_recovery?recovery_target(c,q,raw):ex_arm_targets_to_raw(c->calibration,q,raw);
        if(r==EX_OK && !c->arm_recovery) {
            /* Bound the outstanding physical target if feedback/transport fails.
             * This is a following-error envelope, not a stepped trajectory: a
             * valid continuous stream never has to stop at its sample points. */
            for(unsigned i=0;i<6;++i) {
                double lead=fabs(((double)raw[i]-c->measured_raw[i])*c->calibration[i].radians_per_tick);
                if(lead>.15) { return EX_RANGE; }
            }
        }
        if(r==EX_OK) {
            memcpy(c->arm_raw,raw,sizeof(raw)); c->arm_runtime_ms=runtime;
            c->arm_pending=true; c->arm_expires_us=expiry; ++c->arm_generation;
        }
    }
    if(r==EX_OK) { c->sequence=seq; }
    return r;
}
ex_result_t ec_dispatch(ec_controller_t *c,const ew_frame_t *f,uint64_t now) {
    if(!c || !f) { return EX_ARGUMENT; }
    ex_result_t r=dispatch(c,f,now);
    if(r==EX_OK) { ++c->accepted; } else { ++c->rejected; }
    return r;
}
ex_result_t ec_tick(ec_controller_t *c,uint64_t now,ex_twist_t *v) {
    if(!c || !v) { return EX_ARGUMENT; }
    ex_result_t r=ex_base_tick(&c->base,now,v);
    if(c->base.mode==EX_BASE_FAULT) { cancel_arm(c,c->base.fault); }
    else if(c->arm_enabled && now>=c->arm_expires_us) { cancel_arm(c,EX_EXPIRED); }
    else if(c->arm_enabled && !measured(c,now)) { cancel_arm(c,EX_STALE); }
    return r;
}
void ec_measure(ec_controller_t *c,unsigned i,uint16_t raw,bool valid,uint64_t now) {
    if(c && i<6) {
        c->measured_valid[i]=valid;
        if(valid) { c->measured_raw[i]=raw; c->measured_us[i]=now; }
    }
}
bool ec_arm_commit_allowed(const ec_controller_t *live,const ec_controller_t *copy,
                           uint64_t now,bool cancel) {
    if(!live || !copy || live->base.boot_id!=copy->base.boot_id ||
       live->base.session_id!=copy->base.session_id) { return false; }
    if(cancel) {
        /* A fault may stop normal commands but must not prevent measured hold.
         * Partial holds remain useful when one drive has stopped responding. */
        return live->arm_cancel && copy->arm_cancel &&
            live->arm_cancel_generation==copy->arm_cancel_generation;
    }
    /* The next tick will latch a lost base lease. Do not start another arm
     * frame in the short interval before that interrupt. A finite command
     * ending before its lease expires is a normal hold, not this fault. */
    if(live->base.command_present && now>=live->base.expires_us &&
       (!live->base.finite_end_us || live->base.finite_end_us>live->base.expires_us)) { return false; }
    return live->base.mode==EX_BASE_ACTIVE && live->arm_enabled && live->arm_pending &&
        !live->arm_cancel && copy->arm_enabled && copy->arm_pending && !copy->arm_cancel &&
        live->arm_generation==copy->arm_generation && now<live->arm_expires_us &&
        measured(live,now);
}
