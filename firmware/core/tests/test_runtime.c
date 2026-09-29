#include "explorer_controller.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

static ew_frame_t request(uint8_t type,uint64_t session,uint64_t sequence,uint64_t now) {
    ew_frame_t f={.type=type,.length=40};
    ew_put64(f.payload,123); ew_put64(f.payload+8,session); ew_put64(f.payload+16,sequence);
    ew_put64(f.payload+24,now); ew_put64(f.payload+32,now+200000); return f;
}
static void wire(void) {
    assert(ew_crc32((const uint8_t*)"123456789",9)==0xcbf43926);
    uint8_t p[EW_MAX_PAYLOAD],encoded[EW_MAX_FRAME]; ew_frame_t f;
    for(unsigned n=0;n<=EW_MAX_PAYLOAD;++n) {
        for(unsigned i=0;i<n;++i) { p[i]=(uint8_t)(i*37+n); }
        size_t count=ew_encode(42,p,n,encoded); assert(count>0 && count<=EW_MAX_FRAME);
        ew_parser_t parser={0}; bool got=false;
        for(size_t k=0;k<count;++k) { got=ew_receive(&parser,encoded[k],&f); assert(!got || k==count-1); }
        assert(got && f.type==42 && f.length==n && memcmp(p,f.payload,n)==0);
        encoded[count/2]^=0x10; got=false;
        for(size_t k=0;k<count;++k) { got=ew_receive(&parser,encoded[k],&f)||got; }
        assert(!got);
    }
    ew_parser_t parser={0};
    for(unsigned i=0;i<EW_MAX_FRAME*3;++i) { assert(!ew_receive(&parser,1,&f)); }
    assert(!ew_receive(&parser,0,&f)); assert(parser.oversized==1);
    size_t n=ew_encode(1,NULL,0,encoded); bool ok=false;
    for(size_t i=0;i<n;++i) { ok=ew_receive(&parser,encoded[i],&f); }
    assert(ok);
    uint32_t random=1;
    for(unsigned i=0;i<1000000;++i) {
        random=random*1664525u+1013904223u; (void)ew_receive(&parser,(uint8_t)(random>>24),&f);
    }
}
static void sessions(void) {
    ec_controller_t c; assert(ec_init(&c,123,0)==EX_OK);
    ew_frame_t f=request(EW_OPEN,1,1,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    ex_twist_t v;
    for(uint64_t t=10000;t<=1000000;t+=10000) { assert(ec_tick(&c,t,&v)==EX_OK); }
    f=request(EW_BASE,1,2,1000000); f.length=60;
    ew_putf32(f.payload+40,.1f); ew_putf32(f.payload+44,-.1f); ew_putf32(f.payload+48,.2f);
    ew_put64(f.payload+52,50000);
    assert(ec_dispatch(&c,&f,1010000)==EX_OK);
    assert(c.base.finite_end_us==1050000);
    assert(ec_dispatch(&c,&f,1010000)==EX_REPLAY);
    assert(ec_tick(&c,1010000,&v)==EX_OK && v.y<0);
    for(uint64_t t=1020000;t<=1300000;t+=10000) { assert(ec_tick(&c,t,&v)==EX_OK); }
    assert(v.x==0 && c.base.mode==EX_BASE_ACTIVE);
    f=request(EW_BASE,1,3,1300000); f.length=60; ew_putf32(f.payload+40,.1f);
    assert(ec_dispatch(&c,&f,1300000)==EX_OK);
    for(uint64_t t=1310000;t<1500000;t+=10000) { assert(ec_tick(&c,t,&v)==EX_OK); }
    assert(ec_tick(&c,1500000,&v)==EX_EXPIRED && v.x==0);
    f=request(EW_OPEN,2,1,1500000); assert(ec_dispatch(&c,&f,1500000)==EX_LATCHED);
    f=request(EW_CLEAR,2,1,1500000); assert(ec_dispatch(&c,&f,1500000)==EX_OK);
    f=request(EW_OPEN,2,1,1500000); assert(ec_dispatch(&c,&f,1500000)==EX_SESSION);
    f=request(EW_OPEN,3,1,1500000); assert(ec_dispatch(&c,&f,1500000)==EX_OK);
    f=request(EW_BASE,1,900,1500000); f.length=60; assert(ec_dispatch(&c,&f,1500000)==EX_SESSION);
    f=request(EW_BASE,3,2,1500000); f.length=60; ew_putf32(f.payload+40,NAN);
    assert(ec_dispatch(&c,&f,1500000)==EX_VALUE && c.sequence==1);
    f=request(EW_CANCEL,3,2,1500000); assert(ec_dispatch(&c,&f,1500000)==EX_OK);
    f=request(EW_BASE,3,3,1500000); f.length=60; assert(ec_dispatch(&c,&f,1500000)==EX_DISARMED);
}
static ew_frame_t calibration(uint64_t sequence) {
    ew_frame_t f=request(EW_CALIBRATION,1,sequence,0); f.length=232;
    for(unsigned i=0;i<6;++i) {
        uint8_t *v=f.payload+40+32*i;
        ew_put16(v,96); ew_put16(v+2,4000); ew_put16(v+4,900); ew_put16(v+6,3100);
        ew_putf32(v+8,.001f); ew_putf32(v+12,-2.f);
        ew_putf32(v+16,-1.1f); ew_putf32(v+20,1.1f);
        ex_joint_calibration_t c={96,4000,900,3100,(double).001f,-2.,(double)-1.1f,(double)1.1f,true};
        uint16_t lo,hi; assert(ex_joint_work_bounds(&c,&lo,&hi)==EX_OK);
        if(i==1) { ew_put16(v+24,hi); ew_put16(v+26,3700); }
    }
    return f;
}
static ew_frame_t arm_request(const ec_controller_t *c,uint8_t kind,uint64_t session,uint64_t sequence,uint16_t second) {
    ew_frame_t f=request(kind,session,sequence,0); f.length=66;
    for(unsigned i=0;i<6;++i) {
        const uint16_t raw=i==1?second:2000;
        ew_putf32(f.payload+40+4*i,(float)(c->calibration[i].radians_at_raw_zero+c->calibration[i].radians_per_tick*raw));
    }
    return f;
}
static void arm_generations_and_recovery(void) {
    ec_controller_t c; assert(ec_init(&c,123,0)==EX_OK);
    ew_frame_t f=request(EW_OPEN,1,1,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    f=calibration(2); assert(ec_dispatch(&c,&f,0)==EX_OK);
    for(unsigned i=0;i<6;++i) { ec_measure(&c,i,i==1?3300:2000,true,0); }
    f=request(EW_ARM_ENABLE,1,3,0); assert(ec_dispatch(&c,&f,0)==EX_SOFT_LIMIT);
    f=request(EW_RECOVERY_ENABLE,1,3,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(c.arm_raw[1]==3300 && !c.arm_pending); /* Enable never writes a start pose. */
    f=arm_request(&c,EW_ARM_RECOVER,1,4,3301); assert(ec_dispatch(&c,&f,0)==EX_SOFT_LIMIT);
    f=arm_request(&c,EW_ARM_RECOVER,1,4,3250); assert(ec_dispatch(&c,&f,0)==EX_RANGE);
    f=arm_request(&c,EW_ARM_RECOVER,1,4,3290); assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(c.arm_generation==1 && c.arm_raw[1]==3290);
    f=request(EW_RECOVERY_ENABLE,1,5,0); assert(ec_dispatch(&c,&f,0)==EX_NOT_READY);
    uint16_t lo,hi; assert(ex_joint_work_bounds(&c.calibration[1],&lo,&hi)==EX_OK);
    uint64_t seq=5;
    while(c.arm_raw[1]>hi) {
        ec_measure(&c,1,c.arm_raw[1],true,0);
        uint16_t next=c.arm_raw[1]-hi>10?(uint16_t)(c.arm_raw[1]-10):hi;
        f=arm_request(&c,EW_ARM_RECOVER,1,seq++,next); assert(ec_dispatch(&c,&f,0)==EX_OK);
    }
    assert(c.arm_raw[1]==hi);
    ec_measure(&c,1,hi,true,0);
    ex_joint_sample_t sample;
    assert(ex_joint_observe(&c.calibration[1],hi,0,0,250000,&sample)==EX_OK);
    assert(!sample.outside_soft_limit);
    uint64_t generation=c.arm_generation;
    f=request(EW_CANCEL,1,seq++,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(c.arm_cancel && c.arm_cancel_generation==1);
    f=request(EW_OPEN,2,1,0); assert(ec_dispatch(&c,&f,0)==EX_NOT_READY);
    c.arm_cancel=false; /* Simulated dispatcher completion for THIS generation. */
    assert(ec_dispatch(&c,&f,0)==EX_OK);
    for(unsigned i=0;i<6;++i) { ec_measure(&c,i,2000,true,0); }
    f=request(EW_ARM_ENABLE,2,2,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    f=arm_request(&c,EW_ARM,2,3,2400); assert(ec_dispatch(&c,&f,0)==EX_RANGE);
    assert(c.arm_generation==generation && !c.arm_pending);
    f=arm_request(&c,EW_ARM,2,4,2001); assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(c.arm_generation==generation+1); /* Sequence 4 from session 1 is NOT an acknowledgement. */
    f=request(EW_ARM_CANCEL,2,5,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(c.arm_cancel_generation==2);
    f=request(EW_ARM_CANCEL,2,6,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(c.arm_cancel_generation==2); /* Repeated cancellation does not starve the bus hold. */
}
static void calibration_atomicity_and_zero_runtime(void) {
    ec_controller_t c; assert(ec_init(&c,123,0)==EX_OK);
    ew_frame_t f=request(EW_OPEN,1,1,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    f=calibration(2); assert(ec_dispatch(&c,&f,0)==EX_OK);
    ex_joint_calibration_t before[6]; memcpy(before,c.calibration,sizeof(before));
    f=calibration(3); ew_putf32(f.payload+40+5*32+8,NAN);
    assert(ec_dispatch(&c,&f,0)!=EX_OK && memcmp(before,c.calibration,sizeof(before))==0);
    uint16_t raw[6]={2000,2000,2000,2000,2000,2000}; uint8_t packet[EX_SYNC_BYTES];
    assert(ex_servo_make_sync(raw,c.calibration,0,0,0,packet)==EX_OK);
    for(unsigned i=0;i<6;++i) { assert(packet[10+5*i]==0 && packet[11+5*i]==0); }
    f=request(EW_RGB,1,3,0); f.length=43; f.payload[40]=50;
    assert(ec_dispatch(&c,&f,0)==EX_OK && c.rgb_generation==1);
    f=request(EW_HOLD,1,4,0); assert(ec_dispatch(&c,&f,0)==EX_OK && c.rgb_generation==1);
}
static void arm_transmit_authorization(void) {
    ec_controller_t c; assert(ec_init(&c,123,0)==EX_OK);
    ew_frame_t f=request(EW_OPEN,1,1,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    f=calibration(2); assert(ec_dispatch(&c,&f,0)==EX_OK);
    for(unsigned i=0;i<6;++i) { ec_measure(&c,i,2000,true,0); }
    f=request(EW_ARM_ENABLE,1,3,0); assert(ec_dispatch(&c,&f,0)==EX_OK);
    f=arm_request(&c,EW_ARM,1,4,2001); assert(ec_dispatch(&c,&f,0)==EX_OK);
    ec_controller_t copy=c;
    assert(ec_arm_commit_allowed(&c,&copy,1,false));
    assert(!ec_arm_commit_allowed(&c,&copy,200000,false));
    c.measured_valid[5]=false;
    assert(!ec_arm_commit_allowed(&c,&copy,1,false));
    c.measured_valid[5]=true;
    ++c.arm_generation;
    assert(!ec_arm_commit_allowed(&c,&copy,1,false));
    copy=c; ++copy.base.session_id;
    assert(!ec_arm_commit_allowed(&c,&copy,1,false));
    copy=c; ++copy.base.boot_id;
    assert(!ec_arm_commit_allowed(&c,&copy,1,false));
    copy=c;
    /* Base and arm may coexist in the MCU; Jetson owns collision sequencing. */
    f=request(EW_BASE,1,5,0); f.length=60; ew_putf32(f.payload+40,.1f);
    assert(ec_dispatch(&c,&f,0)==EX_OK);
    assert(ec_arm_commit_allowed(&c,&copy,1,false));
    c.base.expires_us=10;
    assert(!ec_arm_commit_allowed(&c,&copy,10,false));
    c.base.finite_end_us=5;
    assert(ec_arm_commit_allowed(&c,&copy,10,false));
    f=(ew_frame_t){.type=EW_ESTOP}; assert(ec_dispatch(&c,&f,1)==EX_OK);
    assert(!ec_arm_commit_allowed(&c,&copy,1,false));
    assert(!ec_arm_commit_allowed(&c,&copy,1,true));
    copy=c;
    assert(ec_arm_commit_allowed(&c,&copy,300000,true));
    ++c.arm_cancel_generation;
    assert(!ec_arm_commit_allowed(&c,&copy,1,true));
    f=request(EW_CLEAR,2,1,1); assert(ec_dispatch(&c,&f,1)==EX_OK);
    /* A clear cannot make a pending measured hold eligible for new calibration. */
    c.base.mode=EX_BASE_ACTIVE; c.base.session_id=1;
    f=calibration(6); assert(ec_dispatch(&c,&f,1)==EX_NOT_READY);
}
int main(void) {
    wire(); sessions(); arm_generations_and_recovery(); calibration_atomicity_and_zero_runtime();
    arm_transmit_authorization();
    puts("wire fuzz, deadlines, replay, atomic calibration, bounded recovery and independent generations: PASS");
}
