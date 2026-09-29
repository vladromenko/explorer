#include "servo_bus.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

UART_HandleTypeDef board_uart[PORT_COUNT];
volatile uint32_t board_rx_overruns[PORT_COUNT],board_uart_errors[PORT_COUNT];
volatile uint32_t board_tx_completed[PORT_COUNT];
volatile uint64_t board_tx_completed_us[PORT_COUNT];
static uint64_t now,busy_until,write_settled_at;
static uint8_t rx[2048];
static uint64_t rx_at[2048];
static unsigned rx_read,rx_length,queries[6],good[6],bad[6],sync_count,partial_count;
static unsigned target_count,commit_attempts,inject_commit;
static uint64_t first_query[6];
static uint8_t last_sync[38];
static uint16_t last_sync_length;
static bool tx_active,hang_tx,fail_write;
static bool complete_on_irq_restore;
static ec_controller_t *live;
static unsigned fault_seen;
static uint8_t last_report[6][30];
enum response_mode { NORMAL,ABSENT,DROP_ONCE,WRONG_ID,WRONG_HEADER,BAD_CHECKSUM,DEVICE_ERROR,OUT_OF_RANGE,TRUNCATED };
static enum response_mode mode[6];

uint64_t board_time_us(void) { return now; }
void mock_restore_irq(uint32_t value) {
    (void)value;
    if(complete_on_irq_restore) {
        complete_on_irq_restore=false;
        board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
        board_tx_completed_us[PORT_ARM]=now;
        ++board_tx_completed[PORT_ARM]; tx_active=false;
    }
}
static void enqueue(const uint8_t *bytes,unsigned length,uint64_t at) {
    assert(rx_length+length<=sizeof(rx));
    for(unsigned i=0;i<length;++i) { rx[rx_length]=bytes[i]; rx_at[rx_length++]=at; }
}
static void response(uint8_t id,uint16_t raw,uint8_t out[8]) {
    uint8_t frame[8]={255,245,id,4,0,(uint8_t)(raw>>8),(uint8_t)raw,0};
    uint8_t sum=0; for(unsigned i=2;i<7;++i) { sum=(uint8_t)(sum+frame[i]); }
    frame[7]=(uint8_t)~sum; memcpy(out,frame,8);
}
bool board_receive(unsigned port,uint8_t *value) {
    assert(port==PORT_ARM);
    if(rx_read<rx_length && now>=rx_at[rx_read]) {
        *value=rx[rx_read++];
        if(rx_read==rx_length) { rx_read=0; rx_length=0; }
        return true;
    }
    return false;
}
bool board_send(unsigned port,const uint8_t *bytes,uint16_t length) {
    assert(port==PORT_ARM && board_uart[port].gState==HAL_UART_STATE_READY);
    assert(bytes[0]==255 && bytes[1]==255);
    if(bytes[2]==254 && fail_write) { return false; }
    board_uart[port].gState=1; busy_until=now+87u*length; tx_active=true;
    if(bytes[2]==254) {
        write_settled_at=busy_until+SERVO_WRITE_QUIET_US;
        assert(length>=13 && length<=38 && length==bytes[3]+4);
        uint8_t sum=0; for(unsigned i=2;i<length-1u;++i) { sum=(uint8_t)(sum+bytes[i]); }
        assert((uint8_t)~sum==bytes[length-1]);
        ++sync_count; if(length<38) { ++partial_count; }
        memcpy(last_sync,bytes,length); last_sync_length=length;
        for(unsigned at=7;at<length-1u;at+=5) {
            assert(bytes[at]>=1 && bytes[at]<=6);
            assert(bytes[at+3]==0 && bytes[at+4]==0);
        }
    } else {
        assert(now>=write_settled_at);
        assert(length==8 && bytes[2]>=1 && bytes[2]<=6 && bytes[4]==2);
        unsigned id=bytes[2]; ++queries[id-1];
        if(!first_query[id-1]) { first_query[id-1]=now; }
        enum response_mode selected=mode[id-1];
        if(selected==DROP_ONCE) { mode[id-1]=NORMAL; }
        if(selected!=ABSENT && selected!=DROP_ONCE) {
            uint8_t data[8]; response((uint8_t)id,selected==OUT_OF_RANGE?7:2000,data);
            if(selected==WRONG_ID) { response(id==6?1:(uint8_t)(id+1),2000,data); }
            if(selected==WRONG_HEADER) { data[1]=255; }
            if(selected==BAD_CHECKSUM) { data[7]^=1; }
            if(selected==DEVICE_ERROR) { data[4]=1; --data[7]; }
            enqueue(data,selected==TRUNCATED?5:8,busy_until+700);
        }
    }
    return true;
}
bool servo_bus_commit(const ec_controller_t *snapshot,const uint8_t *frame,uint16_t length,bool cancel) {
    ++commit_attempts;
    if(inject_commit==1) { live->arm_cancel=true; ++live->arm_cancel_generation; live->arm_enabled=false; }
    if(inject_commit==2) { ++live->arm_generation; }
    if(inject_commit==3) { ++live->base.session_id; }
    if(inject_commit==4) { live->arm_expires_us=now; }
    inject_commit=0;
    bool allowed=ec_arm_commit_allowed(live,snapshot,now+servo_write_budget_us(length),cancel);
    bool sent=allowed && board_send(PORT_ARM,frame,length);
    if(sent && !cancel) { ++target_count; }
    return sent;
}
bool board_emit(uint8_t type,const uint8_t *payload,size_t length) {
    assert(type==EW_SERVO && length==30 && payload[8]>=1 && payload[8]<=6);
    unsigned i=payload[8]-1u;
    if(payload[9]==EX_OK) { ++good[i]; } else { ++bad[i]; }
    memcpy(last_report[i],payload,30); return true;
}
static void advance(ec_controller_t *c,unsigned microseconds) {
    assert(microseconds%100==0);
    for(unsigned i=0;i<microseconds/100;++i) {
        now+=100;
        if(tx_active && !hang_tx && now>=busy_until) {
            board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
            board_tx_completed_us[PORT_ARM]=now;
            ++board_tx_completed[PORT_ARM]; tx_active=false;
        }
        for(unsigned j=0;j<6;++j) {
            ec_measure(c,j,servo_measurements[j].raw,servo_measurements[j].valid,servo_measurements[j].time);
        }
        if(fault_seen!=servo_write_fault_generation) {
            fault_seen=servo_write_fault_generation;
            c->arm_enabled=false; c->arm_pending=false; c->arm_cancel=true; ++c->arm_cancel_generation;
        }
        ec_controller_t snapshot=*c; live=c; servo_bus_poll(&snapshot);
    }
}
static void reset(ec_controller_t *c) {
    now=1000; busy_until=0; write_settled_at=0;
    memset(board_uart,0,sizeof(board_uart)); memset((void*)board_tx_completed,0,sizeof(board_tx_completed));
    memset((void*)board_tx_completed_us,0,sizeof(board_tx_completed_us));
    memset((void*)board_uart_errors,0,sizeof(board_uart_errors)); memset((void*)board_rx_overruns,0,sizeof(board_rx_overruns));
    memset(mode,0,sizeof(mode)); memset(queries,0,sizeof(queries)); memset(good,0,sizeof(good)); memset(bad,0,sizeof(bad));
    memset(first_query,0,sizeof(first_query)); memset(last_report,0,sizeof(last_report));
    rx_read=rx_length=sync_count=partial_count=target_count=commit_attempts=inject_commit=fault_seen=0;
    tx_active=hang_tx=fail_write=complete_on_irq_restore=false; last_sync_length=0; live=c;
    assert(ec_init(c,1,now)==EX_OK);
    assert(ex_base_begin(&c->base,1,1,now)==EX_OK);
    servo_bus_init();
    for(unsigned i=0;i<6;++i) {
        c->arm_raw[i]=2000;
        c->calibration[i]=(ex_joint_calibration_t){96,4000,900,3100,.001,-2.,-1.1,1.1,true};
    }
}
static void warm(ec_controller_t *c) {
    reset(c); advance(c,50000);
    assert(sync_count==0);
    for(unsigned i=0;i<6;++i) { assert(good[i]>0 && servo_measurements[i].valid); }
}
static void enable(ec_controller_t *c) {
    c->arm_enabled=true; c->arm_pending=true; c->arm_cancel=false;
    ++c->arm_generation; c->arm_expires_us=now+200000;
}
static void until_write(ec_controller_t *c,unsigned expected) {
    for(unsigned i=0;i<1000 && sync_count<expected;++i) { advance(c,100); }
    assert(sync_count==expected);
}
static void test_startup_and_stream(void) {
    ec_controller_t c; warm(&c); enable(&c);
    for(unsigned cycle=0;cycle<100;++cycle) {
        ++c.arm_generation; c.arm_expires_us=now+200000; advance(&c,20000);
        for(unsigned i=0;i<6;++i) { assert(servo_measurements[i].valid && now-servo_measurements[i].time<250000); }
    }
    assert(target_count>=90 && servo_write_fault_generation==0);
}
static void test_retry_age_and_bad_frames(void) {
    ec_controller_t c; reset(&c); mode[0]=DROP_ONCE;
    for(unsigned i=0;i<600 && !good[0];++i) { advance(&c,100); }
    assert(good[0]==1 && queries[0]==2 && bad[0]==0);
    assert(servo_measurements[0].time==first_query[0] && now-first_query[0]>20000);
    const enum response_mode failures[]={ABSENT,WRONG_ID,WRONG_HEADER,BAD_CHECKSUM,TRUNCATED};
    for(unsigned k=0;k<sizeof(failures)/sizeof(failures[0]);++k) {
        reset(&c); mode[0]=failures[k];
        for(unsigned i=0;i<800 && !bad[0];++i) { advance(&c,100); }
        assert(bad[0]==1 && queries[0]==3 && !servo_measurements[0].valid);
        advance(&c,20000); assert(good[1]>0);
    }
    reset(&c); mode[0]=DEVICE_ERROR; advance(&c,3000);
    assert(bad[0]==1 && queries[0]==1 && servo_measurements[0].error==EX_DEVICE_ERROR);
    reset(&c); mode[0]=OUT_OF_RANGE; advance(&c,3000);
    assert(bad[0]==1 && queries[0]==1 && servo_measurements[0].error==EX_RANGE);
}
static void test_late_and_overlapping_replies(void) {
    ec_controller_t c; reset(&c); mode[0]=ABSENT; advance(&c,100);
    uint8_t data[8]; response(1,2000,data);
    enqueue(data,8,first_query[0]+20000); advance(&c,20100);
    assert(good[0]==0 && !servo_measurements[0].valid);
    advance(&c,2200); enqueue(data,8,now); advance(&c,100);
    assert(good[0]==1 && servo_measurements[0].time==first_query[0]);
    reset(&c); mode[0]=ABSENT; advance(&c,1000);
    const uint8_t prefix[5]={255,245,99,4,0};
    enqueue(prefix,5,now); enqueue(data,8,now); advance(&c,100);
    assert(good[0]==1 && bad[0]==0);
    reset(&c);
    uint8_t junk[129]={0}; enqueue(junk,sizeof(junk),now); enqueue(data,8,now);
    advance(&c,100); assert(queries[0]==0 && good[0]==0);
    advance(&c,100); assert(queries[0]==1 && good[0]==0);
}
static void test_completion_and_stale_snapshot(void) {
    ec_controller_t c; warm(&c); enable(&c); until_write(&c,1);
    assert(servo_sent_generation==0);
    advance(&c,1000); assert(servo_sent_generation==0);
    advance(&c,3000); assert(servo_sent_generation==c.arm_generation);
    for(unsigned injection=1;injection<=4;++injection) {
        warm(&c); enable(&c); inject_commit=injection;
        for(unsigned i=0;i<1000 && !commit_attempts;++i) { advance(&c,100); }
        assert(commit_attempts==1 && target_count==0 && servo_sent_generation==0);
    }
    warm(&c); enable(&c); c.arm_expires_us=now+1000; advance(&c,5000);
    assert(target_count==0 && servo_sent_generation==0);
}
static void test_cancel_priority_and_partial_holds(void) {
    ec_controller_t c; warm(&c); mode[0]=ABSENT;
    unsigned before=queries[0];
    for(unsigned i=0;i<300 && queries[0]==before;++i) { advance(&c,100); }
    c.arm_cancel=true; c.arm_cancel_generation=1;
    uint64_t cancelled=now; until_write(&c,1);
    assert(now-cancelled<2000 && servo_cancel_completed_generation==0);
    advance(&c,4000); assert(servo_cancel_completed_generation==1);
    c.arm_cancel=false; advance(&c,90000);
    assert(!servo_measurements[0].valid);
    c.arm_cancel=true; c.arm_cancel_generation=2; unsigned before_partial=partial_count;
    advance(&c,70000);
    assert(partial_count>before_partial && servo_cancel_completed_generation==1);
    mode[0]=NORMAL; advance(&c,100000);
    assert(servo_cancel_completed_generation==2 && last_sync_length==38);
    c.arm_cancel=false; enable(&c); unsigned expected=sync_count+1;
    until_write(&c,expected);
    c.arm_cancel=true; ++c.arm_cancel_generation; c.arm_enabled=false;
    advance(&c,15000);
    assert(servo_cancel_completed_generation==c.arm_cancel_generation);
}
static void test_uart_abort_and_hang(void) {
    ec_controller_t c; warm(&c); enable(&c); until_write(&c,1);
    tx_active=false; board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
    advance(&c,100);
    assert(servo_sent_generation==0 && servo_write_fault_generation==1);
    for(unsigned i=0;i<6;++i) { assert(!servo_measurements[i].valid); }
    warm(&c); enable(&c); until_write(&c,1); hang_tx=true;
    advance(&c,21000);
    assert(servo_sent_generation==0 && servo_write_fault_generation==1);
    advance(&c,20000); assert(servo_write_fault_generation==1);
    hang_tx=false; advance(&c,100);
    assert(servo_sent_generation==0); /* A late TC cannot repair an expired transmission. */
    warm(&c); enable(&c); fail_write=true; advance(&c,5000);
    assert(sync_count==0 && servo_sent_generation==0 && servo_write_fault_generation==0);
    fail_write=false; until_write(&c,1); advance(&c,4000);
    assert(servo_sent_generation==c.arm_generation);
}
static void test_completion_observed_after_poll_delay(void) {
    ec_controller_t c; warm(&c); enable(&c); until_write(&c,1);
    /* The UART interrupt finished normally, but main was not scheduled for
     * 25 ms (below the board's 30 ms main-liveness limit). */
    tx_active=false; board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
    board_tx_completed_us[PORT_ARM]=busy_until;
    ++board_tx_completed[PORT_ARM];
    now+=25000;
    ec_controller_t snapshot=c; servo_bus_poll(&snapshot);
    assert(servo_write_fault_generation==0);
    assert(servo_sent_generation==c.arm_generation);
    /* Timely poll is not assumed: a genuinely late TC must still be rejected
     * when both the deadline and the callback precede the next main poll. */
    warm(&c); enable(&c); until_write(&c,1);
    tx_active=false; board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
    board_tx_completed_us[PORT_ARM]=now+21000;
    ++board_tx_completed[PORT_ARM]; now+=25000;
    snapshot=c; servo_bus_poll(&snapshot);
    assert(servo_write_fault_generation==1 && servo_sent_generation==0);
    /* Old or unrelated completions cannot be attributed to this frame. */
    warm(&c); enable(&c); until_write(&c,1);
    tx_active=false; board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
    board_tx_completed_us[PORT_ARM]=busy_until;
    board_tx_completed[PORT_ARM]+=2; now+=25000;
    snapshot=c; servo_bus_poll(&snapshot);
    assert(servo_write_fault_generation==1 && servo_sent_generation==0);
}
static void test_tc_between_snapshot_and_ready_check(void) {
    ec_controller_t c; warm(&c); enable(&c); until_write(&c,1);
    now=busy_until; complete_on_irq_restore=true;
    ec_controller_t snapshot=c; servo_bus_poll(&snapshot);
    /* TC arrived after the atomic snapshot. READY must come from that same
     * snapshot, otherwise this legitimate completion looks like an abort. */
    assert(!complete_on_irq_restore && servo_write_fault_generation==0);
    servo_bus_poll(&snapshot);
    assert(servo_write_fault_generation==0 && servo_sent_generation==c.arm_generation);
}
int main(void) {
    test_tc_between_snapshot_and_ready_check();
    test_completion_observed_after_poll_delay();
    test_startup_and_stream(); test_retry_age_and_bad_frames(); test_late_and_overlapping_replies();
    test_completion_and_stale_snapshot(); test_cancel_priority_and_partial_holds(); test_uart_abort_and_hang();
    puts("servo dispatcher fault injection: deadlines, bounded retry age, parser resync, final commit guard, TC completion, cancellation priority, partial hold, UART abort/hang: PASS");
}
