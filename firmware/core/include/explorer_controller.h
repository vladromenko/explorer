#ifndef EXPLORER_CONTROLLER_H
#define EXPLORER_CONTROLLER_H
#include "explorer_core.h"
#include "explorer_wire.h"

/* Every mutating request except ESTOP has boot/session/seq/issued/expiry (5*u64).
 * Exactly one owner: the hardware periodic interrupt calls dispatch and tick.
 * Communications posts immutable frames through the board's bounded queue. */
typedef struct {
    ex_base_t base;
    uint64_t sequence, arm_expires_us, arm_generation, rgb_generation;
    uint32_t arm_cancel_generation;
    bool arm_enabled, arm_pending, arm_cancel;
    bool arm_recovery;
    ex_result_t arm_stop_reason;
    uint16_t arm_raw[6], arm_runtime_ms;
    uint16_t recovery_min[6], recovery_max[6];
    ex_joint_calibration_t calibration[6];
    uint16_t measured_raw[6];
    uint64_t measured_us[6];
    bool measured_valid[6];
    uint8_t rgb[3]; bool rgb_pending, beep;
    uint32_t rejected, accepted;
} ec_controller_t;
ex_result_t ec_init(ec_controller_t *c,uint64_t boot,uint64_t now);
ex_result_t ec_dispatch(ec_controller_t *c,const ew_frame_t *frame,uint64_t now);
ex_result_t ec_tick(ec_controller_t *c,uint64_t now,ex_twist_t *velocity);
void ec_measure(ec_controller_t *c,unsigned joint,uint16_t raw,bool valid,uint64_t now);
/* Final check against the ISR-owned live state, immediately before enqueueing
 * a frame on the physical arm bus. The caller makes check + enqueue atomic. */
bool ec_arm_commit_allowed(const ec_controller_t *live,const ec_controller_t *copy,
                           uint64_t now,bool cancel);
/* Called by the control owner for a latched arm transport failure. */
void ec_cancel_arm(ec_controller_t *controller, ex_result_t reason);
#endif
