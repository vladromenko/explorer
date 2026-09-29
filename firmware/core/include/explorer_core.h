#ifndef EXPLORER_CORE_H
#define EXPLORER_CORE_H

/* Portable firmware logic. NOT a board firmware or a hardware safety system.
 * All entry points operate on caller-owned state. A single control-task owner
 * must serialize them; do not call from an ISR and a ROS callback concurrently.
 * No function opens a port, drives GPIO, allocates memory, or writes flash.
 */
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#define EX_WHEELS 4u
#define EX_JOINTS 6u
#define EX_SYNC_BYTES 38u

typedef enum {
    EX_OK = 0, EX_ARGUMENT, EX_VALUE, EX_RANGE, EX_TIME,
    EX_DISARMED, EX_LATCHED, EX_SESSION, EX_REPLAY, EX_EXPIRED,
    EX_FRAME, EX_CHECKSUM, EX_DEVICE_ERROR, EX_UNCALIBRATED, EX_STALE,
    EX_SOFT_LIMIT, EX_NOT_READY, EX_TICK_OVERRUN
} ex_result_t;

typedef struct { double x, y, yaw; } ex_twist_t;
typedef enum { EX_BASE_DISARMED, EX_BASE_ACTIVE, EX_BASE_FAULT } ex_base_mode_t;
typedef struct {
    uint64_t max_command_us, max_tick_gap_us;
    double max_x, max_y, max_yaw;
} ex_base_config_t;
typedef struct {
    uint64_t boot_id, session_id, sequence;
    /* MCU monotonic time domain, NOT unsynchronized Jetson wall clock.
     * Transport adapter must establish and validate the clock mapping. */
    uint64_t issued_us, expires_us;
    ex_twist_t velocity;
    /* 0: stream; otherwise stop velocity after this many microseconds. */
    uint64_t finite_duration_us;
} ex_base_command_t;
typedef struct {
    ex_base_config_t config;
    uint64_t boot_id, session_id, highest_session_id, last_sequence;
    uint64_t last_now_us, last_tick_us, expires_us, finite_end_us;
    bool command_present;
    ex_base_mode_t mode;
    ex_result_t fault;
    ex_twist_t target;
} ex_base_t;

ex_result_t ex_base_init(ex_base_t *state, const ex_base_config_t *config,
                        uint64_t boot_id, uint64_t now_us);
/* Transport's authenticated/authorized session-open endpoint, not auto-retry.
 * session_id strictly increases for a particular boot_id. */
ex_result_t ex_base_begin(ex_base_t *state, uint64_t boot_id,
                         uint64_t session_id, uint64_t now_us);
ex_result_t ex_base_submit(ex_base_t *state, const ex_base_command_t *command,
                          uint64_t now_us);
/* Must run in a periodic control task independent of UART/ROS/servo polling.
 * Output is a VELOCITY TARGET, not an assertion that the wheels stopped. */
ex_result_t ex_base_tick(ex_base_t *state, uint64_t now_us, ex_twist_t *target);
/* A normal hold does not latch a fault or renew the command expiry. */
ex_result_t ex_base_hold(ex_base_t *state, uint64_t now_us);
ex_result_t ex_base_cancel(ex_base_t *state, uint64_t boot_id,
                          uint64_t session_id, uint64_t now_us);
void ex_base_estop(ex_base_t *state);
/* Deliberate recovery endpoint; output remains zero, a NEW session is needed. */
ex_result_t ex_base_clear_fault(ex_base_t *state, uint64_t now_us);

typedef struct {
    double radius_m[EX_WHEELS];
    double half_length_plus_half_width_m;
    double max_wheel_rad_s[EX_WHEELS];
    /* Output shaft/encoder signs are normalized by the board port, not here.
     * Logical order: FRONT_LEFT, FRONT_RIGHT, REAR_LEFT, REAR_RIGHT. */
} ex_mecanum_config_t;
ex_result_t ex_mecanum_inverse(const ex_mecanum_config_t *config,
                              ex_twist_t twist, double wheel_rad_s[EX_WHEELS],
                              double *common_scale);
ex_result_t ex_mecanum_forward(const ex_mecanum_config_t *config,
                              const double wheel_rad_s[EX_WHEELS], ex_twist_t *twist);
ex_result_t ex_encoder_delta16(uint16_t previous, uint16_t current,
                              uint16_t max_abs_delta, int32_t *delta);

typedef struct {
    double kp, ki, kd, kff, output_limit, integral_limit, max_dt_s;
} ex_pid_config_t;
typedef struct { double integral, previous_measurement; bool initialized; } ex_pid_t;
void ex_pid_reset(ex_pid_t *state);
ex_result_t ex_pid_step(ex_pid_t *state, const ex_pid_config_t *config,
                       double target, double measurement, double dt_s, double *output);

typedef struct {
    uint16_t raw_measure_min, raw_measure_max;
    uint16_t raw_command_min, raw_command_max;
    double radians_per_tick, radians_at_raw_zero;
    double soft_min_rad, soft_max_rad;
    bool calibrated;
} ex_joint_calibration_t;
typedef struct {
    uint16_t raw;
    double position_rad;
    uint64_t acquired_us;
    bool raw_valid, position_valid, outside_soft_limit;
    ex_result_t error;
} ex_joint_sample_t;
/* Header is explicitly configured: 0xF5 or 0xFF. No auto-detection from echo. */
ex_result_t ex_servo_decode_position(const uint8_t *frame, size_t length,
                                    uint8_t expected_id, uint8_t reply_header2,
                                    uint16_t *raw, uint8_t *device_error);
ex_result_t ex_joint_observe(const ex_joint_calibration_t *calibration,
                            uint16_t raw, uint64_t acquired_us,
                            uint64_t now_us, uint64_t max_age_us,
                            ex_joint_sample_t *sample);
/* Representable raw interval inside BOTH raw and signed position limits. */
ex_result_t ex_joint_work_bounds(const ex_joint_calibration_t *calibration,
                                uint16_t *lower, uint16_t *upper);
ex_result_t ex_joint_to_raw(const ex_joint_calibration_t *calibration,
                           double position_rad, uint16_t *raw);
/* Validate ALL targets first; on error output is unchanged. No fallback to 2000. */
ex_result_t ex_arm_targets_to_raw(const ex_joint_calibration_t calibration[EX_JOINTS],
                                const double position_rad[EX_JOINTS],
                                uint16_t raw[EX_JOINTS]);
ex_result_t ex_servo_make_read_request(uint8_t id, uint8_t frame[8]);
/* Packet layout from supplied V4 source; requires physical protocol validation.
 * min/max runtime are explicit port configuration, not an assumed servo feature. */
ex_result_t ex_servo_make_sync(const uint16_t raw[EX_JOINTS],
                              const ex_joint_calibration_t calibration[EX_JOINTS],
                              uint16_t runtime_ms, uint16_t min_runtime_ms,
                              uint16_t max_runtime_ms, uint8_t frame[EX_SYNC_BYTES]);

typedef struct {
    double position_min, position_max, velocity_max, acceleration_max, jerk_max;
} ex_profile_limits_t;
typedef struct {
    double start[EX_JOINTS], goal[EX_JOINTS], duration_s;
    bool valid;
} ex_profile_t;
typedef struct {
    double position[EX_JOINTS], velocity[EX_JOINTS];
    double acceleration[EX_JOINTS], jerk[EX_JOINTS];
    bool finished;
} ex_profile_sample_t;
/* Synchronized REST-TO-REST quintic. Not a general path planner, streaming
 * replanner, collision checker, or emergency deceleration algorithm. */
ex_result_t ex_profile_plan(const double start[EX_JOINTS],
                            const double goal[EX_JOINTS],
                            const ex_profile_limits_t limits[EX_JOINTS],
                            double requested_duration_s, ex_profile_t *profile);
ex_result_t ex_profile_sample(const ex_profile_t *profile, double elapsed_s,
                              ex_profile_sample_t *sample);

#endif
