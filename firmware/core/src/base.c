#include "internal.h"

static void fault(ex_base_t *s, ex_result_t reason) {
    s->mode = EX_BASE_FAULT;
    if (s->fault == EX_OK) { s->fault = reason; }
    s->command_present = false;
    s->target = ex_zero();
}
static ex_result_t clock_check(ex_base_t *s, uint64_t now) {
    if (now < s->last_now_us) { fault(s, EX_TIME); return EX_TIME; }
    s->last_now_us = now;
    return EX_OK;
}
static ex_result_t live(ex_base_t *s, uint64_t now) {
    ex_result_t r = clock_check(s, now);
    if (r != EX_OK) { return r; }
    if (s->mode == EX_BASE_FAULT) { return EX_LATCHED; }
    if (s->mode != EX_BASE_ACTIVE) { return EX_DISARMED; }
    if (now - s->last_tick_us > s->config.max_tick_gap_us) {
        fault(s, EX_TICK_OVERRUN); return EX_TICK_OVERRUN;
    }
    if (s->command_present && s->finite_end_us && now >= s->finite_end_us) {
        s->command_present = false;
        s->target = ex_zero();
        s->finite_end_us = 0;
    }
    if (s->command_present && now >= s->expires_us) {
        fault(s, EX_EXPIRED); return EX_EXPIRED;
    }
    return EX_OK;
}
ex_result_t ex_base_init(ex_base_t *s, const ex_base_config_t *c,
                        uint64_t boot, uint64_t now) {
    if (!s || !c) { return EX_ARGUMENT; }
    *s = (ex_base_t){0};
    s->mode = EX_BASE_DISARMED;
    if (!boot || !c->max_command_us || !c->max_tick_gap_us ||
        c->max_command_us > UINT64_C(60000000) ||
        c->max_tick_gap_us >= c->max_command_us ||
        !ex_finite(c->max_x) || c->max_x <= 0.0 ||
        !ex_finite(c->max_y) || c->max_y <= 0.0 ||
        !ex_finite(c->max_yaw) || c->max_yaw <= 0.0) {
        fault(s, EX_RANGE); return EX_RANGE;
    }
    s->config = *c;
    s->boot_id = boot;
    s->last_now_us = now;
    s->last_tick_us = now;
    return EX_OK;
}
ex_result_t ex_base_begin(ex_base_t *s, uint64_t boot, uint64_t session, uint64_t now) {
    if (!s) { return EX_ARGUMENT; }
    ex_result_t r = clock_check(s, now);
    if (r != EX_OK) { return r; }
    if (s->mode == EX_BASE_FAULT) { return EX_LATCHED; }
    if (s->mode != EX_BASE_DISARMED) { return EX_NOT_READY; }
    if (!s->boot_id || boot != s->boot_id || !session || session <= s->highest_session_id) {
        return EX_SESSION;
    }
    if (UINT64_MAX - now < s->config.max_command_us) { return EX_TIME; }
    s->session_id = session;
    s->highest_session_id = session;
    s->last_sequence = 0;
    s->command_present = false;
    s->target = ex_zero();
    s->expires_us = now + s->config.max_command_us;
    s->finite_end_us = 0;
    s->last_tick_us = now;
    s->mode = EX_BASE_ACTIVE;
    return EX_OK;
}
ex_result_t ex_base_submit(ex_base_t *s, const ex_base_command_t *c, uint64_t now) {
    if (!s || !c) { return EX_ARGUMENT; }
    ex_result_t r = live(s, now);
    if (r != EX_OK) { return r; }
    if (c->boot_id != s->boot_id || c->session_id != s->session_id) { return EX_SESSION; }
    if (!c->sequence || c->sequence <= s->last_sequence) { return EX_REPLAY; }
    if (c->issued_us > now || c->expires_us <= now || c->expires_us <= c->issued_us ||
        c->expires_us - c->issued_us > s->config.max_command_us) { return EX_EXPIRED; }
    if (!ex_twist_finite(c->velocity)) { return EX_VALUE; }
    if (ex_abs(c->velocity.x) > s->config.max_x ||
        ex_abs(c->velocity.y) > s->config.max_y ||
        ex_abs(c->velocity.yaw) > s->config.max_yaw) { return EX_RANGE; }
    if (c->finite_duration_us > c->expires_us - c->issued_us) { return EX_RANGE; }
    if (c->finite_duration_us && c->issued_us + c->finite_duration_us <= now) {
        return EX_EXPIRED;
    }
    s->last_sequence = c->sequence;
    s->expires_us = c->expires_us;
    s->finite_end_us = c->finite_duration_us ? c->issued_us + c->finite_duration_us : 0;
    s->target = c->velocity;
    s->command_present = c->velocity.x != 0 || c->velocity.y != 0 || c->velocity.yaw != 0;
    return EX_OK;
}
ex_result_t ex_base_tick(ex_base_t *s, uint64_t now, ex_twist_t *target) {
    if (!s || !target) { return EX_ARGUMENT; }
    *target = ex_zero();
    ex_result_t r = live(s, now);
    s->last_tick_us = now;
    if (r != EX_OK) { return r; }
    if (s->finite_end_us && now >= s->finite_end_us) {
        s->command_present = false;
        s->target = ex_zero();
        s->finite_end_us = 0;
    }
    if (s->command_present) { *target = s->target; }
    return EX_OK;
}
ex_result_t ex_base_hold(ex_base_t *s, uint64_t now) {
    if (!s) { return EX_ARGUMENT; }
    ex_result_t r = live(s, now);
    if (r != EX_OK) { return r; }
    s->command_present = false;
    s->target = ex_zero();
    s->finite_end_us = 0;
    return EX_OK;
}
ex_result_t ex_base_cancel(ex_base_t *s, uint64_t boot, uint64_t session, uint64_t now) {
    if (!s) { return EX_ARGUMENT; }
    ex_result_t r = clock_check(s, now);
    if (r != EX_OK) { return r; }
    if (s->mode == EX_BASE_FAULT) { return EX_LATCHED; }
    if (boot != s->boot_id || session != s->session_id || !session) { return EX_SESSION; }
    s->mode = EX_BASE_DISARMED;
    s->command_present = false;
    s->target = ex_zero();
    s->finite_end_us = 0;
    return EX_OK;
}
void ex_base_estop(ex_base_t *s) { if (s) { fault(s, EX_LATCHED); } }
ex_result_t ex_base_clear_fault(ex_base_t *s, uint64_t now) {
    if (!s) { return EX_ARGUMENT; }
    if (!s->boot_id || now < s->last_now_us) { return EX_TIME; }
    if (s->mode != EX_BASE_FAULT) { return EX_NOT_READY; }
    s->fault = EX_OK;
    s->mode = EX_BASE_DISARMED;
    s->command_present = false;
    s->target = ex_zero();
    s->last_now_us = now;
    s->last_tick_us = now;
    return EX_OK;
}

static bool geometry_valid(const ex_mecanum_config_t *c) {
    if (!c || !ex_finite(c->half_length_plus_half_width_m) ||
        c->half_length_plus_half_width_m <= 0.0) { return false; }
    for (size_t i = 0; i < EX_WHEELS; ++i) {
        if (!ex_finite(c->radius_m[i]) || c->radius_m[i] <= 0.0 ||
            !ex_finite(c->max_wheel_rad_s[i]) || c->max_wheel_rad_s[i] <= 0.0) { return false; }
    }
    return true;
}
ex_result_t ex_mecanum_inverse(const ex_mecanum_config_t *c, ex_twist_t v,
                              double out[EX_WHEELS], double *scale_out) {
    if (!out || !scale_out) { return EX_ARGUMENT; }
    if (!geometry_valid(c) || !ex_twist_finite(v)) { return EX_VALUE; }
    const double k = c->half_length_plus_half_width_m;
    double w[EX_WHEELS] = {
        (v.x - v.y - k*v.yaw)/c->radius_m[0],
        (v.x + v.y + k*v.yaw)/c->radius_m[1],
        (v.x + v.y - k*v.yaw)/c->radius_m[2],
        (v.x - v.y + k*v.yaw)/c->radius_m[3]
    };
    double scale = 1.0;
    for (size_t i = 0; i < EX_WHEELS; ++i) {
        if (!ex_finite(w[i])) { return EX_VALUE; }
        if (ex_abs(w[i])*scale > c->max_wheel_rad_s[i]) {
            scale = c->max_wheel_rad_s[i]/ex_abs(w[i]);
        }
    }
    for (size_t i = 0; i < EX_WHEELS; ++i) { out[i] = w[i]*scale; }
    *scale_out = scale;
    return EX_OK;
}
ex_result_t ex_mecanum_forward(const ex_mecanum_config_t *c,
                              const double w[EX_WHEELS], ex_twist_t *out) {
    if (!w || !out) { return EX_ARGUMENT; }
    if (!geometry_valid(c)) { return EX_VALUE; }
    double u[EX_WHEELS];
    for (size_t i = 0; i < EX_WHEELS; ++i) {
        if (!ex_finite(w[i])) { return EX_VALUE; }
        u[i] = w[i]*c->radius_m[i];
        if (!ex_finite(u[i])) { return EX_VALUE; }
    }
    ex_twist_t v = {(u[0]+u[1]+u[2]+u[3])*0.25,
                    (-u[0]+u[1]+u[2]-u[3])*0.25,
                    (-u[0]+u[1]-u[2]+u[3])/(4.0*c->half_length_plus_half_width_m)};
    if (!ex_twist_finite(v)) { return EX_VALUE; }
    *out = v;
    return EX_OK;
}
ex_result_t ex_encoder_delta16(uint16_t prev, uint16_t cur, uint16_t max_abs, int32_t *out) {
    if (!out) { return EX_ARGUMENT; }
    if (!max_abs || max_abs > 32767u) { return EX_RANGE; }
    const uint32_t modular = ((uint32_t)cur + 65536u - (uint32_t)prev) & 65535u;
    if (modular == 32768u) { return EX_RANGE; } /* Ambiguous half-turn. */
    const int32_t delta = modular <= 32767u ? (int32_t)modular : (int32_t)modular - 65536;
    if (delta > (int32_t)max_abs || delta < -(int32_t)max_abs) { return EX_RANGE; }
    *out = delta;
    return EX_OK;
}
void ex_pid_reset(ex_pid_t *s) { if (s) { *s = (ex_pid_t){0}; } }
ex_result_t ex_pid_step(ex_pid_t *s, const ex_pid_config_t *c,
                       double target, double measured, double dt, double *out) {
    if (!s || !c || !out) { return EX_ARGUMENT; }
    *out = 0.0;
    if (!ex_finite(target) || !ex_finite(measured) || !ex_finite(dt) || dt <= 0.0 ||
        !ex_finite(c->max_dt_s) || c->max_dt_s <= 0.0 || dt > c->max_dt_s ||
        !ex_finite(c->kp) || c->kp < 0.0 || !ex_finite(c->ki) || c->ki < 0.0 ||
        !ex_finite(c->kd) || c->kd < 0.0 || !ex_finite(c->kff) || c->kff < 0.0 ||
        !ex_finite(c->output_limit) || c->output_limit <= 0.0 ||
        !ex_finite(c->integral_limit) || c->integral_limit < 0.0 ||
        !ex_finite(s->integral) || !ex_finite(s->previous_measurement)) { return EX_VALUE; }
    const double error = target - measured;
    const double derivative = s->initialized ? (measured - s->previous_measurement)/dt : 0.0;
    const double base = c->kp*error - c->kd*derivative + c->kff*target;
    const double increment = c->ki*error*dt;
    if (!ex_finite(error) || !ex_finite(derivative) || !ex_finite(base) || !ex_finite(increment)) { return EX_VALUE; }
    double integral = ex_clamp(s->integral+increment, -c->integral_limit, c->integral_limit);
    double candidate = base+integral;
    if (!ex_finite(candidate)) { return EX_VALUE; }
    if ((candidate > c->output_limit && error > 0.0) ||
        (candidate < -c->output_limit && error < 0.0)) { integral = s->integral; }
    candidate = base+integral;
    if (!ex_finite(candidate)) { return EX_VALUE; }
    s->integral = integral;
    s->previous_measurement = measured;
    s->initialized = true;
    *out = ex_clamp(candidate, -c->output_limit, c->output_limit);
    return EX_OK;
}
