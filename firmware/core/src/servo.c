#include "internal.h"
#include <math.h>

static bool raw_ranges_valid(const ex_joint_calibration_t *c) {
    return c && c->raw_measure_min < c->raw_measure_max &&
        c->raw_command_min < c->raw_command_max &&
        c->raw_command_min >= c->raw_measure_min &&
        c->raw_command_max <= c->raw_measure_max;
}
static bool calibration_valid(const ex_joint_calibration_t *c) {
    return raw_ranges_valid(c) && c->calibrated &&
        ex_finite(c->radians_per_tick) && c->radians_per_tick != 0.0 &&
        ex_finite(c->radians_at_raw_zero) &&
        ex_finite(c->soft_min_rad) && ex_finite(c->soft_max_rad) &&
        c->soft_min_rad < c->soft_max_rad;
}
static uint8_t checksum(const uint8_t *frame, size_t end) {
    uint32_t sum = 0;
    for (size_t i = 2; i < end; ++i) { sum += frame[i]; }
    return (uint8_t)(~sum & 255u);
}
ex_result_t ex_servo_decode_position(const uint8_t *f, size_t len,
                                    uint8_t id, uint8_t head2,
                                    uint16_t *raw, uint8_t *device_error) {
    if (!f || !raw || !device_error) { return EX_ARGUMENT; }
    *device_error = 0;
    if (id < 1u || id > EX_JOINTS || (head2 != 0xffu && head2 != 0xf5u)) { return EX_ARGUMENT; }
    if (len != 8u || f[0] != 0xffu || f[1] != head2 || f[2] != id || f[3] != 4u) { return EX_FRAME; }
    if (f[7] != checksum(f, 7u)) { return EX_CHECKSUM; }
    if (f[4] != 0u) { *device_error = f[4]; return EX_DEVICE_ERROR; }
    *raw = (uint16_t)(((uint16_t)f[5] << 8u) | f[6]);
    return EX_OK;
}
ex_result_t ex_joint_observe(const ex_joint_calibration_t *c,
                            uint16_t raw, uint64_t acquired, uint64_t now,
                            uint64_t max_age, ex_joint_sample_t *sample) {
    if (!c || !sample) { return EX_ARGUMENT; }
    *sample = (ex_joint_sample_t){0};
    sample->raw = raw;
    sample->acquired_us = acquired;
    sample->error = EX_RANGE;
    if (!raw_ranges_valid(c) || raw < c->raw_measure_min || raw > c->raw_measure_max) {
        return sample->error;
    }
    sample->raw_valid = true;
    if (!max_age || acquired > now || now-acquired >= max_age) {
        sample->error = EX_STALE; return sample->error;
    }
    if (!calibration_valid(c)) { sample->error = EX_UNCALIBRATED; return sample->error; }
    const double position = c->radians_at_raw_zero + c->radians_per_tick*(double)raw;
    if (!ex_finite(position)) { sample->error = EX_VALUE; return sample->error; }
    sample->position_rad = position;
    sample->position_valid = true;
    sample->outside_soft_limit = position < c->soft_min_rad || position > c->soft_max_rad;
    sample->error = sample->outside_soft_limit ? EX_SOFT_LIMIT : EX_OK;
    /* Data remain valid even outside a soft limit. Recovery planning is separate. */
    return sample->error;
}
ex_result_t ex_joint_work_bounds(const ex_joint_calibration_t *c,uint16_t *lower,uint16_t *upper) {
    if(!c || !lower || !upper) { return EX_ARGUMENT; }
    if(!calibration_valid(c)) { return EX_UNCALIBRATED; }
    double a=(c->soft_min_rad-c->radians_at_raw_zero)/c->radians_per_tick;
    double b=(c->soft_max_rad-c->radians_at_raw_zero)/c->radians_per_tick;
    if(!ex_finite(a) || !ex_finite(b)) { return EX_RANGE; }
    double lo=a<b?a:b,hi=a>b?a:b;
    if(lo<c->raw_command_min) { lo=c->raw_command_min; }
    if(hi>c->raw_command_max) { hi=c->raw_command_max; }
    if(lo>hi || lo>UINT16_MAX || hi<0) { return EX_SOFT_LIMIT; }
    uint32_t first=(uint32_t)ceil(lo),last=(uint32_t)floor(hi);
    if(first>last) { return EX_SOFT_LIMIT; }
    /* Recheck in the forward transform used for measured feedback, so floating
     * point inversion can never admit one tick beyond a signed soft limit. */
    double q=c->radians_at_raw_zero+c->radians_per_tick*first;
    if(q<c->soft_min_rad || q>c->soft_max_rad) { ++first; }
    q=c->radians_at_raw_zero+c->radians_per_tick*last;
    if(q<c->soft_min_rad || q>c->soft_max_rad) {
        if(last==0) { return EX_SOFT_LIMIT; }
        --last;
    }
    if(first>last) { return EX_SOFT_LIMIT; }
    *lower=(uint16_t)first; *upper=(uint16_t)last; return EX_OK;
}
ex_result_t ex_joint_to_raw(const ex_joint_calibration_t *c, double q, uint16_t *raw) {
    if (!c || !raw) { return EX_ARGUMENT; }
    if (!calibration_valid(c)) { return EX_UNCALIBRATED; }
    if (!ex_finite(q)) { return EX_VALUE; }
    if (q < c->soft_min_rad || q > c->soft_max_rad) { return EX_SOFT_LIMIT; }
    const double coordinate = (q-c->radians_at_raw_zero)/c->radians_per_tick;
    /* Check BEFORE narrowing to uint16_t. No signed wrap or midpoint fallback. */
    if (!ex_finite(coordinate) || coordinate < (double)c->raw_command_min-0.5 ||
        coordinate > (double)c->raw_command_max+0.5) { return EX_RANGE; }
    const uint32_t rounded = (uint32_t)(coordinate + 0.5);
    if (rounded < c->raw_command_min || rounded > c->raw_command_max || rounded > UINT16_MAX) { return EX_RANGE; }
    /* Quantization must not silently cross a calibrated joint limit either. */
    double best_error=DBL_MAX; uint16_t best=0; bool found=false;
    for(int offset=-1;offset<=1;++offset) {
        const int32_t candidate=(int32_t)rounded+offset;
        if(candidate>=c->raw_command_min && candidate<=c->raw_command_max) {
            const double quantized_q=c->radians_at_raw_zero+c->radians_per_tick*(double)candidate;
            const double error=ex_abs(quantized_q-q);
            if(ex_finite(quantized_q) && quantized_q>=c->soft_min_rad && quantized_q<=c->soft_max_rad &&
               error<best_error && error<=1.01*ex_abs(c->radians_per_tick)) {
                best=(uint16_t)candidate; best_error=error; found=true;
            }
        }
    }
    if(!found) { return EX_SOFT_LIMIT; }
    /* At a boundary use the nearest representable tick INSIDE both limits.
     * The measured position is never clipped, and no physical limit is widened. */
    *raw = best;
    return EX_OK;
}
ex_result_t ex_arm_targets_to_raw(const ex_joint_calibration_t c[EX_JOINTS],
                                const double q[EX_JOINTS], uint16_t raw[EX_JOINTS]) {
    if (!c || !q || !raw) { return EX_ARGUMENT; }
    uint16_t temporary[EX_JOINTS];
    for (size_t i = 0; i < EX_JOINTS; ++i) {
        const ex_result_t r = ex_joint_to_raw(&c[i], q[i], &temporary[i]);
        if (r != EX_OK) { return r; }
    }
    for (size_t i = 0; i < EX_JOINTS; ++i) { raw[i] = temporary[i]; }
    return EX_OK;
}
ex_result_t ex_servo_make_read_request(uint8_t id, uint8_t out[8]) {
    if (!out || id < 1u || id > EX_JOINTS) { return EX_ARGUMENT; }
    uint8_t f[8] = {0xffu, 0xffu, id, 4u, 2u, 0x38u, 2u, 0u};
    f[7] = checksum(f, 7u);
    for (size_t i = 0; i < 8u; ++i) { out[i] = f[i]; }
    return EX_OK;
}
ex_result_t ex_servo_make_sync(const uint16_t raw[EX_JOINTS],
                              const ex_joint_calibration_t c[EX_JOINTS],
                              uint16_t runtime, uint16_t min_runtime,
                              uint16_t max_runtime, uint8_t out[EX_SYNC_BYTES]) {
    if (!raw || !c || !out) { return EX_ARGUMENT; }
    if (max_runtime < min_runtime || runtime < min_runtime || runtime > max_runtime) { return EX_RANGE; }
    for (size_t i = 0; i < EX_JOINTS; ++i) {
        if (!calibration_valid(&c[i])) { return EX_UNCALIBRATED; }
        if (raw[i] < c[i].raw_command_min || raw[i] > c[i].raw_command_max) { return EX_RANGE; }
        const double q = c[i].radians_at_raw_zero + c[i].radians_per_tick*(double)raw[i];
        if (!ex_finite(q) || q < c[i].soft_min_rad || q > c[i].soft_max_rad) { return EX_SOFT_LIMIT; }
    }
    uint8_t f[EX_SYNC_BYTES] = {0xffu, 0xffu, 0xfeu, 0x22u, 0x83u, 0x2au, 4u};
    for (size_t i = 0; i < EX_JOINTS; ++i) {
        const size_t offset = 7u + 5u*i;
        f[offset] = (uint8_t)(i+1u);
        f[offset+1u] = (uint8_t)(raw[i] >> 8u);
        f[offset+2u] = (uint8_t)(raw[i] & 255u);
        f[offset+3u] = (uint8_t)(runtime >> 8u);
        f[offset+4u] = (uint8_t)(runtime & 255u);
    }
    f[37] = checksum(f, 37u);
    for (size_t i = 0; i < EX_SYNC_BYTES; ++i) { out[i] = f[i]; }
    return EX_OK;
}
