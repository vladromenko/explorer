#include "internal.h"

/* Peak factors for s(u)=10*u^3-15*u^4+6*u^5 on [0,1]:
 * max |s'|=15/8; max |s''|=10/sqrt(3); max |s'''|=60.
 * Profile timing uses bisection, avoiding a dependency on libm. */
static bool duration_fits(const double start[EX_JOINTS], const double goal[EX_JOINTS],
                          const ex_profile_limits_t lim[EX_JOINTS], double t) {
    for (size_t i = 0; i < EX_JOINTS; ++i) {
        const double d = ex_abs(goal[i]-start[i]);
        if (1.875*d/t > lim[i].velocity_max ||
            5.773502691896258*d/t/t > lim[i].acceleration_max ||
            60.0*d/t/t/t > lim[i].jerk_max) { return false; }
    }
    return true;
}
ex_result_t ex_profile_plan(const double start[EX_JOINTS], const double goal[EX_JOINTS],
                            const ex_profile_limits_t lim[EX_JOINTS],
                            double requested, ex_profile_t *out) {
    if (!start || !goal || !lim || !out) { return EX_ARGUMENT; }
    if (!ex_finite(requested) || requested < 0.0 || requested > 3600.0) { return EX_RANGE; }
    for (size_t i = 0; i < EX_JOINTS; ++i) {
        if (!ex_finite(start[i]) || !ex_finite(goal[i]) ||
            !ex_finite(lim[i].position_min) || !ex_finite(lim[i].position_max) ||
            lim[i].position_min >= lim[i].position_max ||
            !ex_finite(lim[i].velocity_max) || lim[i].velocity_max <= 0.0 ||
            !ex_finite(lim[i].acceleration_max) || lim[i].acceleration_max <= 0.0 ||
            !ex_finite(lim[i].jerk_max) || lim[i].jerk_max <= 0.0) { return EX_VALUE; }
        if (start[i] < lim[i].position_min || start[i] > lim[i].position_max ||
            goal[i] < lim[i].position_min || goal[i] > lim[i].position_max) { return EX_SOFT_LIMIT; }
        if (!ex_finite(goal[i]-start[i])) { return EX_VALUE; }
    }
    double low = 0.001, high = 0.001; /* Software profile resolution, not servo rate. */
    while (!duration_fits(start, goal, lim, high) && high < 3600.0) {
        high *= 2.0;
        if (high > 3600.0) { high = 3600.0; }
    }
    if (!duration_fits(start, goal, lim, high)) { return EX_RANGE; }
    for (unsigned int k = 0; k < 48u; ++k) {
        const double mid = (low+high)*0.5;
        if (duration_fits(start, goal, lim, mid)) { high = mid; }
        else { low = mid; }
    }
    if (requested > high) { high = requested; }
    ex_profile_t p = {.duration_s=high, .valid=true};
    for (size_t i = 0; i < EX_JOINTS; ++i) { p.start[i] = start[i]; p.goal[i] = goal[i]; }
    *out = p;
    return EX_OK;
}
ex_result_t ex_profile_sample(const ex_profile_t *p, double elapsed, ex_profile_sample_t *out) {
    if (!p || !out) { return EX_ARGUMENT; }
    if (!p->valid || !ex_finite(p->duration_s) || p->duration_s < 0.001 || p->duration_s > 3600.0) { return EX_NOT_READY; }
    if (!ex_finite(elapsed) || elapsed < 0.0) { return EX_TIME; }
    ex_profile_sample_t sample = {0};
    sample.finished = elapsed >= p->duration_s;
    const double u = sample.finished ? 1.0 : elapsed/p->duration_s;
    const double s = u*u*u*(10.0+u*(-15.0+6.0*u));
    const double ds = 30.0*u*u*(1.0-u)*(1.0-u);
    const double dds = 60.0*u*(1.0-u)*(1.0-2.0*u);
    const double ddds = 60.0-360.0*u+360.0*u*u;
    for (size_t i = 0; i < EX_JOINTS; ++i) {
        if (!ex_finite(p->start[i]) || !ex_finite(p->goal[i])) { return EX_VALUE; }
        const double d = p->goal[i]-p->start[i];
        if (!ex_finite(d)) { return EX_VALUE; }
        sample.position[i] = sample.finished ? p->goal[i] : p->start[i]+d*s;
        if (!sample.finished) {
            sample.velocity[i] = d*ds/p->duration_s;
            sample.acceleration[i] = d*dds/p->duration_s/p->duration_s;
            sample.jerk[i] = d*ddds/p->duration_s/p->duration_s/p->duration_s;
        }
        if (!ex_finite(sample.position[i]) || !ex_finite(sample.velocity[i]) ||
            !ex_finite(sample.acceleration[i]) || !ex_finite(sample.jerk[i])) { return EX_VALUE; }
    }
    *out = sample;
    return EX_OK;
}
