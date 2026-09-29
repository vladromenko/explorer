#ifndef EX_INTERNAL_H
#define EX_INTERNAL_H
#include <float.h>
#include "explorer_core.h"
/* Do not compile this control code with -ffast-math / -ffinite-math-only. */
static inline bool ex_finite(double x) { return x >= -DBL_MAX && x <= DBL_MAX; }
static inline double ex_abs(double x) { return x < 0.0 ? -x : x; }
static inline double ex_clamp(double x, double lo, double hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}
static inline bool ex_twist_finite(ex_twist_t x) {
    return ex_finite(x.x) && ex_finite(x.y) && ex_finite(x.yaw);
}
static inline ex_twist_t ex_zero(void) { return (ex_twist_t){0.0, 0.0, 0.0}; }
#endif
