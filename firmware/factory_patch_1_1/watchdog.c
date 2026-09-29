/* Factory-image candidate: NOT installed or physically accepted.
 * Reuses ONLY the original battery cadence word, with disjoint bit fields.
 * All old code accessing this word is replaced by tick(); no new RAM address.
 */
#include <stdint.h>
#include <stddef.h>
#define BATTERY_MASK 0xffu
#define AGE_MASK 0xff00u
#define ARMED 0x10000u
#define EXPIRE_TICKS 26u
#ifdef HOST_TEST
extern volatile uint32_t test_state;
extern volatile uint8_t test_app_running;
extern unsigned irq_lock(void);
extern void irq_restore(unsigned);
extern void motion_stop(uint8_t);
extern void motion_command(float, float, float);
extern int battery_check(void);
#define STATE test_state
#define RUNNING test_app_running
#else
#define STATE (*(volatile uint32_t *)0x24001a64u)
#define RUNNING (*(volatile uint8_t *)0x24001a5cu)
static unsigned irq_lock(void) {
    unsigned old;
    __asm volatile("mrs %0, primask\ncpsid i" : "=r"(old) :: "memory");
    return old;
}
static void irq_restore(unsigned old) {
    __asm volatile("msr primask, %0" :: "r"(old) : "memory");
}
static void motion_stop(uint8_t brake) { ((void (*)(uint8_t))0x080043b1u)(brake); }
static void motion_command(float x, float y, float z) {
    ((void (*)(float,float,float))0x080049fdu)(x,y,z);
}
static int battery_check(void) { return ((int (*)(void))0x080019a9u)(); }
#endif

/* The geometry_msgs/Twist ABI in this exact image: six IEEE754 doubles. */
typedef struct { double lx,ly,lz,ax,ay,az; } Twist;
_Static_assert(sizeof(Twist)==48, "Twist ABI");
_Static_assert(offsetof(Twist,az)==40, "Twist angular.z ABI");

static int representable(double v) {
    /* Bound only to finite float conversion. Original factory speed clamps
       remain in Motion_Ctrl_Car, before conversion to integer wheel speeds. */
    return v <= 3.4028234663852886e38 && v >= -3.4028234663852886e38;
}

void factory_twist(const Twist *m) {
    unsigned key=irq_lock();
    uint32_t state=STATE;
    int valid=m && representable(m->lx) && representable(m->ly) && representable(m->az);
    if (!valid || !RUNNING) {
        STATE=(state & BATTERY_MASK) | (EXPIRE_TICKS << 8);
        motion_stop(1);
    } else {
        int zero=m->lx==0.0 && m->ly==0.0 && m->az==0.0;
        /* On boot/expiry a nonzero delayed packet cannot restart the motors.
           A zero command is required to re-arm. Twist has no source timestamp:
           this is not proof that any later packet is fresh at the publisher. */
        if (zero || (state & ARMED)) {
            STATE=(state & BATTERY_MASK) | ARMED;
            motion_command((float)m->lx,(float)m->ly,(float)m->az);
        }
    }
    irq_restore(key);
}

void factory_tick(void) {
    unsigned key=irq_lock();
    uint32_t state=STATE;
    unsigned battery=(state & BATTERY_MASK)+1u;
    unsigned age=(state & AGE_MASK)>>8;
    unsigned due=battery>10u;
    if (due) battery=0;
    if (age<EXPIRE_TICKS) age++;
    unsigned armed=state & ARMED;
    if (age>=EXPIRE_TICKS) armed=0;
    STATE=battery | (age<<8) | armed;
    if (!armed) motion_stop(1);
    irq_restore(key);
    /* ADC/battery work is outside the critical section, at its original
       11 x 10 ms cadence, independent of command traffic. */
    if (due && battery_check()) {
        key=irq_lock();
        RUNNING=0;
        STATE &= ~ARMED;
        motion_stop(1);
        irq_restore(key);
    }
}
