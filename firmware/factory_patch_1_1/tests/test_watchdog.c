#include <assert.h>
#include <math.h>
#include <stdio.h>
#define HOST_TEST
#include "../watchdog.c"
volatile uint32_t test_state;
volatile uint8_t test_app_running;
static unsigned irq,stops,commands,battery_calls;
static int battery_low;
unsigned irq_lock(void) { unsigned old=irq;irq=1;return old; }
void irq_restore(unsigned old) { irq=old; }
void motion_stop(uint8_t b) { assert(irq==1);assert(b==1);stops++; }
void motion_command(float x,float y,float z) { assert(irq==1);assert(isfinite(x+y+z));commands++; }
int battery_check(void) { assert(irq==0);battery_calls++;return battery_low; }
static void reset(void) { test_state=0;test_app_running=1;irq=stops=commands=battery_calls=0;battery_low=0; }
int main(void) {
 Twist zero={0}, forward={.lx=.04}, invalid={.lx=NAN};
 reset();factory_twist(&forward);assert(commands==0); /* no boot motion */
 factory_twist(&zero);factory_twist(&forward);assert(commands==2);
 for (unsigned i=0;i<25;i++)factory_tick();
 assert(stops==0);factory_tick();assert(stops==1);assert(!(test_state&ARMED));
 factory_twist(&forward);assert(commands==2); /* expired stale nonzero ignored */
 factory_twist(&zero);factory_twist(&forward);assert(commands==4);
 reset();factory_twist(&zero);
 for(unsigned i=0;i<1100;i++) { factory_twist(&forward);factory_tick(); }
 assert(battery_calls==100);assert(stops==0); /* traffic never starves battery */
 factory_twist(&invalid);assert(stops==1);assert(!(test_state&ARMED));
 invalid.lx=INFINITY;factory_twist(&invalid);assert(stops==2);
 reset();factory_twist(&zero);battery_low=1;
 for(unsigned i=0;i<11;i++)factory_tick();
 assert(!test_app_running);assert(!(test_state&ARMED));
 unsigned previous=commands;factory_twist(&forward);factory_twist(&zero);assert(commands==previous);
 reset();irq=1;factory_twist(&zero);assert(irq==1);irq=0;
 for(unsigned i=0;i<100000;i++)factory_tick();
 assert(((test_state&AGE_MASK)>>8)==EXPIRE_TICKS);assert(battery_calls==100000/11);
 assert(!irq);puts("PASS: independent battery cadence, boot zero, expiry, stale nonzero, invalid input, low battery, PRIMASK, saturation");
}
