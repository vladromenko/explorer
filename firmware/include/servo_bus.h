#ifndef EXPLORER_SERVO_BUS_H
#define EXPLORER_SERVO_BUS_H
#include <board.h>
typedef struct { uint16_t raw; uint64_t time; ex_result_t error; uint8_t device_error; bool valid; } servo_measurement_t;
extern servo_measurement_t servo_measurements[6];
extern volatile uint32_t servo_measure_generation;
extern uint64_t servo_sent_generation;
extern volatile uint32_t servo_cancel_completed_generation;
extern volatile uint32_t servo_write_fault_generation;
extern ex_result_t servo_last_write_error;
#define SERVO_DIAGNOSTIC_BYTES 100u
void servo_bus_diagnostics(const ec_controller_t *snapshot,uint8_t out[SERVO_DIAGNOSTIC_BYTES]);
/* Baud is confirmed by USART3 configuration; processing quiet time still needs
 * physical acceptance. Ten UART bits include start and stop for each byte. */
#define SERVO_BAUD 115200u
#define SERVO_WRITE_QUIET_US UINT64_C(5000)
static inline uint64_t servo_write_budget_us(uint16_t length) {
    return ((uint64_t)length*UINT64_C(10000000)+SERVO_BAUD-1u)/SERVO_BAUD+SERVO_WRITE_QUIET_US;
}
/* Initialize once before control interrupts; no UART or actuator commands. */
void servo_bus_init(void);
/* Main owns live controller state and performs the final atomic validation. */
bool servo_bus_commit(const ec_controller_t *snapshot,const uint8_t *frame,uint16_t length,bool cancel);
void servo_bus_poll(const ec_controller_t *snapshot);
#endif
