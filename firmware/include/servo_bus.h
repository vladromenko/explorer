#ifndef EXPLORER_SERVO_BUS_H
#define EXPLORER_SERVO_BUS_H
#include <board.h>
typedef struct { uint16_t raw; uint64_t time; ex_result_t error; uint8_t device_error; bool valid; } servo_measurement_t;
extern servo_measurement_t servo_measurements[6];
extern volatile uint32_t servo_measure_generation;
extern uint64_t servo_sent_generation;
extern volatile uint32_t servo_cancel_completed_generation;
void servo_bus_poll(const ec_controller_t *snapshot);
#endif
