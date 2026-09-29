#ifndef EXPLORER_SENSOR_MOCK_BOARD_H
#define EXPLORER_SENSOR_MOCK_BOARD_H
#include "explorer_wire.h"
enum board_port { PORT_HOST,PORT_ARM,PORT_LIDAR0,PORT_LIDAR1,PORT_DEBUG,PORT_COUNT };
extern volatile uint32_t board_rx_overruns[PORT_COUNT],board_uart_errors[PORT_COUNT];
uint64_t board_time_us(void);
bool board_receive(unsigned port,uint8_t *value);
bool board_send(unsigned port,const uint8_t *data,uint16_t length);
bool board_emit(uint8_t type,const uint8_t *payload,size_t length);
void board_sensor_init(void);
void board_sensor_poll(void);
#endif
