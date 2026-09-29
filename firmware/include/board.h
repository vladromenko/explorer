#ifndef EXPLORER_BOARD_H
#define EXPLORER_BOARD_H
#include "main.h"
#include "explorer_controller.h"

enum board_port { PORT_HOST, PORT_ARM, PORT_LIDAR0, PORT_LIDAR1, PORT_DEBUG, PORT_COUNT };
extern UART_HandleTypeDef board_uart[PORT_COUNT];
extern volatile uint32_t board_rx_overruns[PORT_COUNT];
extern volatile uint32_t board_uart_errors[PORT_COUNT];
extern volatile uint32_t board_tx_completed[PORT_COUNT];
/* Read timestamp and generation together with interrupts masked on Cortex-M7. */
extern volatile uint64_t board_tx_completed_us[PORT_COUNT];
void board_io_poll(void);
HAL_StatusTypeDef board_uart_receive_start(unsigned port);
void board_init(void);
uint64_t board_time_us(void);
uint64_t board_boot_nonce(void);
bool board_receive(unsigned port,uint8_t *value);
bool board_send(unsigned port,const uint8_t *data,uint16_t length);
void board_watchdog_feed(void);
void board_start_control(void);
void board_control_interrupt(void);
bool board_rgb(uint8_t r,uint8_t g,uint8_t b);
void board_beep(bool enabled);
void board_sensor_init(void);
void board_sensor_poll(void);
bool board_emit(uint8_t type,const uint8_t *payload,size_t length);
#endif
