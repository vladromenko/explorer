#ifndef EXPLORER_MOCK_BOARD_H
#define EXPLORER_MOCK_BOARD_H
#include "explorer_controller.h"
enum board_port { PORT_HOST,PORT_ARM,PORT_LIDAR0,PORT_LIDAR1,PORT_DEBUG,PORT_COUNT };
typedef struct { unsigned gState; } UART_HandleTypeDef;
#define HAL_UART_STATE_READY 0u
extern UART_HandleTypeDef board_uart[PORT_COUNT];
extern volatile uint32_t board_rx_overruns[PORT_COUNT],board_uart_errors[PORT_COUNT];
extern volatile uint32_t board_tx_completed[PORT_COUNT];
extern volatile uint64_t board_tx_completed_us[PORT_COUNT];
static inline uint32_t __get_PRIMASK(void) { return 0; }
static inline void __disable_irq(void) {}
void mock_restore_irq(uint32_t value);
static inline void __set_PRIMASK(uint32_t v) { mock_restore_irq(v); }
uint64_t board_time_us(void);
bool board_receive(unsigned port,uint8_t *value);
bool board_send(unsigned port,const uint8_t *data,uint16_t length);
bool board_emit(uint8_t type,const uint8_t *payload,size_t length);
#endif
