#ifndef EXPLORER_UART_TEST_BOARD_H
#define EXPLORER_UART_TEST_BOARD_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
enum board_port { PORT_HOST,PORT_ARM,PORT_LIDAR0,PORT_LIDAR1,PORT_DEBUG,PORT_COUNT };
typedef enum { HAL_OK,HAL_ERROR,HAL_BUSY } HAL_StatusTypeDef;
enum { HAL_UART_STATE_READY, HAL_UART_STATE_BUSY };
typedef struct { unsigned gState,RxState; struct { uint32_t BaudRate; } Init; } UART_HandleTypeDef;
extern UART_HandleTypeDef board_uart[PORT_COUNT];
extern volatile uint32_t board_rx_overruns[PORT_COUNT],board_uart_errors[PORT_COUNT],board_tx_completed[PORT_COUNT];
extern volatile uint64_t board_tx_completed_us[PORT_COUNT];
extern uint32_t mock_primask;
static inline uint32_t __get_PRIMASK(void) { return mock_primask; }
static inline void __disable_irq(void) { mock_primask=1; }
static inline void __set_PRIMASK(uint32_t v) { mock_primask=v; }
static inline void __DMB(void) {}
#define __HAL_UART_CLEAR_OREFLAG(u) ((void)(u))
uint64_t board_time_us(void);
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *,uint8_t *,uint16_t);
HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *,uint8_t *,uint16_t);
HAL_StatusTypeDef HAL_UART_AbortTransmit(UART_HandleTypeDef *);
HAL_StatusTypeDef board_uart_receive_start(unsigned);
bool board_receive(unsigned,uint8_t *);
bool board_send(unsigned,const uint8_t *,uint16_t);
void board_io_poll(void);
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *);
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *);
void HAL_UART_ErrorCallback(UART_HandleTypeDef *);
#endif
