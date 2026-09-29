#ifndef __SPI_H__
#define __SPI_H__
#include <stdint.h>
typedef enum { HAL_OK,HAL_ERROR,HAL_BUSY,HAL_TIMEOUT } HAL_StatusTypeDef;
typedef struct { unsigned dummy; } SPI_HandleTypeDef;
extern SPI_HandleTypeDef hspi2;
#define GPIOB ((void*)1)
#define GPIO_PIN_12 4096u
#define GPIO_PIN_RESET 0u
#define GPIO_PIN_SET 1u
void HAL_GPIO_WritePin(void *port,uint16_t pin,unsigned state);
HAL_StatusTypeDef HAL_SPI_TransmitReceive(SPI_HandleTypeDef *h,uint8_t *tx,uint8_t *rx,uint16_t n,uint32_t timeout);
void HAL_Delay(uint32_t ms);
#endif
