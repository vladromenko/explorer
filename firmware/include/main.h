#ifndef EXPLORER_MAIN_H
#define EXPLORER_MAIN_H
#include "stm32h7xx_hal.h"
#define BAT_Pin GPIO_PIN_0
#define BAT_GPIO_Port GPIOC
#define SPI2_NSS_Pin GPIO_PIN_12
#define SPI2_NSS_GPIO_Port GPIOB
void Error_Handler(void);
void SystemClock_Config(void);
#endif
