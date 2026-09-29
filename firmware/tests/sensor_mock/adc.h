#ifndef EXPLORER_SENSOR_MOCK_ADC_H
#define EXPLORER_SENSOR_MOCK_ADC_H
#include "spi.h"
typedef struct { unsigned dummy; } ADC_TypeDef;
typedef struct { ADC_TypeDef *Instance; uint32_t State; } ADC_HandleTypeDef;
#define HAL_ADC_STATE_REG_BUSY 1u
#define HAL_ADC_STATE_INJ_BUSY 2u
#define HAL_ADC_STATE_BUSY_INTERNAL 4u
#define HAL_ADC_STATE_ERROR_INTERNAL 8u
#define HAL_ADC_STATE_READY 16u
extern ADC_HandleTypeDef hadc1;
#define ADC_CALIB_OFFSET_LINEARITY 0u
#define ADC_SINGLE_ENDED 0u

HAL_StatusTypeDef HAL_ADC_Start(ADC_HandleTypeDef *h);
HAL_StatusTypeDef HAL_ADC_PollForConversion(ADC_HandleTypeDef *h,uint32_t timeout);
uint32_t HAL_ADC_GetValue(ADC_HandleTypeDef *h);
HAL_StatusTypeDef HAL_ADC_Stop(ADC_HandleTypeDef *h);
#endif
