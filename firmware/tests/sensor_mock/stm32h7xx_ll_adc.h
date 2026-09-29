#ifndef EXPLORER_SENSOR_MOCK_LL_ADC_H
#define EXPLORER_SENSOR_MOCK_LL_ADC_H
#include "adc.h"
uint32_t LL_ADC_IsEnabled(ADC_TypeDef *instance);
uint32_t LL_ADC_IsCalibrationOnGoing(ADC_TypeDef *instance);
void LL_ADC_StartCalibration(ADC_TypeDef *instance,uint32_t mode,uint32_t ended);
#endif
