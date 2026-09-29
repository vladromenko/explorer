#ifndef __APP_MOTOR_H__
#define __APP_MOTOR_H__

#include "main.h"

#define MOTOR_MAX_SPEED     (1000)
#define MOTOR_IGNORE_PULSE  MOTOR_MAX_SPEED
#define MOTOR_MAX_PULSE     (2000)




typedef enum {
    MOTOR_ID_M1 = 0,
    MOTOR_ID_M2,
    MOTOR_ID_M3,
    MOTOR_ID_M4,
    MOTOR_ID_MAX
} Motor_ID;


typedef enum {
    MOTOR_STOP = 0,
    MOTOR_BRAKE=1,
} Motor_Stop_Mode;


void Motor_Init(void);
void Motor_Set_Pwm(uint8_t id, int16_t speed);
void Motor_Stop(uint8_t brake);


#endif
