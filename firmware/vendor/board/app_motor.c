#include "app_motor.h"
#include "tim.h"
#include "app.h"


#define PWM_M1_A  TIM1->CCR3
#define PWM_M1_B  TIM1->CCR4

#define PWM_M2_A  TIM1->CCR1
#define PWM_M2_B  TIM1->CCR2

#define PWM_M3_A  TIM8->CCR1
#define PWM_M3_B  TIM8->CCR2

#define PWM_M4_A  TIM8->CCR3
#define PWM_M4_B  TIM8->CCR4


// Ignore PWM dead band  忽略PWM信号死区
static int16_t Motor_Ignore_Dead_Zone(int16_t pulse)
{
    if (pulse > 0) return pulse + MOTOR_IGNORE_PULSE;
    if (pulse < 0) return pulse - MOTOR_IGNORE_PULSE;
    return 0;
}

static int16_t Motor_Limit_Pulse(int16_t pulse)
{
    if (pulse >= MOTOR_MAX_PULSE) return MOTOR_MAX_PULSE;
    if (pulse <= -MOTOR_MAX_PULSE) return -MOTOR_MAX_PULSE;
    return pulse;
}

static uint16_t Motor_ABS(int16_t value)
{
    if (value < 0) return -value;
    return value;
}

// The PWM port of the motor is initialized  电机PWM口初始化
void Motor_Init(void)
{
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim1, TIM_CHANNEL_4);
    
    HAL_TIMEx_PWMN_Start(&htim8, TIM_CHANNEL_1);
    HAL_TIMEx_PWMN_Start(&htim8, TIM_CHANNEL_2);
    HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_3);
    HAL_TIM_PWM_Start(&htim8, TIM_CHANNEL_4);
}

// All motors stopped  所有电机停止
void Motor_Stop(uint8_t brake)
{
    if (brake != 0) brake = 1;
    PWM_M1_A = brake * MOTOR_MAX_PULSE;
    PWM_M1_B = brake * MOTOR_MAX_PULSE;
    PWM_M2_A = brake * MOTOR_MAX_PULSE;
    PWM_M2_B = brake * MOTOR_MAX_PULSE;
    PWM_M3_A = brake * MOTOR_MAX_PULSE;
    PWM_M3_B = brake * MOTOR_MAX_PULSE;
    PWM_M4_A = brake * MOTOR_MAX_PULSE;
    PWM_M4_B = brake * MOTOR_MAX_PULSE;
}

// 设置电机速度，speed:±MOTOR_MAX_SPEED, 0为停止
// Set motor speed, speed:±MOTOR_MAX_SPEED, 0 indicates stop
void Motor_Set_Pwm(uint8_t id, int16_t speed)
{
    int16_t pulse = Motor_Ignore_Dead_Zone(speed);
    // Limit input  限制输入
    pulse = Motor_Limit_Pulse(pulse);

    switch (id)
    {
    case MOTOR_ID_M1:
    {
        pulse = -pulse;
        if (pulse >= 0)
        {
            PWM_M1_A = pulse;
            PWM_M1_B = 0;
        }
        else
        {
            PWM_M1_A = 0;
            PWM_M1_B = -pulse;
        }
        break;
    }
    case MOTOR_ID_M2:
    {
        pulse = -pulse;
        if (pulse >= 0)
        {
            PWM_M2_A = pulse;
            PWM_M2_B = 0;
        }
        else
        {
            PWM_M2_A = 0;
            PWM_M2_B = -pulse;
        }
        break;
    }
    case MOTOR_ID_M3:
    {
        if (pulse >= 0)
        {
            PWM_M3_A = pulse;
            PWM_M3_B = 0;
        }
        else
        {
            PWM_M3_A = 0;
            PWM_M3_B = -pulse;
        }
        break;
    }
    case MOTOR_ID_M4:
    {
        if (pulse >= 0)
        {
            PWM_M4_A = pulse;
            PWM_M4_B = 0;
        }
        else
        {
            PWM_M4_A = 0;
            PWM_M4_B = -pulse;
        }
        break;
    }
    case MOTOR_ID_MAX:
    {
        if (pulse >= 0)
        {
            PWM_M1_A = 0;
            PWM_M1_B = Motor_ABS(pulse);
            PWM_M2_A = 0;
            PWM_M2_B = Motor_ABS(pulse);
            PWM_M3_A = Motor_ABS(pulse);
            PWM_M3_B = 0;
            PWM_M4_A = Motor_ABS(pulse);
            PWM_M4_B = 0;
        }
        else
        {
            PWM_M1_A = Motor_ABS(pulse);
            PWM_M1_B = 0;
            PWM_M2_A = Motor_ABS(pulse);
            PWM_M2_B = 0;
            PWM_M3_A = 0;
            PWM_M3_B = Motor_ABS(pulse);
            PWM_M4_A = 0;
            PWM_M4_B = Motor_ABS(pulse);
        }
        break;
    }

    default:
        break;
    }
}
