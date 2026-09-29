#include "board.h"
#include "tim.h"
#include "app_motor.h"
#include "spi.h"
#include "adc.h"
#include <string.h>

/* Pins/AF/clock selections are those of the supplied Yahboom examples.
 * UART5: Publisher_lidar, UART1/3/7: Subscriber_uart_servo.
 * UART byte reception uses hardware FIFO + IRQ and CPU-only DTCM buffers.
 * Only RGB uses DMA, from a separate DMA-accessible D2 SRAM buffer. D-cache
 * remains disabled; no DMA operation ever references DTCM.
 */
static TIM_HandleTypeDef tick_timer;
static IWDG_HandleTypeDef watchdog;
static SPI_HandleTypeDef rgb_spi;
static DMA_HandleTypeDef rgb_dma;
static uint8_t rgb_buffer[8*12+100] __attribute__((section(".dma_rgb"),aligned(32)));
static volatile uint32_t time_high;
static uint32_t boot_errors;

static void check(HAL_StatusTypeDef status) { if(status!=HAL_OK) { Error_Handler(); } }
static void pin(GPIO_TypeDef *port,uint16_t pins,uint32_t mode,uint32_t alternate) {
    GPIO_InitTypeDef c={0}; c.Pin=pins; c.Mode=mode;
    c.Pull=GPIO_NOPULL; c.Speed=GPIO_SPEED_FREQ_LOW; c.Alternate=alternate;
    HAL_GPIO_Init(port,&c);
}
static void uart_init(unsigned index,USART_TypeDef *instance,uint32_t baud,IRQn_Type irq) {
    UART_HandleTypeDef *u=&board_uart[index]; u->Instance=instance;
    u->Init.BaudRate=baud; u->Init.WordLength=UART_WORDLENGTH_8B;
    u->Init.StopBits=UART_STOPBITS_1; u->Init.Parity=UART_PARITY_NONE;
    u->Init.Mode=UART_MODE_TX_RX; u->Init.HwFlowCtl=UART_HWCONTROL_NONE;
    u->Init.OverSampling=UART_OVERSAMPLING_16;
    u->Init.ClockPrescaler=UART_PRESCALER_DIV1;
    check(HAL_UART_Init(u));
    /* Keep hardware RX buffering while the higher-priority motor tick runs. */
    check(HAL_UARTEx_SetRxFifoThreshold(u,UART_RXFIFO_THRESHOLD_1_8));
    check(HAL_UARTEx_SetTxFifoThreshold(u,UART_TXFIFO_THRESHOLD_1_8));
    check(HAL_UARTEx_EnableFifoMode(u));
    HAL_NVIC_SetPriority(irq,4,0); HAL_NVIC_EnableIRQ(irq);
    check(board_uart_receive_start(index));
}
void HAL_UART_MspInit(UART_HandleTypeDef *u) { (void)u; /* Configured together below. */ }
void HAL_MspInit(void) {
    __HAL_RCC_SYSCFG_CLK_ENABLE(); HAL_NVIC_SetPriorityGrouping(NVIC_PRIORITYGROUP_4);
}
static void rgb_init(void) {
    __HAL_RCC_SPI4_CLK_ENABLE();
    pin(GPIOE,GPIO_PIN_2|GPIO_PIN_6,GPIO_MODE_AF_PP,GPIO_AF5_SPI4);
    rgb_spi.Instance=SPI4;
    rgb_spi.Init.Mode=SPI_MODE_MASTER; rgb_spi.Init.Direction=SPI_DIRECTION_2LINES_TXONLY;
    rgb_spi.Init.DataSize=SPI_DATASIZE_8BIT; rgb_spi.Init.CLKPolarity=SPI_POLARITY_LOW;
    rgb_spi.Init.CLKPhase=SPI_PHASE_1EDGE; rgb_spi.Init.NSS=SPI_NSS_SOFT;
    rgb_spi.Init.BaudRatePrescaler=SPI_BAUDRATEPRESCALER_32;
    rgb_spi.Init.FirstBit=SPI_FIRSTBIT_MSB; rgb_spi.Init.NSSPMode=SPI_NSS_PULSE_ENABLE;
    rgb_spi.Init.NSSPolarity=SPI_NSS_POLARITY_LOW;
    rgb_spi.Init.FifoThreshold=SPI_FIFO_THRESHOLD_01DATA;
    check(HAL_SPI_Init(&rgb_spi));
    __HAL_RCC_D2SRAM1_CLK_ENABLE(); __HAL_RCC_DMA2_CLK_ENABLE();
    rgb_dma.Instance=DMA2_Stream0;
    rgb_dma.Init.Request=DMA_REQUEST_SPI4_TX;
    rgb_dma.Init.Direction=DMA_MEMORY_TO_PERIPH;
    rgb_dma.Init.PeriphInc=DMA_PINC_DISABLE; rgb_dma.Init.MemInc=DMA_MINC_ENABLE;
    rgb_dma.Init.PeriphDataAlignment=DMA_PDATAALIGN_BYTE;
    rgb_dma.Init.MemDataAlignment=DMA_MDATAALIGN_BYTE;
    rgb_dma.Init.Mode=DMA_NORMAL; rgb_dma.Init.Priority=DMA_PRIORITY_LOW;
    rgb_dma.Init.FIFOMode=DMA_FIFOMODE_DISABLE;
    check(HAL_DMA_Init(&rgb_dma)); __HAL_LINKDMA(&rgb_spi,hdmatx,rgb_dma);
    HAL_NVIC_SetPriority(DMA2_Stream0_IRQn,6,0); HAL_NVIC_EnableIRQ(DMA2_Stream0_IRQn);
    HAL_NVIC_SetPriority(SPI4_IRQn,6,0); HAL_NVIC_EnableIRQ(SPI4_IRQn);
}
bool board_rgb(uint8_t r,uint8_t g,uint8_t b) {
    /* Vendor RGB encoding: each WS2812 bit is SPI nibble E or 8 @3.75MHz.
     * The supplied app_rgb.h defines MAX_RGB=8 and RGB_RESET_WIDTH=100.
     */
    if(rgb_spi.State!=HAL_SPI_STATE_READY) { return false; }
    memset(rgb_buffer,0,sizeof(rgb_buffer)); const uint8_t colors[3]={g,r,b};
    for(unsigned led=0;led<8;++led) {
        for(unsigned c=0;c<3;++c) {
            for(unsigned pair=0;pair<4;++pair) {
                const unsigned shift=6-2*pair;
                rgb_buffer[led*12+c*4+pair]=(uint8_t)(((colors[c]&(2u<<shift))?0xe0:0x80)|
                                                           ((colors[c]&(1u<<shift))?0x0e:0x08));
            }
        }
    }
    __DMB();
    bool ok=HAL_SPI_Transmit_DMA(&rgb_spi,rgb_buffer,sizeof(rgb_buffer))==HAL_OK;
    if(!ok) { ++boot_errors; }
    return ok;
}
void board_beep(bool enabled) { HAL_GPIO_WritePin(GPIOE,GPIO_PIN_5,enabled?GPIO_PIN_SET:GPIO_PIN_RESET); }
void board_init(void) {
    SCB->VTOR=FLASH_BANK1_BASE; __DSB(); __ISB();
    /* The supplied 480 MHz clock setup targets revision V. Do not overclock an
     * earlier silicon revision if someone bypasses the installer identity check. */
    if(HAL_GetDEVID()!=0x450 || HAL_GetREVID()!=REV_ID_V) { Error_Handler(); }
    HAL_Init(); SystemClock_Config(); SCB_EnableICache();
    /* D-cache stays off. RGB DMA uses the dedicated D2 SRAM linker section. */
    __HAL_RCC_GPIOA_CLK_ENABLE(); __HAL_RCC_GPIOB_CLK_ENABLE();
    __HAL_RCC_GPIOC_CLK_ENABLE(); __HAL_RCC_GPIOD_CLK_ENABLE(); __HAL_RCC_GPIOE_CLK_ENABLE();
    HAL_GPIO_WritePin(GPIOE,GPIO_PIN_5,GPIO_PIN_RESET);
    pin(GPIOE,GPIO_PIN_5,GPIO_MODE_OUTPUT_PP,0);
    HAL_GPIO_WritePin(GPIOB,GPIO_PIN_12,GPIO_PIN_SET);
    pin(GPIOB,GPIO_PIN_12,GPIO_MODE_OUTPUT_PP,0);
    pin(GPIOC,GPIO_PIN_13|GPIO_PIN_14,GPIO_MODE_OUTPUT_PP,0);
    RCC_PeriphCLKInitTypeDef clock={0};
    clock.PeriphClockSelection=RCC_PERIPHCLK_USART1|RCC_PERIPHCLK_USART3|
        RCC_PERIPHCLK_UART4|RCC_PERIPHCLK_UART5|RCC_PERIPHCLK_UART7|RCC_PERIPHCLK_SPI4;
    clock.Usart16ClockSelection=RCC_USART16CLKSOURCE_D2PCLK2;
    clock.Usart234578ClockSelection=RCC_USART234578CLKSOURCE_D2PCLK1;
    clock.Spi45ClockSelection=RCC_SPI45CLKSOURCE_D2PCLK1;
    check(HAL_RCCEx_PeriphCLKConfig(&clock));
    __HAL_RCC_USART1_CLK_ENABLE(); __HAL_RCC_USART3_CLK_ENABLE();
    __HAL_RCC_UART4_CLK_ENABLE(); __HAL_RCC_UART5_CLK_ENABLE(); __HAL_RCC_UART7_CLK_ENABLE();
    pin(GPIOA,GPIO_PIN_9|GPIO_PIN_10,GPIO_MODE_AF_PP,GPIO_AF7_USART1);
    pin(GPIOD,GPIO_PIN_8|GPIO_PIN_9,GPIO_MODE_AF_PP,GPIO_AF7_USART3);
    pin(GPIOC,GPIO_PIN_10|GPIO_PIN_11,GPIO_MODE_AF_PP,GPIO_AF8_UART4);
    pin(GPIOC,GPIO_PIN_12,GPIO_MODE_AF_PP,GPIO_AF8_UART5);
    pin(GPIOD,GPIO_PIN_2,GPIO_MODE_AF_PP,GPIO_AF8_UART5);
    pin(GPIOE,GPIO_PIN_7|GPIO_PIN_8,GPIO_MODE_AF_PP,GPIO_AF7_UART7);
    uart_init(PORT_HOST,USART1,2000000,USART1_IRQn);
    uart_init(PORT_ARM,USART3,115200,USART3_IRQn);
    uart_init(PORT_LIDAR0,UART4,230400,UART4_IRQn);
    uart_init(PORT_LIDAR1,UART5,230400,UART5_IRQn);
    uart_init(PORT_DEBUG,UART7,115200,UART7_IRQn);
    MX_TIM1_Init(); MX_TIM8_Init();
    Motor_Stop(MOTOR_STOP);
    check(HAL_TIM_PWM_Start(&htim1,TIM_CHANNEL_1));
    check(HAL_TIM_PWM_Start(&htim1,TIM_CHANNEL_2));
    check(HAL_TIM_PWM_Start(&htim1,TIM_CHANNEL_3));
    check(HAL_TIM_PWM_Start(&htim1,TIM_CHANNEL_4));
    check(HAL_TIMEx_PWMN_Start(&htim8,TIM_CHANNEL_1));
    check(HAL_TIMEx_PWMN_Start(&htim8,TIM_CHANNEL_2));
    check(HAL_TIM_PWM_Start(&htim8,TIM_CHANNEL_3));
    check(HAL_TIM_PWM_Start(&htim8,TIM_CHANNEL_4));
    MX_TIM2_Init(); MX_TIM3_Init(); MX_TIM4_Init(); MX_TIM5_Init();
    check(HAL_TIM_Encoder_Start(&htim2,TIM_CHANNEL_ALL));
    check(HAL_TIM_Encoder_Start(&htim3,TIM_CHANNEL_ALL));
    check(HAL_TIM_Encoder_Start(&htim4,TIM_CHANNEL_ALL));
    check(HAL_TIM_Encoder_Start(&htim5,TIM_CHANNEL_ALL));
    MX_SPI2_Init(); MX_ADC1_Init();
    HAL_NVIC_SetPriority(SPI2_IRQn,6,0); HAL_NVIC_SetPriority(ADC_IRQn,6,0);
    rgb_init(); (void)board_rgb(0,0,0);
    __HAL_RCC_TIM7_CLK_ENABLE();
    /* TIM7 extends a 1MHz 16-bit hardware clock. No ROS epoch dependency. */
    TIM7->PSC=239; TIM7->ARR=65535; TIM7->EGR=TIM_EGR_UG; TIM7->SR=0;
    TIM7->DIER=TIM_DIER_UIE;
    HAL_NVIC_SetPriority(TIM7_IRQn,0,0); HAL_NVIC_EnableIRQ(TIM7_IRQn);
    TIM7->CR1=TIM_CR1_CEN;
}
uint64_t board_time_us(void) {
    uint32_t saved=__get_PRIMASK(); __disable_irq();
    uint32_t hi=time_high; uint32_t lo=TIM7->CNT;
    if(TIM7->SR&TIM_SR_UIF) { ++hi; lo=TIM7->CNT; }
    __set_PRIMASK(saved); return ((uint64_t)hi<<16)|lo;
}
uint64_t board_boot_nonce(void) {
    RCC_OscInitTypeDef osc={0}; osc.OscillatorType=RCC_OSCILLATORTYPE_HSI48;
    osc.HSI48State=RCC_HSI48_ON; osc.PLL.PLLState=RCC_PLL_NONE;
    check(HAL_RCC_OscConfig(&osc));
    RCC_PeriphCLKInitTypeDef clock={0}; clock.PeriphClockSelection=RCC_PERIPHCLK_RNG;
    clock.RngClockSelection=RCC_RNGCLKSOURCE_HSI48; check(HAL_RCCEx_PeriphCLKConfig(&clock));
    __HAL_RCC_RNG_CLK_ENABLE();
    RNG_HandleTypeDef rng={0}; rng.Instance=RNG; rng.Init.ClockErrorDetection=RNG_CED_ENABLE;
    check(HAL_RNG_Init(&rng)); uint32_t a=0,b=0;
    check(HAL_RNG_GenerateRandomNumber(&rng,&a)); check(HAL_RNG_GenerateRandomNumber(&rng,&b));
    const uint64_t nonce=((uint64_t)a<<32)|b;
    if(!nonce) { Error_Handler(); }
    return nonce;
}
void board_start_control(void) {
    watchdog.Instance=IWDG1; watchdog.Init.Prescaler=IWDG_PRESCALER_16;
    watchdog.Init.Reload=511; watchdog.Init.Window=IWDG_WINDOW_DISABLE;
    check(HAL_IWDG_Init(&watchdog)); /* nominal 256 ms; LSI tolerance applies */
    __HAL_RCC_TIM6_CLK_ENABLE(); tick_timer.Instance=TIM6;
    tick_timer.Init.Prescaler=239; tick_timer.Init.Period=9999;
    check(HAL_TIM_Base_Init(&tick_timer));
    HAL_NVIC_SetPriority(TIM6_DAC_IRQn,1,0); HAL_NVIC_EnableIRQ(TIM6_DAC_IRQn);
    check(HAL_TIM_Base_Start_IT(&tick_timer));
}
void board_watchdog_feed(void) { check(HAL_IWDG_Refresh(&watchdog)); }
void TIM7_IRQHandler(void) { if(TIM7->SR&TIM_SR_UIF) { TIM7->SR=0; ++time_high; } }
void TIM6_DAC_IRQHandler(void) {
    if(TIM6->SR&TIM_SR_UIF) { TIM6->SR=0; board_control_interrupt(); }
}
void USART1_IRQHandler(void) { HAL_UART_IRQHandler(&board_uart[PORT_HOST]); }
void USART3_IRQHandler(void) { HAL_UART_IRQHandler(&board_uart[PORT_ARM]); }
void UART4_IRQHandler(void) { HAL_UART_IRQHandler(&board_uart[PORT_LIDAR0]); }
void UART5_IRQHandler(void) { HAL_UART_IRQHandler(&board_uart[PORT_LIDAR1]); }
void UART7_IRQHandler(void) { HAL_UART_IRQHandler(&board_uart[PORT_DEBUG]); }
void SPI2_IRQHandler(void) { HAL_SPI_IRQHandler(&hspi2); }
void SPI4_IRQHandler(void) { HAL_SPI_IRQHandler(&rgb_spi); }
void DMA2_Stream0_IRQHandler(void) { HAL_DMA_IRQHandler(&rgb_dma); }
void ADC_IRQHandler(void) { HAL_ADC_IRQHandler(&hadc1); }
void SysTick_Handler(void) { HAL_IncTick(); }
void Error_Handler(void) {
    __disable_irq();
    /* All PWM channels disabled even if normal HAL state was damaged. */
    TIM1->CCER=0; TIM8->CCER=0; TIM1->BDTR&=~TIM_BDTR_MOE; TIM8->BDTR&=~TIM_BDTR_MOE;
    for(;;) { __NOP(); } /* IWDG resets once started; never fed here. */
}
void HardFault_Handler(void) { Error_Handler(); }
void MemManage_Handler(void) { Error_Handler(); }
void BusFault_Handler(void) { Error_Handler(); }
void UsageFault_Handler(void) { Error_Handler(); }
void NMI_Handler(void) { Error_Handler(); }
/* Vendor Reset_Handler initializes RAM and calls __libc_init_array. There is
 * no operating-system CRT, stdio or heap initialization on this target. */
void _init(void) {}
void _fini(void) {}
