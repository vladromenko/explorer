#include "board.h"
#include <string.h>

/* The motor ISR never waits for UART. All ports use interrupt-mode I/O;
 * buffers live in CPU-accessible DTCM, never in a DMA transaction. */
#define RX_SIZE 4096u
#define TX_SIZE 2048u
UART_HandleTypeDef board_uart[PORT_COUNT];
volatile uint32_t board_rx_overruns[PORT_COUNT], board_uart_errors[PORT_COUNT];
volatile uint32_t board_tx_completed[PORT_COUNT];
volatile uint64_t board_tx_completed_us[PORT_COUNT];
static uint64_t tx_deadline[PORT_COUNT];
static uint8_t rx_byte[PORT_COUNT],rx[PORT_COUNT][RX_SIZE],tx[PORT_COUNT][TX_SIZE];
static volatile uint16_t rx_write[PORT_COUNT],rx_read[PORT_COUNT];
HAL_StatusTypeDef board_uart_receive_start(unsigned index) {
    if(index>=PORT_COUNT) { return HAL_ERROR; }
    return HAL_UART_Receive_IT(&board_uart[index],&rx_byte[index],1);
}
bool board_receive(unsigned i,uint8_t *value) {
    if(i>=PORT_COUNT || !value || rx_read[i]==rx_write[i]) { return false; }
    *value=rx[i][rx_read[i]]; __DMB(); rx_read[i]=(uint16_t)((rx_read[i]+1u)%RX_SIZE); return true;
}
bool board_send(unsigned i,const uint8_t *data,uint16_t n) {
    if(i>=PORT_COUNT || !data || !n || n>TX_SIZE || board_uart[i].gState!=HAL_UART_STATE_READY) { return false; }
    memcpy(tx[i],data,n);
    /* 8N1, physical wire duration plus a bounded ISR scheduling allowance.
     * This is not a servo response deadline. */
    tx_deadline[i]=board_time_us()+((uint64_t)n*10000000u+board_uart[i].Init.BaudRate-1u)/
        board_uart[i].Init.BaudRate+20000u;
    return HAL_UART_Transmit_IT(&board_uart[i],tx[i],n)==HAL_OK;
}
void board_io_poll(void) {
    const uint64_t now=board_time_us();
    for(unsigned i=0;i<PORT_COUNT;++i) {
        UART_HandleTypeDef *u=&board_uart[i];
        uint32_t saved=__get_PRIMASK(); __disable_irq();
        if(u->gState!=HAL_UART_STATE_READY && tx_deadline[i] && now>=tx_deadline[i]) {
            /* UARTs use IT, never DMA: this HAL operation is bounded and does
             * not wait for a wire response. A truncated frame is not replayed.
             * No TC callback is fabricated. Consumers see a transport error. */
            (void)HAL_UART_AbortTransmit(u); ++board_uart_errors[i]; tx_deadline[i]=0;
        }
        if(u->RxState==HAL_UART_STATE_READY) {
            if(HAL_UART_Receive_IT(u,&rx_byte[i],1)!=HAL_OK) { ++board_uart_errors[i]; }
        }
        __set_PRIMASK(saved);
    }
}
void HAL_UART_TxCpltCallback(UART_HandleTypeDef *u) {
    unsigned i=(unsigned)(u-board_uart);
    if(i<PORT_COUNT) {
        /* ISR observation of TC, not a hardware timestamp of the last bit. */
        board_tx_completed_us[i]=board_time_us();
        __DMB(); ++board_tx_completed[i];
    }
}
void HAL_UART_RxCpltCallback(UART_HandleTypeDef *u) {
    unsigned i=(unsigned)(u-board_uart);
    if(i<PORT_COUNT) {
        uint16_t next=(uint16_t)((rx_write[i]+1u)%RX_SIZE);
        if(next==rx_read[i]) { ++board_rx_overruns[i]; }
        else { rx[i][rx_write[i]]=rx_byte[i]; __DMB(); rx_write[i]=next; }
        if(HAL_UART_Receive_IT(u,&rx_byte[i],1)!=HAL_OK) { ++board_uart_errors[i]; }
    }
}
void HAL_UART_ErrorCallback(UART_HandleTypeDef *u) {
    unsigned i=(unsigned)(u-board_uart);
    if(i<PORT_COUNT) {
        ++board_uart_errors[i]; __HAL_UART_CLEAR_OREFLAG(u);
        (void)HAL_UART_Receive_IT(u,&rx_byte[i],1);
    }
}
