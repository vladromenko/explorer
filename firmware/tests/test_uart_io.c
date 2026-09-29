#include "board.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
uint32_t mock_primask;
static uint64_t now;
static uint8_t *rx_target[PORT_COUNT];
static unsigned aborts,tx_requests,rx_failures;
uint64_t board_time_us(void) { return now; }
HAL_StatusTypeDef HAL_UART_Receive_IT(UART_HandleTypeDef *u,uint8_t *p,uint16_t n) {
    assert(n==1);
    if(rx_failures) { --rx_failures; return HAL_ERROR; }
    if(u->RxState!=HAL_UART_STATE_READY) { return HAL_BUSY; }
    rx_target[u-board_uart]=p; u->RxState=HAL_UART_STATE_BUSY; return HAL_OK;
}
HAL_StatusTypeDef HAL_UART_Transmit_IT(UART_HandleTypeDef *u,uint8_t *p,uint16_t n) {
    assert(n>0 && p && u->gState==HAL_UART_STATE_READY);
    u->gState=HAL_UART_STATE_BUSY; ++tx_requests; return HAL_OK;
}
HAL_StatusTypeDef HAL_UART_AbortTransmit(UART_HandleTypeDef *u) {
    assert(mock_primask==1); ++aborts; u->gState=HAL_UART_STATE_READY; return HAL_OK;
}
static void receive(unsigned port,uint8_t value) {
    assert(board_uart[port].RxState==HAL_UART_STATE_BUSY);
    *rx_target[port]=value; board_uart[port].RxState=HAL_UART_STATE_READY;
    HAL_UART_RxCpltCallback(&board_uart[port]);
}
int main(void) {
    for(unsigned i=0;i<PORT_COUNT;++i) {
        board_uart[i].Init.BaudRate=i==PORT_HOST?2000000:115200;
        assert(board_uart_receive_start(i)==HAL_OK);
    }
    assert(board_uart_receive_start(PORT_COUNT)==HAL_ERROR);
    uint8_t byte=0;
    assert(!board_receive(PORT_ARM,&byte));
    receive(PORT_ARM,17); receive(PORT_HOST,23);
    assert(board_receive(PORT_HOST,&byte) && byte==23);
    assert(board_receive(PORT_ARM,&byte) && byte==17);
    assert(!board_receive(PORT_ARM,&byte));
    for(unsigned i=0;i<4100;++i) { receive(PORT_LIDAR0,(uint8_t)i); }
    assert(board_rx_overruns[PORT_LIDAR0]==5);
    for(unsigned i=0;i<4095;++i) { assert(board_receive(PORT_LIDAR0,&byte) && byte==(uint8_t)i); }
    assert(!board_receive(PORT_LIDAR0,&byte));
    receive(PORT_LIDAR0,123); assert(board_receive(PORT_LIDAR0,&byte) && byte==123);
    /* Error IRQ and a one-off rearm failure must not leave a silent receiver. */
    board_uart[PORT_ARM].RxState=HAL_UART_STATE_READY; rx_failures=1;
    HAL_UART_ErrorCallback(&board_uart[PORT_ARM]);
    assert(board_uart[PORT_ARM].RxState==HAL_UART_STATE_READY);
    board_io_poll(); receive(PORT_ARM,42);
    assert(board_receive(PORT_ARM,&byte) && byte==42);
    uint8_t packet[38]={1}; unsigned errors=board_uart_errors[PORT_ARM];
    assert(!board_send(PORT_ARM,NULL,38) && !board_send(PORT_ARM,packet,0));
    assert(board_send(PORT_ARM,packet,38));
    assert(!board_send(PORT_ARM,packet,38));
    now=20000; board_io_poll(); assert(aborts==0);
    now=24000; board_io_poll();
    assert(aborts==1 && board_uart_errors[PORT_ARM]==errors+1);
    assert(board_tx_completed[PORT_ARM]==0); /* Aborted is never sent. */
    assert(board_send(PORT_ARM,packet,38));
    board_uart[PORT_ARM].gState=HAL_UART_STATE_READY;
    HAL_UART_TxCpltCallback(&board_uart[PORT_ARM]);
    assert(board_tx_completed[PORT_ARM]==1);
    now=100000; board_io_poll(); assert(aborts==1 && tx_requests==2 && mock_primask==0);
    /* Preserve an already-masked caller's state; do not reopen interrupts. */
    mock_primask=1; board_io_poll(); assert(mock_primask==1);
    puts("real UART transport: ring overflow, RX recovery, bounded TX abort, true TC: PASS");
}
