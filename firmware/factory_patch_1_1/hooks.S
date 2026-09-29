.syntax unified
.cpu cortex-m7
.thumb
.section .twist_hook,"ax",%progbits
b.w factory_twist
.section .tick_hook,"ax",%progbits
b.w factory_tick
