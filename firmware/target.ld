/* Conservative target placement: first 128 KiB only, shared by H743 G/I.
 * This linker region is NOT confirmation of the installed part/backup coverage.
 * No persistent configuration region is erased or programmed by the application.
 */
ENTRY(Reset_Handler)
MEMORY {
 FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 128K
 DTCM (xrw) : ORIGIN = 0x20000000, LENGTH = 128K
 D2_DMA (rw) : ORIGIN = 0x30000000, LENGTH = 1K
}
_estack = ORIGIN(DTCM) + LENGTH(DTCM);
PHDRS {
 code PT_LOAD FLAGS(5);
 data PT_LOAD FLAGS(6);
 ram PT_LOAD FLAGS(6);
 dma PT_LOAD FLAGS(6);
}
SECTIONS {
 .isr_vector : { KEEP(*(.isr_vector)) } > FLASH :code
 .text : { *(.text*) *(.rodata*) KEEP(*(.init)) KEEP(*(.fini)) . = ALIGN(4); } > FLASH
 .ARM.extab : { *(.ARM.extab*) } > FLASH
 .ARM.exidx : { __exidx_start = .; *(.ARM.exidx*) __exidx_end = .; } > FLASH
 .preinit_array : { PROVIDE_HIDDEN(__preinit_array_start = .); KEEP(*(.preinit_array*)) PROVIDE_HIDDEN(__preinit_array_end = .); } > FLASH
 .init_array : { PROVIDE_HIDDEN(__init_array_start = .); KEEP(*(SORT(.init_array.*))) KEEP(*(.init_array*)) PROVIDE_HIDDEN(__init_array_end = .); } > FLASH
 .fini_array : { PROVIDE_HIDDEN(__fini_array_start = .); KEEP(*(SORT(.fini_array.*))) KEEP(*(.fini_array*)) PROVIDE_HIDDEN(__fini_array_end = .); } > FLASH
 _sidata = LOADADDR(.data);
 .data : { . = ALIGN(8); _sdata = .; *(.data*) . = ALIGN(8); _edata = .; } > DTCM AT > FLASH :data
 .bss (NOLOAD) : { . = ALIGN(8); _sbss = .; __bss_start__ = .; *(.bss*) *(COMMON) . = ALIGN(8); _ebss = .; __bss_end__ = .; } > DTCM :ram
 .heap_stack (NOLOAD) : { . = ALIGN(8); PROVIDE(end = .); PROVIDE(_end = .); . += 16384; } > DTCM
 ASSERT(_ebss + 16384 <= _estack, "DTCM exhausted; reserve 16 KiB stack")
 .dma_rgb (NOLOAD) : { . = ALIGN(32); *(.dma_rgb) . = ALIGN(32); } > D2_DMA :dma
 /DISCARD/ : { *(.note.GNU-stack) *(.comment) }
}
