SECTIONS {
 .twist_hook 0x0800b65c : { KEEP(*(.twist_hook)) }
 .tick_hook 0x08000dfc : { KEEP(*(.tick_hook)) }
 .patch 0x08057000 : { *(.text*) *(.rodata*) }
 /DISCARD/ : { *(.ARM.exidx*) *(.ARM.extab*) *(.comment*) *(.note*) }
}
ASSERT(SIZEOF(.patch) < 4096, "Patch exceeds audited flash window")
ASSERT(SIZEOF(.data) == 0, "No new RAM data permitted")
ASSERT(SIZEOF(.bss) == 0, "No new RAM allocation permitted")
