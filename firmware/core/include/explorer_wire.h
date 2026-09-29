#ifndef EXPLORER_WIRE_H
#define EXPLORER_WIRE_H
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

/* COBS-delimited, CRC32-protected frames. No packed struct/wire alignment ABI. */
#define EW_MAX_PAYLOAD 1024u
#define EW_MAX_FRAME 1040u
#define EW_VERSION 1u
enum ew_type {
    EW_HELLO=1, EW_OPEN=2, EW_BASE=3, EW_HOLD=4, EW_CANCEL=5,
    EW_ESTOP=6, EW_CLEAR=7, EW_ARM=8, EW_ARM_ENABLE=9,
    EW_ARM_CANCEL=10, EW_RGB=11, EW_BEEP=12, EW_CALIBRATION=13,
    EW_RECOVERY_ENABLE=14, EW_ARM_RECOVER=15,
    EW_STATUS=64, EW_RESULT=65, EW_SERVO=66, EW_IMU=67,
    EW_BATTERY=68, EW_LIDAR0=69, EW_LIDAR1=70, EW_IDENTITY=71, EW_ARM_DIAGNOSTICS=73
};
typedef struct { uint8_t type; uint16_t length; uint8_t payload[EW_MAX_PAYLOAD]; } ew_frame_t;
typedef struct {
    uint8_t encoded[EW_MAX_FRAME]; size_t used; bool overflow;
    uint32_t bad_frames, oversized;
} ew_parser_t;
uint32_t ew_crc32(const uint8_t *data, size_t length);
size_t ew_encode(uint8_t type, const uint8_t *payload, size_t length,
                 uint8_t output[EW_MAX_FRAME]);
bool ew_receive(ew_parser_t *parser, uint8_t byte, ew_frame_t *frame);
uint16_t ew_u16(const uint8_t *p);
uint32_t ew_u32(const uint8_t *p);
uint64_t ew_u64(const uint8_t *p);
float ew_f32(const uint8_t *p);
void ew_put16(uint8_t *p, uint16_t v);
void ew_put32(uint8_t *p, uint32_t v);
void ew_put64(uint8_t *p, uint64_t v);
void ew_putf32(uint8_t *p, float v);
#endif
