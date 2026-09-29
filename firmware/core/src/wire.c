#include "explorer_wire.h"
#include <string.h>

uint16_t ew_u16(const uint8_t *p) { return (uint16_t)((uint16_t)p[0] | ((uint16_t)p[1]<<8)); }
uint32_t ew_u32(const uint8_t *p) { return (uint32_t)ew_u16(p) | ((uint32_t)ew_u16(p+2)<<16); }
uint64_t ew_u64(const uint8_t *p) { return (uint64_t)ew_u32(p) | ((uint64_t)ew_u32(p+4)<<32); }
float ew_f32(const uint8_t *p) { uint32_t bits=ew_u32(p); float v; memcpy(&v,&bits,4); return v; }
void ew_put16(uint8_t *p,uint16_t v) { p[0]=(uint8_t)v; p[1]=(uint8_t)(v>>8); }
void ew_put32(uint8_t *p,uint32_t v) { ew_put16(p,(uint16_t)v); ew_put16(p+2,(uint16_t)(v>>16)); }
void ew_put64(uint8_t *p,uint64_t v) { ew_put32(p,(uint32_t)v); ew_put32(p+4,(uint32_t)(v>>32)); }
void ew_putf32(uint8_t *p,float v) { uint32_t bits; memcpy(&bits,&v,4); ew_put32(p,bits); }
uint32_t ew_crc32(const uint8_t *p,size_t n) {
    uint32_t crc=UINT32_MAX;
    for(size_t i=0;i<n;++i) {
        crc^=p[i];
        for(unsigned k=0;k<8;++k) { crc=(crc>>1)^((crc&1u)?UINT32_C(0xedb88320):0u); }
    }
    return ~crc;
}
size_t ew_encode(uint8_t type,const uint8_t *p,size_t n,uint8_t out[EW_MAX_FRAME]) {
    if(n>EW_MAX_PAYLOAD || (!p && n) || !out) { return 0; }
    uint8_t raw[EW_MAX_PAYLOAD+8u];
    raw[0]=EW_VERSION; raw[1]=type; ew_put16(raw+2,(uint16_t)n);
    if(n) { memcpy(raw+4,p,n); }
    ew_put32(raw+n+4,ew_crc32(raw,n+4));
    size_t write=1,code_at=0; uint8_t code=1;
    for(size_t i=0;i<n+8;++i) {
        if(raw[i]==0) { out[code_at]=code; code_at=write++; code=1; }
        else {
            out[write++]=raw[i]; ++code;
            if(code==255) { out[code_at]=code; code_at=write++; code=1; }
        }
    }
    out[code_at]=code; out[write++]=0; return write;
}
static bool decode(const uint8_t *in,size_t n,ew_frame_t *f) {
    uint8_t raw[EW_MAX_PAYLOAD+8u]; size_t read=0,write=0;
    while(read<n) {
        const unsigned code=in[read++];
        if(code==0 || read+code-1u>n || write+code-1u>sizeof(raw)) { return false; }
        for(unsigned k=1;k<code;++k) { raw[write++]=in[read++]; }
        if(code<255 && read<n) {
            if(write==sizeof(raw)) { return false; }
            raw[write++]=0;
        }
    }
    if(write<8 || raw[0]!=EW_VERSION || ew_u16(raw+2)!=write-8 ||
       ew_u32(raw+write-4)!=ew_crc32(raw,write-4)) { return false; }
    f->type=raw[1]; f->length=(uint16_t)(write-8);
    memcpy(f->payload,raw+4,f->length); return true;
}
bool ew_receive(ew_parser_t *p,uint8_t b,ew_frame_t *f) {
    if(!p || !f) { return false; }
    if(b) {
        if(p->used<sizeof(p->encoded) && !p->overflow) { p->encoded[p->used++]=b; }
        else if(!p->overflow) { p->overflow=true; ++p->oversized; }
        return false;
    }
    bool ok=false;
    if(p->used && !p->overflow) {
        ok=decode(p->encoded,p->used,f);
        if(!ok) { ++p->bad_frames; }
    }
    p->used=0; p->overflow=false; return ok;
}
