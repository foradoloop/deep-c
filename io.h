#ifndef IO_H
#define IO_H

#include <stdint.h>
#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>

typedef uint8_t  u8;
typedef uint32_t u32;

FILE *xfopen(const char *filename, const char *mode);

void xfclose(FILE *f);

void xfread(void *buffer, size_t size, size_t count, FILE *stream);

void xfwrite(void *buffer, size_t size, size_t count, FILE *stream);

u32 xfread_u32be(FILE *stream);

void xfprintf(FILE *stream, const char *format_string, ...);

#endif

