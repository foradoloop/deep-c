#include "io.h"
#include <stdarg.h>

void xkill(const char *string, ...)
{
	if (!string) {
		exit(1);
	}

	va_list ap;
	va_start(ap, string);

	vfprintf(stderr, string, ap);

	va_end(ap);

	exit(1);
}

FILE *xfopen(const char *filename, const char *mode)
{
	FILE *f;
	
	f = fopen(filename, mode);
	if (!f) {
		xkill("Could not open %s on %s mode\n", filename, mode);
	}

	return f;
}

void xfclose(FILE *f)
{
	if (fclose(f)) {
		xkill("Could not close file\n");
	}
}

void xfread(void *buffer, size_t size, size_t count, FILE *stream)
{
	size_t nread;

	nread = fread(buffer, size, count, stream);
	if (nread != count) {
		xkill("Read %lu elements expected %lu\n", nread, count);
	}
}

void xfwrite(void *buffer, size_t size, size_t count, FILE *stream)
{
	size_t nwritten;

	nwritten = fwrite(buffer, size, count, stream);
	if (nwritten != count) {
		xkill("Written %lu elements expected %lu\n", nwritten, count);
	}
}

u32 xfread_u32be(FILE *stream)
{
	u8 w[4];

	xfread(w, sizeof(u8), 4, stream);

	return (
			(u32)w[0] << 24 |
			(u32)w[1] << 16 |
			(u32)w[2] << 8  |
			(u32)w[3]
			);
}

void xfprintf(FILE *stream, const char *format_string, ...)
{
	va_list ap;
	int n;

	va_start(ap, format_string);

	n = vfprintf(stream, format_string, ap);

	va_end(ap);

	if (n < 0) {
		xkill("Could not write string on output stream\n");
	}
}

