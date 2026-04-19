#include "arena.h"
#include <stdlib.h>
#include <stdio.h>

void arena_create(Arena *a, size_t size)
{
	a->ptr = malloc(size);
	if (!a->ptr) {
		fprintf(stderr, "Arena malloc fail\n");
		exit(1);
	}
	a->size = size;
	a->offset = 0;
	a->checkpoint = 0;
}

void arena_init(Arena *a, unsigned char *buffer, size_t buflen)
{
	a->ptr = buffer;
	a->size = buflen;
	a->offset = 0;
	a->checkpoint = 0;
}

void *arena_alloc(Arena *a, size_t alloc)
{
	if (a->offset + alloc > a->size) {
		fprintf(stderr, "Arena overflow\n");
		exit(1);
	}

	void *ptr;

	ptr = a->ptr + a->offset;
	a->offset += alloc;

	return ptr;
}

size_t arena_checkpoint(Arena *a)
{
	return a->offset;
}

void arena_restore(Arena *a, size_t checkpoint)
{
	a->offset = checkpoint;
}

void arena_reset(Arena *a)
{
	a->offset = 0;
	a->checkpoint = 0;
}

void arena_destroy(Arena *a)
{
	free(a->ptr);
	a->ptr = NULL;
	a->size = 0;
	a->offset = 0;
	a->checkpoint = 0;
}

