#ifndef ARENA_H
#define ARENA_H

#include <stddef.h>

#define arena_alloc_obj(a, type) \
	arena_alloc(a, sizeof(type))

#define arena_alloc_arr(a, type, count) \
	arena_alloc(a, sizeof(type) * (count))

#define  B (sizeof(unsigned char))
#define KB (1024 * B)
#define MB (1024 * KB)
#define GB (1024 * MB)

struct arena {
	unsigned char *ptr;
	size_t size;
	size_t offset;
};
typedef struct arena Arena;

void arena_create(Arena *a, size_t size);
void arena_init(Arena *a, unsigned char *buffer, size_t buflen);
void *arena_alloc(Arena *a, size_t alloc);
size_t arena_checkpoint(Arena *a);
void arena_restore(Arena *a, size_t checkpoint);
void arena_reset(Arena *a);
void arena_destroy(Arena *a);

#endif

