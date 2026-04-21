#include "prng.h"

void prng_seed(uint64_t initstate, uint64_t initseq)
{
	pcg32_srandom(initstate, initseq);
}

uint32_t prng_u32()
{
	return pcg32_random();
}

uint32_t prng_u32bounded(uint32_t bound)
{
	return pcg32_boundedrand(bound);
}

float prng_f32()
{
	uint32_t mantissa = prng_u32() >> 9;
	uint32_t r = (0x3F800000 | mantissa);

	union { uint32_t u32; float f; } u = { .u32 = r };

	return u.f - 1.0f;
}

float prng_f32bounded(float min, float max)
{
	return min + (max - min) * prng_f32();
}

