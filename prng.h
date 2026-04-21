#ifndef PRNG_H
#define PRNG_H

#include "pcg_basic.h"

void prng_seed(uint64_t initstate, uint64_t initseq);
uint32_t prng_u32();
uint32_t prng_u32bounded(uint32_t bound);
float prng_f32();
float prng_f32bounded(float min, float max);

#endif

