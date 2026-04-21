#include "initializer.h"
#include "prng.h"
#include <math.h>

static float _glorot_apply(int in, int out); 
static float _he_apply(int in, int out);

static const Init init_table[] = {
	{ GLOROT, _glorot_apply },
	{ HE, _he_apply }
};

const Init *initializer(int type)
{
	return &init_table[type];
}

static float _glorot_apply(int in, int out)
{
	float limit;

	limit = sqrtf(6.0f / (float)(in + out));

	return prng_f32bounded(-limit, limit);
}

static float _he_apply(int in, int out)
{
	(void)out;
	float limit;

	limit = sqrtf(6.0f / (float)in);

	return prng_f32bounded(-limit, limit);
}

