#include "activation.h"
#include <math.h>

static float _tanh_forward(float);
static float _tanh_backward(float);

static float _relu_forward(float);
static float _relu_backward(float);

static float _sigmoid_forward(float);
static float _sigmoid_backward(float);

static const Activation tanh_act = {
	.type = TANH,
	.forward = _tanh_forward,
	.backward = _tanh_backward
};

static const Activation relu_act = {
	.type = RELU,
	.forward = _relu_forward,
	.backward = _relu_backward
};

static const Activation sigmoid_act = {
	.type = SIGMOID,
	.forward = _sigmoid_forward,
	.backward = _sigmoid_backward
};

static const Activation act_table[] = {
	tanh_act,
	relu_act,
	sigmoid_act
};

void activation_init(Activation *act, int type)
{
	*act = act_table[type];
}

static float _tanh_forward(float x)
{
	return tanhf(x);
}

static float _tanh_backward(float y)
{
	return 1.0f - y * y;
}

static float _relu_forward(float x)
{
	return (x > 0.0f) ? x : 0.0f;
}

static float _relu_backward(float y)
{
	return (y > 0.0f) ? 1.0f : 0.0f;
}

static float _sigmoid_forward(float x)
{
	return 1.0f / (1.0f + expf(-x));
}

static float _sigmoid_backward(float y)
{
	return y * (1.0f - y);
}

