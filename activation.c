#include "activation.h"
#include <math.h>

static float _tanh_forward(float);
static float _tanh_backward(float);

static float _relu_forward(float);
static float _relu_backward(float);

static float _sigmoid_forward(float);
static float _sigmoid_backward(float);

static float _linear_forward(float x);
static float _linear_backward(float y);

static const Activation act_table[] = {
	{ .type = TANH, .forward = _tanh_forward, .backward = _tanh_backward },
	{ .type = RELU, .forward = _relu_forward, .backward = _relu_backward },
	{ .type = SIGMOID, .forward = _sigmoid_forward, .backward = _sigmoid_backward },
	{ .type = LINEAR, .forward = _linear_forward, .backward = _linear_backward }
};

const Activation *_act(int type)
{
	return &act_table[type];
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

static float _linear_forward(float x)
{
	return x;
}

static float _linear_backward(float y)
{
	(void)y;

	return 1.0f;
}

