#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"
#include "initializer.h"

struct layer {
	Matrix w;
	Matrix x;
	Matrix b;
	Matrix i;
	Matrix y;
	Matrix di;
	Matrix delta;
	Matrix grad_w;
	Matrix grad_b;
	Matrix grad_out;
	const Activation *act;
	const Init *init;
};
typedef struct layer Layer;

void layer_create(Layer *l, Arena *a, int in, int out, int act_type, int init_type);
void layer_setup(Layer *l);
Matrix *layer_forward(Layer *l, Matrix *input);
Matrix *layer_backward(Layer *l, Matrix *output_grad);
void layer_copy(Layer *dst, Layer *src);

#endif

