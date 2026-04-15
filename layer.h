#ifndef LAYER_H
#define LAYER_H

#include "matrix.h"
#include "activation.h"

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
	Activation act;
};
typedef struct layer Layer;

void layer_create(Layer *l, Arena *a, int in, int out, int type);
Matrix *layer_forward(Layer *l, Matrix *input);
Matrix *layer_backward(Layer *l, Matrix *output_grad);

#endif

