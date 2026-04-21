#include "layer.h"
#include "prng.h"

void layer_create(Layer *l, Arena *a, int in, int out, int act_type, int init_type)
{
	matrix_create(&l->w, a, out, in);
	matrix_create(&l->x, a, in, 1);
	matrix_create(&l->b, a, out, 1);
	matrix_create(&l->i, a, out, 1);
	matrix_create(&l->y, a, out, 1);
	matrix_create(&l->di, a, out, 1);
	matrix_create(&l->delta, a, out, 1);
	matrix_create(&l->grad_w, a, out, in);
	matrix_create(&l->grad_b, a, out, 1);
	matrix_create(&l->grad_out, a, in, 1);
	l->act = activation(act_type);
	l->init = initializer(init_type);
}

void layer_setup(Layer *l)
{
	Matrix *w = &l->w;
	int in = MAT_COLS(w);
	int out = MAT_ROWS(w);

	for (int i = 0; i < MAT_SIZE(&l->w); i++) {
		MAT_GET(w, i) = l->init->apply(in, out);
	}

	matrix_zero(&l->b);
}

Matrix *layer_forward(Layer *l, Matrix *input)
{
	matrix_copy(&l->x, input);
	matrix_mul_ab(&l->i, &l->w, &l->x);
	matrix_add(&l->i, &l->i, &l->b);
	matrix_map(&l->y, &l->i, l->act->forward);

	return &l->y;
}

Matrix *layer_backward(Layer *l, Matrix *grad_out)
{
	matrix_map(&l->di, &l->y, l->act->backward);
	matrix_hadamard(&l->delta, grad_out, &l->di);

	matrix_mul_abt(&l->grad_w, &l->delta, &l->x);

	matrix_copy(&l->grad_b, &l->delta);

	matrix_mul_atb(&l->grad_out, &l->w, &l->delta);

	return &l->grad_out;
}

void layer_copy(Layer *dst, Layer *src)
{
	matrix_copy(&dst->w, &src->w);
	matrix_copy(&dst->b, &src->b);
}

