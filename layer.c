#include "layer.h"

void layer_create(Layer *l, Arena *a, int in, int out, int type)
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
	activation_init(&l->act, type);
}

Matrix *layer_forward(Layer *l, Matrix *input)
{
	matrix_copy(&l->x, input);
	matrix_mul_ab(&l->i, &l->w, &l->x);
	matrix_add(&l->i, &l->i, &l->b);
	matrix_map(&l->y, &l->i, l->act.forward);

	return &l->y;
}

Matrix *layer_backward(Layer *l, Matrix *grad_out)
{
	matrix_map(&l->di, &l->y, l->act.backward);
	matrix_hadamard(&l->delta, grad_out, &l->di);

	matrix_mul_abt(&l->grad_w, &l->delta, &l->x);

	matrix_copy(&l->grad_b, &l->delta);

	matrix_mul_atb(&l->grad_out, &l->w, &l->delta);

	return &l->grad_out;
}

