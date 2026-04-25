#include "layer.h"
#include "prng.h"

void layer_create(Layer *l, Arena *a, int in, int out, int act_type, int init_type)
{
	/* Forward */
	matrix_create(&l->w, a, in, out);
	matrix_create(&l->x, a, 1, in);
	matrix_create(&l->b, a, 1, out);
	matrix_create(&l->i, a, 1, out);
	matrix_create(&l->y, a, 1, out);

	/* Backward */
	matrix_create(&l->di, a, 1, out);
	matrix_create(&l->delta, a, 1, out);
	matrix_create(&l->grad_w, a, in, out);
	matrix_create(&l->grad_b, a, 1, out);
	matrix_create(&l->grad_y, a, 1, in);

	l->act = activation(act_type);
	l->init = initializer(init_type);
}

void layer_setup(Layer *l)
{
	Matrix *w = &l->w;
	int in = MAT_ROWS(w);
	int out = MAT_COLS(w);

	for (int i = 0; i < MAT_SIZE(&l->w); i++) {
		MAT_GET(w, i) = l->init->apply(in, out);
	}

	matrix_zero(&l->b);
}

Matrix *layer_forward(Layer *l, Matrix *input)
{
	/* Input cache  */
	matrix_copy(&l->x, input);

	/* Linear combination */
	matrix_mul_ab(&l->i, &l->x, &l->w);
	matrix_add(&l->i, &l->i, &l->b);

	/* Output */
	matrix_map(&l->y, &l->i, l->act->forward);

	return &l->y;
}

Matrix *layer_backward(Layer *l, Matrix *grad_y)
{
	/* Delta */
	matrix_map(&l->di, &l->y, l->act->backward);
	matrix_hadamard(&l->delta, grad_y, &l->di);

	/* Weight gradient */
	matrix_mul_atb(&l->grad_w, &l->x, &l->delta);

	/* Bias gradient */
	matrix_copy(&l->grad_b, &l->delta);

	/* Output gradient */
	matrix_mul_abt(&l->grad_y, &l->delta, &l->w);

	return &l->grad_y;
}

void layer_copy(Layer *dst, Layer *src)
{
	matrix_copy(&dst->w, &src->w);
	matrix_copy(&dst->b, &src->b);
}

