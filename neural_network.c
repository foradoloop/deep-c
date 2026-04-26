#include "neural_network.h"

void net_init(Net *net, Layer *layers, int num_layers, int loss_type)
{
	net->layers = layers;
	net->num_layers = num_layers;
	net->loss = loss(loss_type);
}

void net_setup(Net *net)
{
	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = net->layers + i;

		layer_setup(l);
	}
}

Matrix *net_forward(Net *net, Matrix *x_batch)
{
	Matrix *x_out = x_batch;

	for (int i = 0; i < net->num_layers; i++) {
		x_out = layer_forward(&net->layers[i], x_out);
	}

	return x_out;
}

void net_backward(Net *net, Matrix *grad_loss)
{
	Matrix *grad_y = grad_loss;

	for (int i = net->num_layers - 1; i >= 0; i--) {
		grad_y = layer_backward(&net->layers[i], grad_y);
	}
}

float train_batch(Net *net, Arena *a, Matrix *x_batch, Matrix *y_batch)
{
	Matrix *pred = net_forward(net, x_batch);

	float loss = loss_forward(net->loss, pred, y_batch);

	size_t cp = arena_checkpoint(a);

	Matrix grad_loss;
	matrix_create(&grad_loss, a, MAT_ROWS(pred), MAT_COLS(pred));

	loss_backward(net->loss, &grad_loss, pred, y_batch);

	net_backward(net, &grad_loss);

	arena_restore(a, cp);

	return loss;
}

Matrix **net_params(Net *net, Arena *a)
{
	Matrix **params;

	params = arena_alloc_arr(a, Matrix *, net->num_layers * 2);

	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = &net->layers[i];

		params[i * 2] = &l->w;
		params[i * 2 + 1] = &l->b;
	}

	return params;
}

Matrix **net_grad_params(Net *net, Arena *a)
{
	Matrix **grads;

	grads = arena_alloc_arr(a, Matrix *, net->num_layers * 2);

	for (int i = 0; i < net->num_layers; i++) {
		Layer *l = &net->layers[i];

		grads[i * 2] = &l->grad_w;
		grads[i * 2 + 1] = &l->grad_b;
	}	

	return grads;
}

int net_num_params(Net *net)
{
	return net->num_layers * 2;
}

void net_copy(Net *dst, Net *src)
{
	for (int i = 0; i < src->num_layers; i++) {
		Layer *l_dst = dst->layers + i;
		Layer *l_src = src->layers + i;

		layer_copy(l_dst, l_src);
	}
}

Net *net_clone(Net *src, int batch_size, Arena *a)
{
	Net *net;
	int num_layers;
	Layer *layers;
	int act_type;
	int init_type;

	net = arena_alloc_obj(a, Net);
	num_layers = src->num_layers;
	layers = arena_alloc_arr(a, Layer, num_layers);

	for (int i = 0; i < num_layers; i++) {
		Layer *ls = src->layers + i;
		act_type = ls->act->type;
		init_type = ls->init->type;
		int in = MAT_ROWS(&ls->w);
		int out = MAT_COLS(&ls->w);

		layer_create(layers + i, a, in, out, batch_size, act_type, init_type);
		layer_copy(layers + i, ls);
	}

	net->layers = layers;
	net->num_layers = num_layers;
	net->loss = src->loss;

	return net;
}

