#include "neural_network.h"

void net_init(Net *net, Layer *layers, int num_layers, int loss_type)
{
	net->layers = layers;
	net->num_layers = num_layers;
	loss_init(&net->loss, loss_type);
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
	Matrix *grad_out = grad_loss;

	for (int i = net->num_layers - 1; i >= 0; i--) {
		grad_out = layer_backward(&net->layers[i], grad_out);
	}
}

float train_batch(Net *net, Arena *a, Matrix *x_batch, Matrix *y_batch)
{
	Matrix *pred = net_forward(net, x_batch);

	float loss = loss_forward(&net->loss, pred, y_batch);

	arena_checkpoint(a);

	Matrix grad_loss;
	matrix_create(&grad_loss, a, MAT_ROWS(pred), 1);

	loss_backward(&net->loss, &grad_loss, pred, y_batch);

	net_backward(net, &grad_loss);

	arena_restore(a);

	return loss;
}

