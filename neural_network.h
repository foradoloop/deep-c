#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "layer.h"
#include "arena.h"
#include "matrix.h"
#include "loss.h"

struct neural_network {
	Layer *layers;
	int num_layers;
	Loss loss;
};
typedef struct neural_network Net;

void net_init(Net *net, Layer *layers, int num_layers, int loss_type);
Matrix *net_forward(Net *net, Matrix *x_batch);
void net_backward(Net *net, Matrix *grad_loss);
float train_batch(Net *net, Arena *a, Matrix *x_batch, Matrix *y_batch);

#endif

