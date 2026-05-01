#ifndef NEURAL_NETWORK
#define NEURAL_NETWORK

#include "layer.h"
#include "arena.h"
#include "matrix.h"
#include "loss.h"

struct neural_network {
	Layer *layers;
	int num_layers;
	const Loss *loss;
};
typedef struct neural_network Net;

void net_init(Net *net, Layer *layers, int num_layers, int loss_type);
void net_setup(Net *net);
Matrix *net_forward(Net *net, Matrix *x_batch);
void net_backward(Net *net, Matrix *grad_loss);
float train_batch(Net *net, Arena *a, Matrix *x_batch, Matrix *y_batch);
Matrix **net_params(Net *net, Arena *a);
Matrix **net_grad_params(Net *net, Arena *a);
int net_num_params(Net *net);
void net_copy(Net *dst, Net *src);
Net *net_clone(Net *src, int batch_size, Arena *a);

void net_save_binary(Net *net, FILE *f);
void net_load_binary(Net *net, Arena *a, int batch_size, FILE *f);

#endif

