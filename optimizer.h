#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "neural_network.h"
#include "matrix.h"

enum {
	SGD = 0
};

typedef struct optimizer Optimizer;

struct optimizer {
	int type;
	void (*step)(Optimizer *optim, Matrix *param, Matrix *grad_param);
	float lr;
};

void optim_init(Optimizer *optim, int type, float lr);
void optim_step(Optimizer *optim, Matrix **params, Matrix **grad_params, int num_params);

#endif

