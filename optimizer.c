#include "optimizer.h"

static void _sgd_step(Optimizer *optim, Matrix *param, Matrix *grad_param);

static const Optimizer _sgd_opt = {
	.type = SGD,
	.step = _sgd_step,
	.lr = 0.01f
};

static const Optimizer opt_table[] = {
	_sgd_opt
};

void optim_init(Optimizer *optim, int type, float lr)
{
	*optim = opt_table[type];
	optim->lr = lr;
}

void optim_step(Optimizer *optim, Matrix **params, Matrix **grad_params, int num_params)
{
	for (int i = 0; i < num_params; i++) {
		optim->step(optim, params[i], grad_params[i]);
	}
}

static void _sgd_step(Optimizer *optim, Matrix *param, Matrix *grad_param)
{
	for (int i = 0; i < MAT_SIZE(param); i++) {
		MAT_GET(param, i) -= optim->lr * MAT_GET(grad_param, i);
	}
}

