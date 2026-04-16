#ifndef LOSS_H
#define LOSS_H

#include "matrix.h"

enum {
	MSE = 0
};

struct loss {
	int type;
	float (*_output)(Matrix *pred, Matrix *tar);
	void (*_input_grad)(Matrix *grad_loss, Matrix *pred, Matrix *tar);
};
typedef struct loss Loss;

void loss_init(Loss *loss, int type);
float loss_forward(Loss *loss, Matrix *pred, Matrix *tar);
void loss_backward(Loss *loss, Matrix *grad_loss, Matrix *pred, Matrix *tar);

#endif

