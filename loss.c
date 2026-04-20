#include "loss.h"

static float _mse_output(Matrix *pred, Matrix *tar);
void _mse_input_grad(Matrix *grad_loss, Matrix *pred, Matrix *tar);

static const Loss _mse_loss = {
	.type = MSE,
	._output = _mse_output,
	._input_grad = _mse_input_grad
};

static const Loss loss_table[] = {
	_mse_loss
};

const Loss *loss(int type)
{
	return &loss_table[type];
}

float loss_forward(const Loss *loss, Matrix *pred, Matrix *tar)
{
	return loss->_output(pred, tar);
}

void loss_backward(const Loss *loss, Matrix *grad_loss, Matrix *pred, Matrix *tar)
{
	loss->_input_grad(grad_loss, pred, tar);

}

static float _mse_output(Matrix *pred, Matrix *tar)
{
	float loss = 0.0f;

	for (int i = 0; i < MAT_SIZE(pred); i++) {
		float diff = MAT_GET(pred, i) - MAT_GET(tar, i);

		loss += diff * diff;
	}

	loss /= MAT_ROWS(pred);

	return loss;
}

void _mse_input_grad(Matrix *grad_loss, Matrix *pred, Matrix *tar)
{
	for (int i = 0; i < MAT_SIZE(grad_loss); i++) {
		MAT_GET(grad_loss, i) = 2.0f * (MAT_GET(pred, i) - MAT_GET(tar, i)) / MAT_ROWS(pred);
	}
}

