#include "loss.h"
#include <math.h>

static float _mse_output(Matrix *pred, Matrix *tar);
static void _mse_input_grad(Matrix *grad_loss, Matrix *pred, Matrix *tar);

static float _sce_output(Matrix *pred, Matrix *tar);
static void _sce_input_grad(Matrix *grad_loss, Matrix *pred, Matrix *tar);

static const Loss loss_table[] = {
	{ .type = MSE, ._output = _mse_output, ._input_grad = _mse_input_grad },
	{ .type = SCE, ._output = _sce_output, ._input_grad = _sce_input_grad }
};

const Loss *_loss(int type)
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

	loss /= MAT_SIZE(pred);

	return loss;
}

static void _mse_input_grad(Matrix *grad_loss, Matrix *pred, Matrix *tar)
{
	for (int i = 0; i < MAT_SIZE(grad_loss); i++) {
		MAT_GET(grad_loss, i) = 2.0f * (MAT_GET(pred, i) - MAT_GET(tar, i)) / MAT_SIZE(pred);
	}
}

static void softmax_row(Matrix *pred, int row, float *den, float *max)
{
	Matrix m;
	float maxval;
	float denominator;
	float x;

	matrix_init(&m, &MAT_AT(pred, row, 0), 1, MAT_COLS(pred));
	maxval = matrix_maxval(&m);
	denominator = 0.0f;

	for (int i = 0; i < MAT_COLS(pred); i++) {
		x = MAT_AT(pred, row, i);
		denominator += expf(x - maxval);
	}

	*den = denominator;
	*max = maxval;
}

static float _sce_output(Matrix *pred, Matrix *tar)
{
	float den = 0.0f;
	float p = 0.0f;
	float y = 0.0f;
	float loss = 0.0f;
	float maxval = 0.0f;
	float z = 0.0f;
	float eps = 1e-7;

	for (int i = 0; i < MAT_SIZE(pred); i++) {
		if ((i % MAT_COLS(pred)) == 0) {
			softmax_row(pred, i / MAT_COLS(pred), &den, &maxval);
		}

		z = MAT_GET(pred, i) - maxval;
		p = expf(z) / den;
		p = fmaxf(p, eps);
		p = fminf(p, 1.0f - eps);
		y = MAT_GET(tar, i);

		loss += -logf(1 - p + y * (-1 + 2 * p));
	}

	return loss / MAT_ROWS(pred);
}

static void _sce_input_grad(Matrix *grad_loss, Matrix *pred, Matrix *tar)
{
	float den = 0.0f;
	float p = 0.0f;
	float y = 0.0f;
	float maxval = 0.0f;
	float z = 0.0f;

	for (int i = 0; i < MAT_SIZE(pred); i++) {
		if ((i % MAT_COLS(pred)) == 0) {
			softmax_row(pred, i / MAT_COLS(pred), &den, &maxval);
		}

		z = MAT_GET(pred, i) - maxval;
		p = expf(z) / den;
		y = MAT_GET(tar, i);

		/* No clamping needed: softmax guarantees p in (0, 1), p - y is numerically stable */
		MAT_GET(grad_loss, i) = p - y;
	}
}

