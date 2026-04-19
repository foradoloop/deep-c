#include "matrix.h"
#include <string.h>

void matrix_create(Matrix *m, Arena *a, int rows, int cols)
{
	MAT_DATA(m) = arena_alloc_arr(a, float, rows * cols);
	MAT_ROWS(m) = rows;
	MAT_COLS(m) = cols;
}

void matrix_init(Matrix *m, float *data, int rows, int cols)
{
	MAT_DATA(m) = data;
	MAT_ROWS(m) = rows;
	MAT_COLS(m) = cols;
}

void matrix_add(Matrix *rop, Matrix *op1, Matrix *op2)
{
	for (int i = 0; i < MAT_SIZE(rop); i++) {
		MAT_GET(rop, i) = MAT_GET(op1, i) + MAT_GET(op2, i);
	}
}

void matrix_sub(Matrix *rop, Matrix *op1, Matrix *op2)
{
	for (int i = 0; i < MAT_SIZE(rop); i++) {
		MAT_GET(rop, i) = MAT_GET(op1, i) - MAT_GET(op2, i);
	}
}

void matrix_mul_ab(Matrix *rop, Matrix *op1, Matrix *op2)
{
	for (int i = 0; i < MAT_ROWS(op1); i++) {
		for (int j = 0; j < MAT_COLS(op2); j++) {
			float sum = 0;
			for (int k = 0; k < MAT_COLS(op1); k++) {
				sum += MAT_AT(op1, i, k) * MAT_AT(op2, k, j);
			}
			MAT_AT(rop, i, j) = sum;
		}
	}
}

void matrix_mul_atb(Matrix *rop, Matrix *at, Matrix *b)
{
	for (int i = 0; i < MAT_COLS(at); i++) {
		for (int j = 0; j < MAT_COLS(b); j++) {
			float sum = 0;
			for (int k = 0; k < MAT_ROWS(at); k++) {
				sum += MAT_AT(at, k, i) * MAT_AT(b, k, j);
			}
			MAT_AT(rop, i, j) = sum;
		}
	}
}

void matrix_mul_abt(Matrix *rop, Matrix *a, Matrix *bt)
{
	for (int i = 0; i < MAT_ROWS(a); i++) {
		for (int j = 0; j < MAT_ROWS(bt); j++) {
			float sum = 0;
			for (int k = 0; k < MAT_COLS(a); k++) {
				sum += MAT_AT(a, i, k) * MAT_AT(bt, j, k);
			}
			MAT_AT(rop, i, j) = sum;
		}
	}
}

void matrix_hadamard(Matrix *rop, Matrix *op1, Matrix *op2)
{
	for (int i = 0; i < MAT_SIZE(rop); i++) {
		MAT_GET(rop, i) = MAT_GET(op1, i) * MAT_GET(op2, i);
	}
}

void matrix_copy(Matrix *dst, Matrix *src)
{
	memcpy(dst->data, src->data, sizeof(float) * MAT_SIZE(dst));
}

void matrix_map(Matrix *rop, Matrix *op, float (*fn)(float))
{
	for (int i = 0; i < MAT_SIZE(rop); i++) {
		MAT_GET(rop, i) = fn(MAT_GET(op, i));
	}
}

void matrix_zero(Matrix *rop)
{
	memset(rop->data, 0, sizeof(float) * MAT_SIZE(rop));
}

