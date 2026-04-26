#ifndef MATRIX_H
#define MATRIX_H

#include "arena.h"

#define MAT_DATA(m) ((m)->data)
#define MAT_ROWS(m) ((m)->rows)
#define MAT_COLS(m) ((m)->cols)

#define MAT_SIZE(m) (MAT_ROWS(m) * MAT_COLS(m))
#define MAT_AT(m, i, j) (MAT_DATA(m)[MAT_COLS(m) * (i) + (j)])
#define MAT_GET(m, i) (MAT_DATA(m)[i])

struct matrix {
	float *data;
	int rows;
	int cols;
};
typedef struct matrix Matrix;

void matrix_create(Matrix *m, Arena *a, int rows, int cols);
void matrix_init(Matrix *m, float *data, int rows, int cols);
void matrix_add(Matrix *rop, Matrix *op1, Matrix *op2);
void matrix_sub(Matrix *rop, Matrix *op1, Matrix *op2);
void matrix_mul_ab(Matrix *rop, Matrix *op1, Matrix *op2);
void matrix_mul_atb(Matrix *rop, Matrix *at, Matrix *b);
void matrix_mul_abt(Matrix *rop, Matrix *a, Matrix *bt);
void matrix_hadamard(Matrix *rop, Matrix *op1, Matrix *op2);
void matrix_copy(Matrix *dst, Matrix *src);
void matrix_map(Matrix *rop, Matrix *op, float (*fn)(float));
void matrix_zero(Matrix *rop);
void matrix_broadcast_add(Matrix *rop, Matrix *broadcast, Matrix *op);
void matrix_sum_cols(Matrix *rop, Matrix *op);

#endif

