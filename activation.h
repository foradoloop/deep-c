#ifndef ACTIVATION_H
#define ACTIVATION_H

enum {
	TANH = 0,
	RELU,
	SIGMOID
};

struct activation {
	int type;
	float (*forward)(float);
	float (*backward)(float);
};
typedef struct activation Activation;

void activation_init(Activation *act, int type);

#endif

