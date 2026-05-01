#ifndef ACTIVATION_H
#define ACTIVATION_H

enum {
	TANH = 0,
	RELU,
	SIGMOID,
	LINEAR
};

struct activation {
	int type;
	float (*forward)(float);
	float (*backward)(float);
};
typedef struct activation Activation;

const Activation *_act(int type);

#endif

