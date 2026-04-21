#ifndef INITIALIZER_H 
#define INITIALIZER_H 

enum {
	GLOROT = 0,
	HE
};

typedef struct initializer Init;

struct initializer {
	int type;
	float (*apply)(int in, int out);
};

const Init *initializer(int type);

#endif

