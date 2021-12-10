#include <cstdlib>
#include <vector>
#include <memory>
#include <cublas_v2.h>
#include <cuda.h>

#ifndef LAYER_H
#define LAYER_H
#endif

const static float dt = 1.0E-01f;
const static float threshold = 1.0E-02f;

class Layer {
	public:
	int A, B, C;

	float *Output;
	float *Preact;

	float *Bias;
	float *Weight;

	float *d_Output;
	float *d_Preact;
	float *d_Weight;

	Layer(int A, int B, int C);

	~Layer();

	void setOutput(float *data);

	void bp_clear();

	void clear();
	
};


__device__ float Step_function(float v);
__global__ void Apply_Step_Function(float *input, float *Output, const int B);
__global__ void MakeError(float *Error, float *Output, unsigned int Y, const int B);
__global__ void Apply_grad(float *Output, float *grad, const int B);

__global__ void forwardPass_preact_c1(float input[28][28], float Preact[6][24][24], float Weight[6][5][5]);
__global__ void forwardPass_preact_s1(float input[6][24][24], float Preact[6][6][6], float Weight[1][4][4]);
__global__ void forwardPass_preact_f(float input[6][6][6], float Preact[10], float Weight[10][6][6][6]);
__global__ void forwardPass_bias_c1(float Preact[6][24][24], float Bias[6]);
__global__ void forwardPass_bias_s1(float Preact[6][6][6], float Bias[1]);
__global__ void forwardPass_bias_f(float Preact[10], float Bias[10]);

__global__ void backwardPass_weight_f(float d_Weight[10][6][6][6], float d_Preact[10], float p_output[6][6][6]);
__global__ void backwardPass_weight_s1(float d_Weight[1][4][4], float d_Preact[6][6][6], float p_output[6][24][24]);
__global__ void backwardPass_weight_c1(float d_Weight[6][5][5], float d_Preact[6][24][24], float p_output[28][28]);
__global__ void backwardPass_bias_f(float Bias[10], float d_Preact[10]);
__global__ void backwardPass_bias_s1(float Bias[1], float d_Preact[6][6][6]);
__global__ void backwardPass_bias_c1(float Bias[6], float d_Preact[6][24][24]);
__global__ void backwardPass_output_s1(float d_Output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10]);
__global__ void backwardPass_output_c1(float d_Output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6]);
__global__ void backwardPass_preact_s1(float d_Preact[6][6][6], float d_Output[6][6][6], float Preact[6][6][6]);
__global__ void backwardPass_preact_c1(float d_Preact[6][24][24], float d_Output[6][24][24], float Preact[6][24][24]);


