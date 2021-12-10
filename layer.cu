#include "layer.h"


Layer::Layer(int A, int B, int C)
{
	this->A = A;
	this->B = B;
	this->C = C;

	float h_weight[B][A];
	float h_bias[B];
	
	Output = NULL;	
	Bias   = NULL;
	Preact = NULL;
	Weight = NULL;

	for (int i = 0; i < B; i++)
	{
		h_bias[i] = 0.5f - float(rand()) / float(RAND_MAX);

		for (int j = 0; j < A; j++) 
		{
			h_weight[i][j] = 0.5f - float(rand()) / float(RAND_MAX);
		}
	}

	cudaMalloc(&Preact, sizeof(float) * C);
	cudaMalloc(&Output, sizeof(float) * C);

	cudaMalloc(&Weight, sizeof(float) * A * B);
	cudaMalloc(&Bias, sizeof(float) * B);

	cudaMalloc(&d_Preact, sizeof(float) * C);
	cudaMalloc(&d_Output, sizeof(float) * C);
	cudaMalloc(&d_Weight, sizeof(float) * A * B);

	cudaMemcpy(Bias, h_bias, sizeof(float) * B, cudaMemcpyHostToDevice);
	cudaMemcpy(Weight, h_weight, sizeof(float) * A * B, cudaMemcpyHostToDevice);
}

Layer::~Layer()
{
	cudaFree(Preact);
	cudaFree(Output);

	cudaFree(Weight);
	cudaFree(Bias);

	cudaFree(d_Preact);
	cudaFree(d_Output);
	cudaFree(d_Weight);
}


void Layer::setOutput(float *data)
{
	cudaMemcpy(Output, data, sizeof(float) * C, cudaMemcpyHostToDevice);
}

void Layer::clear()
{
	cudaMemset(Output, 0x00, sizeof(float) * C);
	cudaMemset(Preact, 0x00, sizeof(float) * C);
}

void Layer::bp_clear()
{
	cudaMemset(d_Output, 0x00, sizeof(float) * C);
	cudaMemset(d_Preact, 0x00, sizeof(float) * C);
	cudaMemset(d_Weight, 0x00, sizeof(float) * A * B);
}


__device__ float Step_function(float x)
{
	return 1 / (1 + exp(-x));
}

__global__ void Apply_Step_Function(float *Input, float *Output, const int B)
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = B * position / size; idx < B * (position +1) / size; ++idx)
	{
		Output[idx] = Step_function(Input[idx]);
	}
}

__global__ void MakeError(float *Error, float *Output, unsigned int Y, const int B)
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = B * position / size; idx < B * (position+1) / size; ++idx) {
		Error[idx] = ((Y == idx ? 1.0f : 0.0f) - Output[idx]);
	}
}

__global__ void forwardPass_preact_c1(float input[28][28], float Preact[6][24][24], float Weight[6][5][5])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 5*5*6*24*24;

	for (int x = B * position / size; x < B * (position+1) / size; x++)
	{
		int idx = x;
		const int m1 = ((idx /= 1	) % 5);
		const int m2 = ((idx /= 5	) % 5);
		const int m3 = ((idx /= 5	) % 6);
		const int m4 = ((idx /= 6	) % 24);
		const int m5 = ((idx /= 24	) % 24);

		atomicAdd(&Preact[m3][m4][m5], Weight[m3][m1][m2] * input[m4 + m1][m5 + m2]);
	}
}

__global__ void Apply_grad(float *Output, float *grad, const int B)
{
	const int position = threadIdx.x + blockIdx.x * blockDim.x;
	const int size = blockDim.x * gridDim.x;

	for (int idx = B * position / size; idx < B * (position+1) / size; idx++)
	{
		Output[idx] += dt * grad[idx];
	}
}

__global__ void forwardPass_bias_c1(float Preact[6][24][24], float Bias[6])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*24*24;

	for (int x = B * position / size; x < B * (position+1) / size; x++)
	{
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 24);
		const int m3 = ((idx /= 24	) % 24);

		Preact[m1][m2][m3] += Bias[m1];
	}
}

__global__ void forwardPass_preact_s1(float input[6][24][24], float Preact[6][6][6], float Weight[1][4][4])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 4*4*6*6*6;

	for (int x = B * position / size; x < B * (position+1) / size; x++)
	{
		int idx = x;
		const int m1 = ((idx /= 1	) % 4);
		const int m2 = ((idx /= 4	) % 4);
		const int m3 = ((idx /= 4	) % 6);
		const int m4 = ((idx /= 6	) % 6);
		const int m5 = ((idx /= 6	) % 6);

		atomicAdd(&Preact[m3][m4][m5], Weight[0][m1][m2] * input[m3][m4 * 4 + m1][m5 * 4 + m2]);
	}
}

__global__ void forwardPass_bias_s1(float Preact[6][6][6], float Bias[1])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*6*6;

	for (int x = B * position / size; x < B * (position +1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 6);
		const int m3 = ((idx /= 6	) % 6);

		Preact[m1][m2][m3] += Bias[0];
	}
}

__global__ void forwardPass_preact_f(float input[6][6][6], float Preact[10], float Weight[10][6][6][6])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 10*6*6*6;

	for (int x = B * position / size; x < B * (position +1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 10);
		const int m2 = ((idx /= 10	) % 6);
		const int m3 = ((idx /= 6	) % 6);
		const int m4 = ((idx /= 6	) % 6);

		atomicAdd(&Preact[m1], Weight[m1][m2][m3][m4] * input[m2][m3][m4]);
	}
}

__global__ void forwardPass_bias_f(float Preact[10], float Bias[10])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 10;

	for (int idx = B * position / size; idx < B * (position+1) / size; ++idx) 
	{
		Preact[idx] += Bias[idx];
	}
}

__global__ void backwardPass_bias_f(float Bias[10], float d_Preact[10])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 10;

	for (int idx = B * position / size; idx < B * (position+1) / size; ++idx)
	{
		Bias[idx] += dt * d_Preact[idx];
	}
}

__global__ void backwardPass_weight_f(float d_Weight[10][6][6][6], float d_Preact[10], float p_output[6][6][6])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 10*6*6*6;

	for (int x = B * position / size; x < B * (position +1) / size; x++)
	{
		int idx = x;
		const int m1 = ((idx /= 1	) % 10);
		const int m2 = ((idx /= 10	) % 6);
		const int m3 = ((idx /= 6	) % 6);
		const int m4 = ((idx /= 6	) % 6);

		d_Weight[m1][m2][m3][m4] = d_Preact[m1] * p_output[m2][m3][m4];
	}
}

__global__ void backwardPass_output_s1(float d_Output[6][6][6], float n_weight[10][6][6][6], float nd_preact[10])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 10*6*6*6;

	for (int x = B * position / size; x < B * (position +1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 10);
		const int m2 = ((idx /= 10	) % 6);
		const int m3 = ((idx /= 6	) % 6);
		const int m4 = ((idx /= 6	) % 6);

		atomicAdd(&d_Output[m2][m3][m4], n_weight[m1][m2][m3][m4] * nd_preact[m1]);
	}
}

__global__ void backwardPass_preact_s1(float d_Preact[6][6][6], float d_Output[6][6][6], float Preact[6][6][6])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*6*6;

	for (int x = B * position / size; x < B * (position +1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 6);
		const int m3 = ((idx /= 6	) % 6);

		const float a = Step_function(Preact[m1][m2][m3]);

		d_Preact[m1][m2][m3] = d_Output[m1][m2][m3] * a * (1 - a);
	}
}

__global__ void backwardPass_bias_s1(float Bias[1], float d_Preact[6][6][6])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int x = B * position / size; x < B * (position+1) / size; x++)
	{
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 6);
		const int m3 = ((idx /= 6	) % 6);

		atomicAdd(&Bias[0], dt * d_Preact[m1][m2][m3] / d);
	}
}


__global__ void backwardPass_weight_s1(float d_Weight[1][4][4], float d_Preact[6][6][6], float p_output[6][24][24])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 1*4*4*6*6*6;
	const float d = pow(6.0f, 3.0f);

	for (int x = B * position / size; x < B * (position + 1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 1);
		const int m2 = ((idx /= 1	) % 4);
		const int m3 = ((idx /= 4	) % 4);
		const int m4 = ((idx /= 4	) % 6);
		const int m5 = ((idx /= 6	) % 6);
		const int m6 = ((idx /= 6	) % 6);

		atomicAdd(&d_Weight[m1][m2][m3], d_Preact[m4][m5][m6] * p_output[m4][m5 * 4 + m2][m6 * 4 + m3]);
	}
}

__global__ void backwardPass_output_c1(float d_Output[6][24][24], float n_weight[1][4][4], float nd_preact[6][6][6])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 1*4*4*6*6*6;

	for (int x = B * position / size; x < B * (position+1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 1);
		const int m2 = ((idx /= 1	) % 4);
		const int m3 = ((idx /= 4	) % 4);
		const int m4 = ((idx /= 4	) % 6);
		const int m5 = ((idx /= 6	) % 6);
		const int m6 = ((idx /= 6	) % 6);

		atomicAdd(&d_Output[m4][m5 * 4 + m2][m6 * 4 + m3], n_weight[m1][m2][m3] * nd_preact[m4][m5][m6]);
	}
}

__global__ void backwardPass_preact_c1(float d_Preact[6][24][24], float d_Output[6][24][24], float Preact[6][24][24])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*24*24;

	for (int x = B * position / size; x < B * (position+1) / size; x++) 
	{
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 24);
		const int m3 = ((idx /= 24	) % 24);

		const float a = Step_function(Preact[m1][m2][m3]);

		d_Preact[m1][m2][m3] = d_Output[m1][m2][m3] * a * (1 - a);
	}
}

__global__ void backwardPass_bias_c1(float Bias[6], float d_Preact[6][24][24])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int x = B * position / size; x < B * (position+1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 24);
		const int m3 = ((idx /= 24	) % 24);

		atomicAdd(&Bias[m1], dt * d_Preact[m1][m2][m3] / d);
	}
}


__global__ void backwardPass_weight_c1(float d_Weight[6][5][5], float d_Preact[6][24][24], float p_output[28][28])
{
	const int position = blockIdx.x * blockDim.x + threadIdx.x;
	const int size = blockDim.x * gridDim.x;

	const int B = 6*5*5*24*24;
	const float d = pow(24.0f, 2.0f);

	for (int x = B * position / size; x < B * (position +1) / size; x++) {
		int idx = x;
		const int m1 = ((idx /= 1	) % 6);
		const int m2 = ((idx /= 6	) % 5);
		const int m3 = ((idx /= 5	) % 5);
		const int m4 = ((idx /= 5	) % 24);
		const int m5 = ((idx /= 24	) % 24);

		atomicAdd(&d_Weight[m1][m2][m3], d_Preact[m1][m4][m5] * p_output[m4 + m2][m5 + m3] / d);
	}
}
