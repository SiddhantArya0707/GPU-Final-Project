#define USE_MNIST_LOADER
#define MNIST_DOUBLE
#include "mnist.h"
#include "layer.h"

#include <cuda.h>
#include <cstdio>
#include <time.h>

static mnist_data *train_set, *test_set;
static unsigned int train_cnt, test_cnt;

static Layer layer_input = Layer(0, 0, 28*28);
static Layer layer_c1 = Layer(5*5, 6, 24*24*6);
static Layer layer_s1 = Layer(4*4, 1, 6*6*6);
static Layer layer_f = Layer(6*6*6, 10, 10);

static void Learn();
static unsigned int Classify(double data[28][28]);
static void Test();
static double forward_Pass(double data[28][28]);
static double backward_Pass();

static inline void loadData()
{
	mnist_load("data/train-images.idx3-ubyte", "data/train-labels.idx1-ubyte",
		&train_set, &train_cnt);
	mnist_load("data/t10k-images.idx3-ubyte", "data/t10k-labels.idx1-ubyte",
		&test_set, &test_cnt);
}

int main(int argc, const  char **argv)
{
	srand(time(NULL));

	CUresult Error = cuInit(0);
	if (Error != CUDA_SUCCESS) {
		fprintf(stderr, "CUDA initialization failed with the error code - %d\n", Error);
		return 1;
	}

	loadData();
	Learn();
	Test();
	return 0;
}

static double forward_Pass(double data[28][28])
{
	float inputData[28][28];

	for (int i = 0; i < 28; i++) 
	{
		for (int j = 0; j < 28; j++) 
		{
			inputData[i][j] = data[i][j];
		}
	}

	layer_input.clear();
	layer_c1.clear();
	layer_s1.clear();
	layer_f.clear();

	clock_t start, end;
	start = clock();

	layer_input.setOutput((float *)inputData);
	
	forwardPass_preact_c1<<<64, 64>>>((float (*)[28])layer_input.Output, (float (*)[24][24])layer_c1.Preact, (float (*)[5][5])layer_c1.Weight);
	forwardPass_bias_c1<<<64, 64>>>((float (*)[24][24])layer_c1.Preact, layer_c1.Bias);
	Apply_Step_Function<<<64, 64>>>(layer_c1.Preact, layer_c1.Output, layer_c1.C);

	forwardPass_preact_s1<<<64, 64>>>((float (*)[24][24])layer_c1.Output, (float (*)[6][6])layer_s1.Preact, (float (*)[4][4])layer_s1.Weight);
	forwardPass_bias_s1<<<64, 64>>>((float (*)[6][6])layer_s1.Preact, layer_s1.Bias);
	Apply_Step_Function<<<64, 64>>>(layer_s1.Preact, layer_s1.Output, layer_s1.C);

	forwardPass_preact_f<<<64, 64>>>((float (*)[6][6])layer_s1.Output, layer_f.Preact, (float (*)[6][6][6])layer_f.Weight);
	forwardPass_bias_f<<<64, 64>>>(layer_f.Preact, layer_f.Bias);
	Apply_Step_Function<<<64, 64>>>(layer_f.Preact, layer_f.Output, layer_f.C);
	
	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static double backward_Pass()
{
	clock_t start, end;

	start = clock();

	backwardPass_weight_f<<<64, 64>>>((float (*)[6][6][6])layer_f.d_Weight, layer_f.d_Preact, (float (*)[6][6])layer_s1.Output);
	backwardPass_bias_f<<<64, 64>>>(layer_f.Bias, layer_f.d_Preact);

	backwardPass_output_s1<<<64, 64>>>((float (*)[6][6])layer_s1.d_Output, (float (*)[6][6][6])layer_f.Weight, layer_f.d_Preact);
	backwardPass_preact_s1<<<64, 64>>>((float (*)[6][6])layer_s1.d_Preact, (float (*)[6][6])layer_s1.d_Output, (float (*)[6][6])layer_s1.Preact);
	backwardPass_weight_s1<<<64, 64>>>((float (*)[4][4])layer_s1.d_Weight, (float (*)[6][6])layer_s1.d_Preact, (float (*)[24][24])layer_c1.Output);
	backwardPass_bias_s1<<<64, 64>>>(layer_s1.Bias, (float (*)[6][6])layer_s1.d_Preact);

	backwardPass_output_c1<<<64, 64>>>((float (*)[24][24])layer_c1.d_Output, (float (*)[4][4])layer_s1.Weight, (float (*)[6][6])layer_s1.d_Preact);
	backwardPass_preact_c1<<<64, 64>>>((float (*)[24][24])layer_c1.d_Preact, (float (*)[24][24])layer_c1.d_Output, (float (*)[24][24])layer_c1.Preact);
	backwardPass_weight_c1<<<64, 64>>>((float (*)[5][5])layer_c1.d_Weight, (float (*)[24][24])layer_c1.d_Preact, (float (*)[28])layer_input.Output);
	backwardPass_bias_c1<<<64, 64>>>(layer_c1.Bias, (float (*)[24][24])layer_c1.d_Preact);


	Apply_grad<<<64, 64>>>(layer_f.Weight, layer_f.d_Weight, layer_f.A * layer_f.B);
	Apply_grad<<<64, 64>>>(layer_s1.Weight, layer_s1.d_Weight, layer_s1.A * layer_s1.B);
	Apply_grad<<<64, 64>>>(layer_c1.Weight, layer_c1.d_Weight, layer_c1.A * layer_c1.B);

	end = clock();
	return ((double) (end - start)) / CLOCKS_PER_SEC;
}

static void unfold_input(double inputData[28][28], double unfolded[24*24][5*5])
{
	int a = 0;
	(void)unfold_input;

	for (int i = 0; i < 2; i++)
		for (int j = 0; j < 2; j++) {
			int b = 0;
			for (int x = i; x < i + 2; x++)
				for (int y = j; y < j+2; y++)
					unfolded[a][b++] = inputData[x][y];
			a++;
		}
}

static void Learn()
{
	static cublasHandle_t blas;
	cublasCreate(&blas);

	float Error;
	int iteration = 50;
	
	double timeTaken = 0.0;

	fprintf(stdout ,"Training and Learning\n");

	while (iteration < 0 || iteration-- > 0) {
		Error = 0.0f;

		for (int i = 0; i < train_cnt; i++) {
			float tmp_err;

			timeTaken += forward_Pass(train_set[i].data);

			layer_f.bp_clear();
			layer_s1.bp_clear();
			layer_c1.bp_clear();

			MakeError<<<10, 1>>>(layer_f.d_Preact, layer_f.Output, train_set[i].label, 10);
			cublasSnrm2(blas, 10, layer_f.d_Preact, 1, &tmp_err);
			Error += tmp_err;

			timeTaken += backward_Pass();
		}

		Error /= train_cnt;
		fprintf(stdout, "Error: %e, time_on_gpu: %lf\n", Error, timeTaken);

		if (Error < threshold) {
			fprintf(stdout, "Training/Learning complete, Error less than threshold\n\n");
			break;
		}

	}
	
	fprintf(stdout, "\n Time Taken - %lf\n", timeTaken);
}

static unsigned int Classify(double data[28][28])
{
	float res[10];
	forward_Pass(data);
	unsigned int max = 0;
	cudaMemcpy(res, layer_f.Output, sizeof(float) * 10, cudaMemcpyDeviceToHost);

	for (int i = 1; i < 10; i++)
	{
		if (res[max] < res[i])
		{
			max = i;
		}
	}

	return max;
}

static void Test()
{
	int error = 0;
    int accu = 0;
	for (int i = 0; i < test_cnt; i++) 
	{
		if (Classify(test_set[i].data) != test_set[i].label) 
		{
			++error;
		}
	    else
	    {
		    ++accu;
	    }
	}

	fprintf(stdout, "Approx Error Rate: %.2lf%%\t Approx Accuracy Rate: %.2lf%%\n",
		double(error) / double(test_cnt) * 100.0, double(accu)/ double(test_cnt)*100.0);
}
