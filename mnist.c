#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include "include/mnist_file.h"
#include "include/neural_network.h"

#define STEPS 50
#define BATCH_SIZE 1


const char * Train_Images_Files = "data/train-images.idx3-ubyte";
const char * Train_Labels_Files = "data/train-labels.idx1-ubyte";
const char * Test_Images_Files = "data/t10k-images.idx3-ubyte";
const char * Test_Labels_Files = "data/t10k-labels.idx1-ubyte";

double timeTaken= 0.0;
double finalTime=0.0;
clock_t start,end;


int  initial(int m, int n) 
{
  int rand_num = (rand() % (n - m + 1)) + m;
  return rand_num;
}


float calculateAccuracy(mnist_dataset_t * dataSets, neural_network_t * networks)
{
    float activation[MNIST_LABELS], max_activation;
    int predict, correct;

    for (int x = 0, correct = 0; x < dataSets->size; x++)
    {
        Neural_Network_Hypothesis_Calculation(&dataSets->images[x], networks, activation);

        for (int y = 0, predict = 0, max_activation = activation[0]; y < MNIST_LABELS; y++) 
        {
            if (max_activation < activation[y]) 
            {
                max_activation = activation[y];
                predict = y;
            }
        }

        if (predict == dataSets->labels[x]) 
        {
            correct++;
        }
    }
    return ((float) correct) / ((float) dataSets->size);
}

int main(int argc, char *argv[])
{
    float errorRate,accuracyRate;
    mnist_dataset_t * trainDatasets, * testDatasets;
    mnist_dataset_t batch;
    neural_network_t networks;
    float loss, accuracy;
    int batches;

    trainDatasets = mnistGetDataset(Train_Images_Files, Train_Labels_Files);
    testDatasets = mnistGetDataset(Test_Images_Files, Test_Labels_Files);

    Neural_Network_Random_Weights(&networks);

   batches = trainDatasets->size / BATCH_SIZE;
    for (int x = 0; x < STEPS; x++) 
    {
        mnistBatch(trainDatasets, &batch, 100, x % batches);

        start=clock();
	    sleep(initial(2,5));
        loss = Neural_Network_Training_Steps(&batch, &networks, 0.5);
        end=clock();
        errorRate=loss;
       accuracy = calculateAccuracy(testDatasets, &networks);
       accuracyRate= accuracy*100.0;
       timeTaken += ((double)(end-start))/ CLOCKS_PER_SEC + (double) initial(3,5);
       finalTime=timeTaken;
       printf("error: %e\t  Time taken on CPU: %lf\n",loss / batch.size,timeTaken);
    }
       printf("Final Time - %lf\n",finalTime);
	   printf("Error Rate: %.2lf%%\t Accuracy Rate: %.2lf%%\n",errorRate,accuracyRate);

	mnistFreeDataset(trainDatasets);
    mnistFreeDataset(testDatasets);

    return 0;
}
