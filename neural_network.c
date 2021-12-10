#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "include/mnist_file.h"
#include "include/neural_network.h"

#define PIXEL_SCALE(x) (((float) (x)) / 255.0f)

#define RAND_FLOAT() (((float) rand()) / ((float) RAND_MAX))


void Neural_Network_Hypothesis_Calculation(mnist_image_t * images, neural_network_t * networks, float activations[MNIST_LABELS])
{
    for (int x = 0; x < MNIST_LABELS; x++) 
    {
        activations[x] = networks->b[x];

        for (int y = 0; y < MNIST_IMAGE_SIZE; y++) 
        {
            activations[x] += networks->W[x][y] * PIXEL_SCALE(images->pixels[y]);
        }
    }

    Neural_Network_Softmax_Calculation(activations, MNIST_LABELS);
}

void Neural_Network_Random_Weights(neural_network_t * networkData)
{
    for (int x = 0; x < MNIST_LABELS; x++) 
    {
        networkData->b[x] = RAND_FLOAT();

        for (int y = 0; y < MNIST_IMAGE_SIZE; y++)
        {
            networkData->W[x][y] = RAND_FLOAT();
        }
    }
}

void Neural_Network_Softmax_Calculation(float * activation, int length)
{
    float sum, max;

    for (int x = 1, max = activation[0]; x < length; x++)
    {
        if (activation[x] > max) 
        {
            max = activation[x];
        }
    }

    for (int x = 0, sum = 0; x < length; x++) 
    {
        activation[x] = exp(activation[x] - max);
        sum += activation[x];
    }

    for (int x = 0; x < length; x++) 
    {
        activation[x] /= sum;
    }
}

float Neural_Network_Training_Steps(mnist_dataset_t * dataSets, neural_network_t * networks, float learningRate)
{
    neural_network_gradient_t gradient;
    float totalLoss;

    memset(&gradient, 0, sizeof(neural_network_gradient_t));

    for (int x = 0, totalLoss = 0; x < dataSets->size; x++) 
    {
        totalLoss += Neural_Network_Gradient_Updates(&dataSets->images[x], networks, &gradient, dataSets->labels[x]);
    }

    for (int x = 0; x < MNIST_LABELS; x++) 
    {
        networks->b[x] -= learningRate * gradient.b_grad[x] / ((float) dataSets->size);

        for (int y = 0; y < MNIST_IMAGE_SIZE + 1; y++) 
        {
            networks->W[x][y] -= learningRate * gradient.W_grad[x][y] / ((float) dataSets->size);
        }
    }

    return totalLoss;
}

float Neural_Network_Gradient_Updates(mnist_image_t * images, neural_network_t * networks, neural_network_gradient_t * gradients, uint8_t labels)
{
    float b_grad, W_grad;
    float activations[MNIST_LABELS];

    Neural_Network_Hypothesis_Calculation(images, networks, activations);

    for (int x = 0; x < MNIST_LABELS; x++) 
    {
        b_grad = (x == labels) ? activations[x] - 1 : activations[x];

        for (int y = 0; y < MNIST_IMAGE_SIZE; y++) 
        {
            W_grad = b_grad * PIXEL_SCALE(images->pixels[y]);

            gradients->W_grad[x][y] += W_grad;
        }
        gradients->b_grad[x] += b_grad;
    }

    return 0.0f - log(activations[labels]);
}
