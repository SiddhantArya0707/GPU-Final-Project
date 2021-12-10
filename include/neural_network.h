#ifndef NEURAL_NETWORK_H_
#define NEURAL_NETWORK_H_

#include "mnist_file.h"

typedef struct neural_network_t_ {
    float b[MNIST_LABELS];
    float W[MNIST_LABELS][MNIST_IMAGE_SIZE];
} neural_network_t;

typedef struct neural_network_gradient_t_ {
    float b_grad[MNIST_LABELS];
    float W_grad[MNIST_LABELS][MNIST_IMAGE_SIZE];
} neural_network_gradient_t;

void Neural_Network_Random_Weights(neural_network_t * network);
void Neural_Network_Hypothesis_Calculation(mnist_image_t * image, neural_network_t * network, float activations[MNIST_LABELS]);
float Neural_Network_Gradient_Updates(mnist_image_t * image, neural_network_t * network, neural_network_gradient_t * gradient, uint8_t label);
float Neural_Network_Training_Steps(mnist_dataset_t * dataset, neural_network_t * network, float learning_rate);

#endif
