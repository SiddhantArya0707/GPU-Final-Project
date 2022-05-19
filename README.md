# cuDNN implementation of CNN
**Project Description**</br>
Convolutional Neural Network is a Deep Learning algorithm that takes in an input as images and appoints weights and predispositions to significant variables 
to recognize the components of the image. CNN utilizes an organization of neurons to foresee the image dependent on the trained data. 
The bigger the CNN organization the more exact it should accomplish including some different factors excessively, 
for example, input information size, streamlining agent being utilized.

CNN’s performance relies on the way the network calculations are done. If they are done in parallel, the output is faster to achieve. 
This is where cuDNN implementation arrives at the big picture. 

**This project is on cuDNN implementation on CNN and comparison of accuracy rate and time taken on CPU vs GPU**

**Project Structure**</br>
The architecture of the project is as follow-
-	mnist .c (CPU implementation file)
-	mnist_file.c (CPU implementation file) 
-	neural_network.c (CPU implementation file)
-	layer.cu (GPU implementation file) 
-	layer.h (GPU implementation file) 
-	main.cu (GPU implementation file) 
-	mnist.h (GPU implementation file)
-	/data (MNIST dataset)
-	/include(mnist and neural network header files for CPU implementation)


**How to install and run the project**
1.	Extract all the files from the project folder. 
2.	Run make command inside CuDNN folder. 
3.	Two files, CNN and mnist will be generated.
4.	Run - ./CNN for GPU implementation.
5.	Run- ./mnist for CPU implementation.

Please find the report and code for more details.


