all:
	nvcc -lcuda -lcublas *.cu -o CNN  -Wno-deprecated-gpu-targets
	gcc mnist.c mnist_file.c neural_network.c -lm -o mnist	
run:
	./CNN
clean:
	rm CNN
