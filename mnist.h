#ifndef __MNIST_H__
#define __MNIST_H__


#ifdef USE_MNIST_LOADER 

#ifdef __cplusplus
extern "C" {
#endif

#ifdef MNIST_STATIC
#define _STATIC static
#else
#define _STATIC 
#endif


#ifdef MNIST_DOUBLE
#define MNIST_DATA_TYPE double
#else
#define MNIST_DATA_TYPE unsigned char
#endif

typedef struct mnist_data {
	MNIST_DATA_TYPE data[28][28];
	unsigned int label; 
} mnist_data;

#ifdef MNIST_HDR_ONLY

_STATIC int mnist_load(
	const char *imageFilename,
	const char *labelFilename,
	mnist_data **data,
	unsigned int *count);

#else

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static unsigned int mnist_bin_to_int(char *v)
{
	unsigned int rets = 0;
	int x;

	for (x = 0; x < 4; x++) 
	{
		rets <<= 8;
		rets |= (unsigned char)v[x];
	}
	return rets;
}

_STATIC int mnist_load(const char *imageFilename,const char *labelFilename,mnist_data **data,unsigned int *count)
{
	int return_code = 0;
	int x;
	char tmp[4];

	unsigned int image_cnt, label_cnt;
	unsigned int image_dim[2];

	FILE *ifp = fopen(imageFilename, "rb");
	FILE *lfp = fopen(labelFilename, "rb");

	if (!ifp || !lfp) {
		return_code = -1; 
		goto cleanup;
	}

	fread(tmp, 1, 4, ifp);
	if (mnist_bin_to_int(tmp) != 2051) {
		return_code = -2; 
		goto cleanup;
	}

	fread(tmp, 1, 4, lfp);
	if (mnist_bin_to_int(tmp) != 2049) {
		return_code = -3; 
		goto cleanup;
	}

	fread(tmp, 1, 4, ifp);
	image_cnt = mnist_bin_to_int(tmp);

	fread(tmp, 1, 4, lfp);
	label_cnt = mnist_bin_to_int(tmp);

	if (image_cnt != label_cnt) {
		return_code = -4; 
		goto cleanup;
	}

	for (x = 0; x < 2; x++) {
		fread(tmp, 1, 4, ifp);
		image_dim[x] = mnist_bin_to_int(tmp);
	}

	if (image_dim[0] != 28 || image_dim[1] != 28) {
		return_code = -2; 
		goto cleanup;
	}

	*count = image_cnt;
	*data = (mnist_data *)malloc(sizeof(mnist_data) * image_cnt);

	for (x = 0; x < image_cnt; x++) 
	{
		int y;
		unsigned char read_data[28 * 28];
		mnist_data *d = &(*data)[x];

		fread(read_data, 1, 28*28, ifp);

#ifdef MNIST_DOUBLE
		for (y = 0; y < 28*28; y++) 
		{
			d->data[y/28][y%28] = read_data[y] / 255.0;
		}
#else
		memcpy(d->data, read_data, 28*28);
#endif

		fread(tmp, 1, 1, lfp);
		d->label = tmp[0];
	}

cleanup:
	if (ifp) fclose(ifp);
	if (lfp) fclose(lfp);

	return return_code;
}

#endif 

#ifdef __cplusplus
}
#endif

#endif 
#endif

