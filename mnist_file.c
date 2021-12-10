#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

#include "include/mnist_file.h"

uint32_t map_uint32(uint32_t in)
{
#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
    return (
        ((in & 0xFF000000) >> 24) |
        ((in & 0x00FF0000) >>  8) |
        ((in & 0x0000FF00) <<  8) |
        ((in & 0x000000FF) << 24)
    );
#else
    return in;
#endif
}

mnist_image_t * get_images(const char * path, uint32_t * number_of_images)
{
    FILE * streams;
    mnist_image_file_header_t header;
    mnist_image_t * images;

    streams = fopen(path, "rb");

    if (NULL == streams) {
        fprintf(stderr, "Was not able to open the file: %s\n", path);
        return NULL;
    }

    if (1 != fread(&header, sizeof(mnist_image_file_header_t), 1, streams)) {
        fprintf(stderr, "Was not able to read the image file header from: %s\n", path);
        fclose(streams);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.number_of_images = map_uint32(header.number_of_images);
    header.number_of_rows = map_uint32(header.number_of_rows);
    header.number_of_columns = map_uint32(header.number_of_columns);

    if (MNIST_IMAGE_MAGIC != header.magic_number) {
        fprintf(stderr, "Invalid header read from the image file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_IMAGE_MAGIC);
        fclose(streams);
        return NULL;
    }

    if (MNIST_IMAGE_WIDTH != header.number_of_rows) {
        fprintf(stderr, "Invalid number of image rows in the image file %s (%d not %d)\n", path, header.number_of_rows, MNIST_IMAGE_WIDTH);
    }

    if (MNIST_IMAGE_HEIGHT != header.number_of_columns) {
        fprintf(stderr, "Invalid number of image columns in the image file %s (%d not %d)\n", path, header.number_of_columns, MNIST_IMAGE_HEIGHT);
    }

    *number_of_images = header.number_of_images;
    images = malloc(*number_of_images * sizeof(mnist_image_t));

    if (images == NULL) {
        fprintf(stderr, "Was not able to allocate memory for %d images\n", *number_of_images);
        fclose(streams);
        return NULL;
    }

    if (*number_of_images != fread(images, sizeof(mnist_image_t), *number_of_images, streams)) {
        fprintf(stderr, "Was not able to read %d the images from: %s\n", *number_of_images, path);
        free(images);
        fclose(streams);
        return NULL;
    }

    fclose(streams);

    return images;
}

uint8_t * get_labels(const char * path, uint32_t * numberLabels)
{
    FILE * streams;
    mnist_label_file_header_t header;
    uint8_t * labels;

    streams = fopen(path, "rb");

    if (NULL == streams) {
        fprintf(stderr, "Was not able to open the file: %s\n", path);
        return NULL;
    }

    if (1 != fread(&header, sizeof(mnist_label_file_header_t), 1, streams)) {
        fprintf(stderr, "Was not able to read label file header from: %s\n", path);
        fclose(streams);
        return NULL;
    }

    header.magic_number = map_uint32(header.magic_number);
    header.numberLabels = map_uint32(header.numberLabels);

    if (MNIST_LABEL_MAGIC != header.magic_number) {
        fprintf(stderr, "Invalid header read from the label file: %s (%08X not %08X)\n", path, header.magic_number, MNIST_LABEL_MAGIC);
        fclose(streams);
        return NULL;
    }

    *numberLabels = header.numberLabels;

    labels = malloc(*numberLabels * sizeof(uint8_t));

    if (labels == NULL) {
        fprintf(stderr, "Was not able to allocate memory for %d labels\n", *numberLabels);
        fclose(streams);
        return NULL;
    }

    if (*numberLabels != fread(labels, 1, *numberLabels, streams)) {
        fprintf(stderr, "Was not able to read %d the labels from: %s\n", *numberLabels, path);
        free(labels);
        fclose(streams);
        return NULL;
    }

    fclose(streams);

    return labels;
}

mnist_dataset_t * mnistGetDataset(const char * imagePath, const char * labelPath)
{
    mnist_dataset_t * dataset;
    uint32_t number_of_images, numberLabels;

    dataset = calloc(1, sizeof(mnist_dataset_t));

    if (NULL == dataset) {
        return NULL;
    }

    dataset->images = get_images(imagePath, &number_of_images);

    if (NULL == dataset->images) {
        mnistFreeDataset(dataset);
        return NULL;
    }

    dataset->labels = get_labels(labelPath, &numberLabels);

    if (NULL == dataset->labels) {
        mnistFreeDataset(dataset);
        return NULL;
    }

    if (number_of_images != numberLabels) {
        fprintf(stderr, "Number of images does not match with the number of labels (%d != %d)\n", number_of_images, numberLabels);
        mnistFreeDataset(dataset);
        return NULL;
    }
    dataset->size = number_of_images;
    return dataset;
}

int mnistBatch(mnist_dataset_t * dataset, mnist_dataset_t * batch, int size, int number)
{
    int start_offset;

    start_offset = size * number;

    if (start_offset >= dataset->size) {
        return 0;
    }

    batch->images = &dataset->images[start_offset];
    batch->labels = &dataset->labels[start_offset];
    batch->size = size;

    if (start_offset + batch->size > dataset->size) {
        batch->size = dataset->size - start_offset;
    }

    return 1;
}

void mnistFreeDataset(mnist_dataset_t * dataset)
{
    free(dataset->images);
    free(dataset->labels);
    free(dataset);
}
