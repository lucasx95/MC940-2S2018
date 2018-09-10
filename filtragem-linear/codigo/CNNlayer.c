#include "ift.h"

/* This code exploits the concept of linear filtering to illustrate
   the main operations in a convolutional neural network. Operations
   based on adjacency relation usually reduce the image size by
   disconsidering pixels to which there are adjacent pixels outside
   the image domain. The max-pooling operation also has a parameter
   called stride s >= 1, which can be used to reduce the image domain
   by subsampling pixels with displacement s. For instance, for s = 2
   the image domain is reduced to half. We will simplify their
   implementations by avoiding image reduction. Your task will be the
   extension of this code to read multiple kernels from a same file
   and implement the convolution between the multi-band image and the
   kernel bank by matrix multiplication (see iftMatrix.h in
   ./include). Examples of kernel and kernel bank are given in
   kernel.txt and kernel-bank.txt. Read ExplicacaoArquivosKernels.txt
   to understand their content. */

typedef struct mkernel {
    iftAdjRel *A;
    iftBand *weight;
    float bias;
    int nbands;
} MKernel;

MKernel *CreateMKernel(iftAdjRel *A, int nbands) {
    MKernel *kernel = (MKernel *) iftAlloc(1, sizeof(MKernel));

    kernel->A = iftCopyAdjacency(A);
    kernel->nbands = nbands;
    kernel->bias = 0.0;
    kernel->weight = (iftBand *) iftAlloc(nbands, sizeof(iftBand));

    for (int b = 0; b < nbands; b++) {
        kernel->weight[b].val = iftAllocFloatArray(A->n);
    }

    return kernel;
}

iftMatrix *CreateMMatrix(int n_columns, int n_rows) {
    iftMatrix *kernel_matrix = (iftMatrix *) iftAlloc(1, sizeof(iftMatrix));

    kernel_matrix->ncols = n_columns;
    kernel_matrix->nrows = n_rows;
    kernel_matrix->n = n_columns * n_rows;
    kernel_matrix->allocated = true;

    kernel_matrix->val = iftAllocFloatArray(kernel_matrix->n);

    return kernel_matrix;
}

void DestroyMKernel(MKernel **K) {
    MKernel *kernel = *K;

    for (int b = 0; b < kernel->nbands; b++)
        iftFree(kernel->weight[b].val);
    iftFree(kernel->weight);

    iftDestroyAdjRel(&kernel->A);

    iftFree(kernel);
    *K = NULL;
}

/* Read a 2D multi-band kernel */

MKernel *ReadMKernel(char *filename) {
    FILE *fp = fopen(filename, "r");
    iftAdjRel *A;
    int nbands, xsize, ysize;
    MKernel *K;

    fscanf(fp, "%d %d %d", &nbands, &xsize, &ysize);
    A = iftRectangular(xsize, ysize);
    K = CreateMKernel(A, nbands);
    for (int i = 0; i < A->n; i++) { // read the weights
        for (int b = 0; b < K->nbands; b++) { // for each band
            fscanf(fp, "%f", &K->weight[b].val[i]);
        }
    }
    fscanf(fp, "%f", &K->bias);

    fclose(fp);

    return (K);
}

/* Activation function known as Rectified Linear Unit (ReLu) */

iftMImage *ReLu(iftMImage *mult_img) {
    iftMImage *activ_img = iftCreateMImage(mult_img->xsize, mult_img->ysize, mult_img->zsize, mult_img->m);

    for (int p = 0; p < mult_img->n; p++) {
        for (int b = 0; b < mult_img->m; b++)
            if (mult_img->band[b].val[p] > 0)
                activ_img->band[b].val[p] = mult_img->band[b].val[p];
    }

    return (activ_img);
}

/* This function is used to emphasize isolated (important)
   activations */

iftMImage *DivisiveNormalization(iftMImage *mult_img, iftAdjRel *A) {
    iftMImage *norm_img = iftCreateMImage(mult_img->xsize, mult_img->ysize, mult_img->zsize, mult_img->m);

    for (int p = 0; p < mult_img->n; p++) {
        float sum = 0.0;
        iftVoxel u = iftMGetVoxelCoord(mult_img, p);
        for (int i = 1; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            if (iftMValidVoxel(mult_img, v)) {
                int q = iftMGetVoxelIndex(mult_img, v);
                for (int b = 0; b < mult_img->m; b++) {
                    sum += mult_img->band[b].val[q] * mult_img->band[b].val[q];
                }
            }
        }
        sum = sqrtf(sum);
        if (sum > IFT_EPSILON) {
            for (int b = 0; b < mult_img->m; b++) {
                norm_img->band[b].val[p] = (mult_img->band[b].val[p] / sum);
            }
        }
    }

    return (norm_img);
}

/* Aggregate activations within a neighborhood (stride s = 1) */

iftMImage *MaxPooling(iftMImage *mult_img, iftAdjRel *A) {
    iftMImage *pool_img = iftCreateMImage(mult_img->xsize, mult_img->ysize, mult_img->zsize, mult_img->m);

    for (int p = 0; p < mult_img->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mult_img, p);
        for (int b = 0; b < mult_img->m; b++) {
            float max = IFT_INFINITY_FLT_NEG;
            for (int i = 0; i < A->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                if (iftMValidVoxel(mult_img, v)) {
                    int q = iftMGetVoxelIndex(mult_img, v);
                    if (mult_img->band[b].val[q] > max)
                        max = mult_img->band[b].val[q];
                }
            }
            pool_img->band[b].val[p] = max;
        }
    }

    return (pool_img);
}

iftMImage *MinPooling(iftMImage *mult_img, iftAdjRel *A) {
    iftMImage *pool_img = iftCreateMImage(mult_img->xsize, mult_img->ysize, mult_img->zsize, mult_img->m);

    for (int p = 0; p < mult_img->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mult_img, p);
        for (int b = 0; b < mult_img->m; b++) {
            float min = IFT_INFINITY_FLT;
            for (int i = 0; i < A->n; i++) {
                iftVoxel v = iftGetAdjacentVoxel(A, u, i);
                if (iftMValidVoxel(mult_img, v)) {
                    int q = iftMGetVoxelIndex(mult_img, v);
                    if (mult_img->band[b].val[q] < min)
                        min = mult_img->band[b].val[q];
                }
            }
            pool_img->band[b].val[p] = min;
        }
    }

    return (pool_img);
}

iftMImage *Convolution(iftMImage *mult_img, MKernel *K) {
    iftMImage *filt_img = iftCreateMImage(mult_img->xsize,
                                          mult_img->ysize,
                                          mult_img->zsize,
                                          1); // multi-band image with one band

    for (int p = 0; p < mult_img->n; p++) { // convolution
        filt_img->band[0].val[p] = 0;
        iftVoxel u = iftMGetVoxelCoord(mult_img, p);
        for (int i = 0; i < K->A->n; i++) { // for each adjacent voxel
            iftVoxel v = iftGetAdjacentVoxel(K->A, u, i);
            if (iftMValidVoxel(mult_img, v)) { // inside the image domain
                int q = iftMGetVoxelIndex(mult_img, v);
                for (int b = 0; b < K->nbands; b++) { // for each band
                    filt_img->band[0].val[p] +=
                            K->weight[b].val[i] * mult_img->band[b].val[q];
                }
            }
        }
        filt_img->band[0].val[p] += K->bias;
    }

    return (filt_img);
}

/* Read file kernel-bank.txt (see ExplicacaoArquivosKernels.txt) and
   return it in a iftMatrix */

iftMatrix *ReadMKernelBank(char *filename, int *xsizeref, int *ysizeref) {

    FILE *fp = fopen(filename, "r");
    int nbands, xsize, ysize, nkernels;
    iftMatrix *kernel_matrix;

    fscanf(fp, "%d %d %d %d", &nbands, &xsize, &ysize, &nkernels);

    *xsizeref = xsize;
    *ysizeref = ysize;

    int n_columns = nbands * ysize * xsize + 1; // Number of weights per kernel with the bias.

    kernel_matrix = CreateMMatrix(n_columns, nkernels);

    kernel_matrix->ncols = n_columns;
    kernel_matrix->nrows = nkernels;
    kernel_matrix->n = kernel_matrix->ncols * kernel_matrix->nrows;
    for (int i = 0; i < kernel_matrix->n; ++i) { // For each kernel/ Matrix line
        fscanf(fp, "%f", &kernel_matrix->val[i]);
    }

    fclose(fp);
    return kernel_matrix;
}

/* Extend a multi-band image to include the adjacent values in a same
   matrix */
iftMatrix *MImageToMatrix(iftMImage *mult_img, iftAdjRel *A) {
    int n_columns = mult_img->n;
    int n_rows = (A->n * mult_img->m) + 1;
    iftMatrix *image_matrix = CreateMMatrix(n_columns, n_rows);

    for (int p = 0; p < mult_img->n; p++) {
        iftVoxel u = iftMGetVoxelCoord(mult_img, p);
        for (int i = 0; i < A->n; i++) {
            iftVoxel v = iftGetAdjacentVoxel(A, u, i);
            int matrix_position = i * mult_img->m * mult_img->n + p;
            // If it's not a valid position put 0.
            if (!iftMValidVoxel(mult_img, v)) {
                for (int j = 0; j < mult_img->m; ++j) {
                    image_matrix->val[matrix_position] = 0.0;
                    matrix_position += mult_img->n;
                }
                image_matrix->val[matrix_position] = 1.0; // Include columns to be multiplied by the bias.
                continue;
            }
            int q = iftMGetVoxelIndex(mult_img, v);
            for (int j = 0; j < mult_img->m; ++j) {
                image_matrix->val[matrix_position] = mult_img->band[j].val[q];
                matrix_position += mult_img->n;
            }
            image_matrix->val[matrix_position] = 1.0; // Include columns to be multiplied by the bias.
        }
    }
    return image_matrix;
}

/* Extend a multi-band image to include the adjacent values in a same
   image matrix */

iftMatrix *ConvolutionByMatrixMult(iftMatrix *Ximg, iftMatrix *W) {
    return iftMultMatrices(W, Ximg);
}


iftMImage *MatrixToMImage(iftMatrix *Ximg, int xsize, int ysize) {
    return iftMatrixToMImage(Ximg, xsize, ysize, 1, Ximg->nrows);
}

void PrintKernelMMatrix(iftMatrix *ift_matrix) {
    for (int j = 0; j < ift_matrix->n; ++j) {
        if (j % ift_matrix->ncols == 0) {
            printf("\n");
        }
        printf("%.3f ", ift_matrix->val[j]);
    }
    printf("\n");
}

iftMImage *ImageFromInput() {
    printf("\nReading test image from input. Enter nbands, xsize and ysize: ");
    int xsize, ysize, nbands;
    scanf("%d %d %d", &nbands, &xsize, &ysize);

    iftMImage *image = calloc(1, sizeof(iftMImage));
    image->m = nbands;
    image->n = xsize * ysize;
    image->xsize = xsize;
    image->ysize = ysize;
    image->zsize = 1;
    image->dx = xsize;
    image->dy = ysize;
    image->dz = 1;
    image->band = calloc((size_t) nbands, sizeof(iftBand));
    for (int i = 0; i < nbands; ++i) {
        image->band[i].val = iftAllocFloatArray(image->n);
        for (int j = 0; j < image->n; ++j) {
            scanf("%f", &image->band[i].val[j]);
        }
    }
    return image;
}

int main(int argc, char *argv[]) {
    iftImage *orig = NULL, *filt_img = NULL; // integer images
    iftMImage *mult_img = NULL;  // multi-band image
    MKernel *K = NULL; // multi-band kernel
    timer *tstart = NULL;

    if (argc != 4)
        iftError("LinearFilter <orig-image.[png, *]> <multi-band kernel.txt> <filtered-image.[png, *]>", "main");

    orig = iftReadImageByExt(argv[1]);

    if (iftIs3DImage(orig)) {
        iftExit("Image %s is not a 2D image", "LinearFilter", argv[1]);
    }

    if (iftIsColorImage(orig)) {
        mult_img = iftImageToMImage(orig, YCbCr_CSPACE);
    } else {
        mult_img = iftImageToMImage(orig, GRAY_CSPACE);
    }

    tstart = iftTic();


    iftMImage *aux_mult_img;
    iftAdjRel *A;

    A = iftCircular(15.0);
    aux_mult_img = DivisiveNormalization(mult_img, A);
    iftDestroyAdjRel(&A);

    iftDestroyMImage(&mult_img);
    int *kernelx = calloc(1, sizeof(int)), *kernely = calloc(1,
                                                             sizeof(int)), imagex = aux_mult_img->xsize, imagey = aux_mult_img->ysize;

    iftMatrix *kernel_matrix = ReadMKernelBank(argv[2], kernelx, kernely);
    A = iftRectangular(*kernelx, *kernely);
    iftMatrix *image_matrix = MImageToMatrix(aux_mult_img, A);
    iftDestroyMImage(&aux_mult_img);
    iftMatrix *convulation_matrix = ConvolutionByMatrixMult(image_matrix, kernel_matrix);
    mult_img = MatrixToMImage(convulation_matrix, imagex, imagey);
    aux_mult_img = ReLu(mult_img); /* activation */
    iftDestroyMImage(&mult_img);
    iftDestroyMatrix(&kernel_matrix);
    iftDestroyMatrix(&image_matrix);
    iftDestroyMatrix(&convulation_matrix);
    A = iftRectangular(10, 5);
    mult_img = MaxPooling(aux_mult_img, A);
    iftDestroyAdjRel(&A);
    iftDestroyMImage(&aux_mult_img);
    A = iftRectangular(20, 1);
    aux_mult_img = MinPooling(mult_img, A);
    iftDestroyAdjRel(&A);
    filt_img = iftMImageToImage(aux_mult_img, 255, 0); /* Extract band 0
						       normalized in
						       [0,255] */

    iftWriteImageByExt(filt_img, argv[3]);

    puts("\nDone...");
    puts(iftFormattedTime(iftCompTime(tstart, iftToc())));

    DestroyMKernel(&K);
    iftDestroyImage(&orig);
    iftDestroyImage(&filt_img);
    iftDestroyMImage(&mult_img);
    iftDestroyMImage(&aux_mult_img);

    return (0);
}

