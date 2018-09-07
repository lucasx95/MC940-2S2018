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
  iftBand   *weight;
  float      bias;
  int        nbands;
} MKernel;

MKernel *CreateMKernel(iftAdjRel *A, int nbands)
{
  MKernel* kernel = (MKernel*)iftAlloc(1,sizeof(MKernel));
  
  kernel->A      = iftCopyAdjacency(A);
  kernel->nbands = nbands;
  kernel->bias   = 0.0;
  kernel->weight = (iftBand*)iftAlloc(nbands,sizeof(iftBand));
  
  for(int b = 0; b < nbands; b++){
    kernel->weight[b].val = iftAllocFloatArray(A->n);
  }
  
  return kernel;
}

void DestroyMKernel(MKernel **K)
{
  MKernel *kernel = *K;

  for(int b = 0; b < kernel->nbands; b++)
    iftFree(kernel->weight[b].val);
  iftFree(kernel->weight);
  
  iftDestroyAdjRel(&kernel->A);
  
  iftFree(kernel);
  *K = NULL;
}

/* Read a 2D multi-band kernel */

MKernel *ReadMKernel(char *filename) 
{
  FILE *fp = fopen(filename, "r");
  iftAdjRel *A;
  int        nbands, xsize, ysize;
  MKernel *K; 

  fscanf(fp,"%d %d %d",&nbands, &xsize, &ysize);
  A      = iftRectangular(xsize,ysize);
  K      = CreateMKernel(A, nbands);
  for (int i=0; i < A->n; i++) { // read the weights
    for (int b=0; b < K->nbands; b++) { // for each band
      fscanf(fp,"%f",&K->weight[b].val[i]);
    }
  }
  fscanf(fp,"%f",&K->bias);
  
  fclose(fp);
  
  return(K);
}

/* Activation function known as Rectified Linear Unit (ReLu) */

iftMImage  *ReLu(iftMImage *mult_img)
{
  iftMImage *activ_img = iftCreateMImage(mult_img->xsize,mult_img->ysize,mult_img->zsize,mult_img->m);

  for (int p=0; p < mult_img->n; p++){
    for (int b=0; b < mult_img->m; b++)
      if (mult_img->band[b].val[p] > 0)
	activ_img->band[b].val[p] = mult_img->band[b].val[p];
  }
  
  return(activ_img);
}

/* This function is used to emphasize isolated (important)
   activations */

iftMImage  *DivisiveNormalization(iftMImage *mult_img, iftAdjRel *A)
{
  iftMImage *norm_img = iftCreateMImage(mult_img->xsize,mult_img->ysize,mult_img->zsize,mult_img->m);

  for (int p=0; p < mult_img->n; p++){
    float sum  = 0.0;
    iftVoxel u = iftMGetVoxelCoord(mult_img,p);
    for (int i=1; i < A->n; i++) { 
      iftVoxel v = iftGetAdjacentVoxel(A,u,i);
      if (iftMValidVoxel(mult_img,v)){ 
	int q = iftMGetVoxelIndex(mult_img,v);
	for (int b=0; b < mult_img->m; b++) { 
	  sum += mult_img->band[b].val[q]*mult_img->band[b].val[q];
	}
      }
    }
    sum = sqrtf(sum);
    if (sum > IFT_EPSILON){
      for (int b=0; b < mult_img->m; b++) {
	norm_img->band[b].val[p] = (mult_img->band[b].val[p]/sum);
      }
    }
  }
  
  return(norm_img);
}

/* Aggregate activations within a neighborhood (stride s = 1) */

iftMImage  *MaxPooling(iftMImage *mult_img, iftAdjRel *A)
{
  iftMImage *pool_img = iftCreateMImage(mult_img->xsize,mult_img->ysize,mult_img->zsize,mult_img->m);

  for (int p=0; p < mult_img->n; p++){
    iftVoxel u = iftMGetVoxelCoord(mult_img,p);
    for (int b=0; b < mult_img->m; b++) { 
      float max  = IFT_INFINITY_FLT_NEG;
      for (int i=0; i < A->n; i++) { 
	iftVoxel v = iftGetAdjacentVoxel(A,u,i);
	if (iftMValidVoxel(mult_img,v)){ 
	  int q = iftMGetVoxelIndex(mult_img,v);
	  if (mult_img->band[b].val[q] > max)
	    max = mult_img->band[b].val[q];
	}
      }
      pool_img->band[b].val[p] = max;
    }
  }
  
  return(pool_img);
}

iftMImage  *MinPooling(iftMImage *mult_img, iftAdjRel *A)
{
  iftMImage *pool_img = iftCreateMImage(mult_img->xsize,mult_img->ysize,mult_img->zsize,mult_img->m);

  for (int p=0; p < mult_img->n; p++){
    iftVoxel u = iftMGetVoxelCoord(mult_img,p);
    for (int b=0; b < mult_img->m; b++) { 
      float min  = IFT_INFINITY_FLT;
      for (int i=0; i < A->n; i++) { 
	iftVoxel v = iftGetAdjacentVoxel(A,u,i);
	if (iftMValidVoxel(mult_img,v)){ 
	  int q = iftMGetVoxelIndex(mult_img,v);
	  if (mult_img->band[b].val[q] < min)
	    min = mult_img->band[b].val[q];
	}
      }
      pool_img->band[b].val[p] = min;
    }
  }
  
  return(pool_img);
}

iftMImage *Convolution(iftMImage *mult_img, MKernel *K)
{
  iftMImage *filt_img=iftCreateMImage(mult_img->xsize,mult_img->ysize,mult_img->zsize,1); // multi-band image with one band
  
  for (int p=0; p < mult_img->n; p++) { // convolution
    filt_img->band[0].val[p]=0;
    iftVoxel u = iftMGetVoxelCoord(mult_img,p);
    for (int i=0; i < K->A->n; i++) { // for each adjacent voxel
      iftVoxel v = iftGetAdjacentVoxel(K->A,u,i);
      if (iftMValidVoxel(mult_img,v)){ // inside the image domain
	int q = iftMGetVoxelIndex(mult_img,v);
	for (int b=0; b < K->nbands; b++) { // for each band
	  filt_img->band[0].val[p] +=
	    K->weight[b].val[i]*mult_img->band[b].val[q];
	}	
      }
    }
    filt_img->band[0].val[p] += K->bias;
  }

  return(filt_img);
}

/* Read file kernel-bank.txt (see ExplicacaoArquivosKernels.txt) and
   return it in a iftMatrix */

iftMatrix *ReadMKernelBank(char *filename)
{

  return(NULL);
}

/* Extend a multi-band image to include the adjacent values in a same
   matrix */

iftMatrix *MImageToMatrix(iftMImage *mult_img, iftAdjRel *A)
{

  return(NULL);
}

/* Extend a multi-band image to include the adjacent values in a same
   image matrix */

iftMatrix *ConvolutionByMatrixMult(iftMatrix *Ximg, iftMatrix *W)
{


  return(NULL);
}

/* Convert an image matrix into a multi-band image */

iftMImage *MatrixToMImage(iftMatrix *Ximg)
{

  return(NULL);
}

int main(int argc, char *argv[]) 
{
  iftImage        *orig=NULL, *filt_img=NULL; // integer images
  iftMImage       *mult_img =NULL;  // multi-band image
  MKernel         *K=NULL; // multi-band kernel  
  timer           *tstart=NULL;

  if (argc!=4)
    iftError("LinearFilter <orig-image.[png, *]> <multi-band kernel.txt> <filtered-image.[png, *]>","main");

  orig = iftReadImageByExt(argv[1]);

  if (iftIs3DImage(orig)){
    iftExit("Image %s is not a 2D image","LinearFilter",argv[1]);
  } 

  if (iftIsColorImage(orig)){
    mult_img   = iftImageToMImage(orig,YCbCr_CSPACE);
  } else {
    mult_img   = iftImageToMImage(orig,GRAY_CSPACE);
  }

  tstart = iftTic();

  iftAdjRel *A;
  iftMImage *aux_mult_img;
  K                         = ReadMKernel(argv[2]);

  A         = iftCircular(15.0);
  aux_mult_img  = DivisiveNormalization(mult_img, A);
  iftDestroyAdjRel(&A);

  iftDestroyMImage(&mult_img);
  mult_img   = Convolution(aux_mult_img,K); /* it includes bias */
  iftDestroyMImage(&aux_mult_img);
  
  aux_mult_img                  = ReLu(mult_img); /* activation */
  iftDestroyMImage(&mult_img);

  A = iftRectangular(10,5);
  mult_img = MaxPooling(aux_mult_img, A);
  iftDestroyAdjRel(&A);
  iftDestroyMImage(&aux_mult_img);
  A            = iftRectangular(20,1);
  aux_mult_img     = MinPooling(mult_img, A);
  iftDestroyAdjRel(&A);
  filt_img  = iftMImageToImage(aux_mult_img,255,0); /* Extract band 0
						       normalized in
						       [0,255] */
			       
  iftWriteImageByExt(filt_img,argv[3]);
  
  puts("\nDone...");
  puts(iftFormattedTime(iftCompTime(tstart, iftToc())));

  DestroyMKernel(&K);
  iftDestroyImage(&orig);  
  iftDestroyImage(&filt_img);  
  iftDestroyMImage(&mult_img);  
  iftDestroyMImage(&aux_mult_img);  

  return(0);
}

