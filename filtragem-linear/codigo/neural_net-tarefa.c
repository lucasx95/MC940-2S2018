typedef struct mkernel { /* multiband kernel */
  iftAdjRel *A;          /* adjacency relation */
  iftBand   *weight;     /* weights of the kernel */
  float      bias;       /* bias */
  int        nbands;     /* number of bands */
} MKernel;

typedef struct mkernelbank { /* kernel bank */
  MKernel **K;         /* a vetor of multiband kernels */         
  int       nkernels;  /* number of kernels */
} MKernelBank;

typedef struct net_parameters { /* parameters of the system */
  float *weight;    /* weight of each band (kernel) */
  int    nkernels;  /* number of kernels */
  float  threshold; /* final threshold */ 
  float *maxactiv;  /* maximum activation for normalization per kernel */
  iftBoundingBox bb; /* region in which the training plates are found */
  float  mean_width, mean_height; /* mean width and height of the plates */
} NetParameters;

NetParameters *CreateNetParameters(int nkernels)
{
  NetParameters *nparam=(NetParameters *)calloc(1,sizeof(NetParameters));
  nparam->weight       = iftAllocFloatArray(nkernels);
  nparam->maxactiv     = iftAllocFloatArray(nkernels);
  nparam->nkernels     = nkernels;
  nparam->threshold    = 0.0;
  nparam->mean_width   = nparam->mean_height = 0;
  return(nparam);
}

void DestroyNetParameters(NetParameters **nparam)
{
  NetParameters *aux = *nparam;
  if (aux != NULL){
    if (aux->weight != NULL)   iftFree(aux->weight);
    if (aux->maxactiv != NULL) iftFree(aux->maxactiv);
    iftFree(aux);
    *nparam = NULL;
  }
}

NetParameters *ReadNetParameters(char *filename)
{
  FILE *fp = fopen(filename,"r");
  int nkernels;
  
  fscanf(fp,"%d",&nkernels);
  NetParameters *nparam = CreateNetParameters(nkernels);
  fscanf(fp,"%f %d %d %d %d %f %f\n",&nparam->threshold,&nparam->bb.begin.x,&nparam->bb.begin.y,&nparam->bb.end.x,&nparam->bb.end.y,&nparam->mean_width,&nparam->mean_height);

  for (int i=0; i < nkernels; i++)
    fscanf(fp,"%f",&nparam->weight[i]);
  for (int i=0; i < nkernels; i++)
    fscanf(fp,"%f",&nparam->maxactiv[i]);
  fclose(fp);

  return(nparam);
}

void WriteNetParameters(NetParameters *nparam, char *filename)
{
  FILE *fp = fopen(filename,"w");
  
  fprintf(fp,"%d\n",nparam->nkernels);
  fprintf(fp,"%f %d %d %d %d %f %f\n",nparam->threshold,nparam->bb.begin.x,nparam->bb.begin.y,nparam->bb.end.x,nparam->bb.end.y,nparam->mean_width,nparam->mean_height);
  for (int i=0; i < nparam->nkernels; i++)
    fprintf(fp,"%f ",nparam->weight[i]);
  fprintf(fp,"\n");
  for (int i=0; i < nparam->nkernels; i++)
    fprintf(fp,"%f ",nparam->maxactiv[i]);
  fprintf(fp,"\n");
  
  fclose(fp);
}


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

/* Read multi-band kernel bank */

MKernelBank *ReadMKernelBank(char *filename) 
{
  FILE        *fp = fopen(filename, "r");
  iftAdjRel   *A;
  int          nbands, xsize, ysize, nkernels;
  MKernelBank *Kbank = (MKernelBank *)calloc(1,sizeof(MKernelBank)); 

  fscanf(fp,"%d %d %d %d",&nbands, &xsize, &ysize, &nkernels);

  A  = iftRectangular(xsize,ysize);
  if (A->n != xsize*ysize)
    iftError("Define kernels with odd dimensions (e.g., 3 x 5)","ReadMKernelBank");
  
  Kbank->K = (MKernel **)calloc(nkernels,sizeof(MKernel *));
  Kbank->nkernels = nkernels;

  for (int k=0; k < nkernels; k++){ 
    Kbank->K[k] = CreateMKernel(A, nbands);

    for (int i=0; i < A->n; i++) { // read the weights
      for (int b=0; b < Kbank->K[0]->nbands; b++) { // for each band
	fscanf(fp,"%f",&Kbank->K[k]->weight[b].val[i]);
      }
    }
    fscanf(fp,"%f",&Kbank->K[k]->bias);
  }
  
  fclose(fp);
  
  return(Kbank);
}

void DestroyMKernelBank(MKernelBank **Kbank)
{
  MKernelBank *aux = *Kbank;
  for (int k=0; k < aux->nkernels; k++)
    DestroyMKernel(&aux->K[k]);
  free(aux);
  *Kbank = NULL;
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

iftMImage *SingleLayer(iftImage *img, MKernelBank *Kbank)
{
  iftMImage *out = iftCreateMImage(img->xsize,img->ysize,img->zsize,Kbank->nkernels);
  
  iftAdjRel *A[2];
  iftMImage *aux[2], *mimg;

  if (iftIsColorImage(img)){
    mimg   = iftImageToMImage(img,YCbCr_CSPACE);
  } else {
    mimg   = iftImageToMImage(img,GRAY_CSPACE);
  }
  
  A[0]             = iftRectangular(7,3);
  A[1]             = iftRectangular(9,9);

  for (int k=0; k < Kbank->nkernels; k++) {

    aux[0]        = Convolution(mimg,Kbank->K[k]);   
  
    aux[1]        = ReLu(aux[0]); /* activation */
    iftDestroyMImage(&aux[0]);

    aux[0]        = MaxPooling(aux[1], A[0]);
    iftDestroyMImage(&aux[1]);

    aux[1]        = MinPooling(aux[0], A[1]);
    iftDestroyMImage(&aux[0]);

    
    for (int p = 0; p < out->n; p++){
      out->band[k].val[p] = aux[1]->band[0].val[p];
    }
    
    iftDestroyMImage(&aux[1]);
  }

  for (int i = 0; i < 2; i++)
    iftDestroyAdjRel(&A[i]);
  
  return(out);
}

void ComputeAspectRatioParameters(iftImage **mask, int nimages, NetParameters *nparam)
{
  nparam->mean_width  = 0.0;
  nparam->mean_height = 0.0;
  
  for (int i=0; i < nimages; i++){ /* For each image */
    iftVoxel pos;
    iftBoundingBox bb  = iftMinBoundingBox(mask[i],&pos);
    float width        = (float)(bb.end.x - bb.begin.x);
    float height       = (float)(bb.end.y - bb.begin.y);
    
    nparam->mean_width   += width;
    nparam->mean_height  += height;
  }
  nparam->mean_width /= nimages; 
  nparam->mean_height /= nimages; 
}

void NormalizeActivationValues(iftMImage **mimg,int nimages, int maxval, NetParameters *nparam)
{
  float *maxactiv = nparam->maxactiv;

    for (int i=0; i < nimages; i++){ /* For each image */
      for (int b=0; b < mimg[i]->m; b++) { /* For each band */
  	maxactiv[b] = 0.0;
  	for (int p=0; p < mimg[i]->n; p++) { /* Find the maximum
  						activation value */
  	  if (mimg[i]->band[b].val[p] > maxactiv[b])
  	    maxactiv[b] = mimg[i]->band[b].val[p];
  	}
      }
    }

    for (int i=0; i < nimages; i++){ /* For each image */
      for (int b=0; b < mimg[i]->m; b++) { /* For each band */
  	if (maxactiv[b] > 0.0){
  	  for (int p=0; p < mimg[i]->n; p++) { /* Normalize activation values */
  	    mimg[i]->band[b].val[p] = maxval * mimg[i]->band[b].val[p]/maxactiv[b];

  	  }
  	}
      }
    }
}

void FindBestKernelWeights(iftMImage **mimg, iftImage **mask, int nimages, NetParameters *nparam)
{
  float *w  = nparam->weight;
  
}

void RegionOfPlates(iftImage **mask, int nimages, NetParameters *nparam)
{
  iftBoundingBox bb;

  bb.begin.x = mask[0]->n;
  bb.begin.y = mask[0]->n;
  bb.end.x   = -1;
  bb.end.y   = -1;
  for (int i=0; i < nimages; i++) {
    for (int p=0; p < mask[i]->n; p++) {
      if (mask[i]->val[p] > 0) {
	iftVoxel u = iftGetVoxelCoord(mask[i],p);
	if (u.x < bb.begin.x) bb.begin.x = u.x;
	if (u.y < bb.begin.y) bb.begin.y = u.y;
	if (u.x > bb.end.x)   bb.end.x   = u.x;
	if (u.y > bb.end.y)   bb.end.y   = u.y;
      }
    }
  }
  nparam->bb.begin.x = bb.begin.x; 
  nparam->bb.end.x   = bb.end.x; 
  nparam->bb.begin.y = bb.begin.y; 
  nparam->bb.end.y   = bb.end.y; 
}


void RemoveActivationsOutOfRegionOfPlates(iftMImage **mimg, int nimages, NetParameters *nparam)
{
  for (int i=0; i < nimages; i++){ /* For each image */
    for (int p=0; p < mimg[i]->n; p++) { /* remove activations out
					      of region of plates */
      iftVoxel u = iftMGetVoxelCoord(mimg[i],p);
      if ((u.x < nparam->bb.begin.x)||(u.y < nparam->bb.begin.y)||
	  (u.x > nparam->bb.end.x)||(u.y > nparam->bb.end.y)) {
	for (int b=0; b < mimg[i]->m; b++){  
	  mimg[i]->band[b].val[p] = 0;
	}
      }
    }
  }  
}

iftMImage **CombineBands(iftMImage **mimg, int nimages, float *weight)
{
  iftMImage **cbands = (iftMImage **) calloc(nimages,sizeof(iftMImage *));  
  iftAdjRel *A[2];

  A[0] = iftRectangular(7,3);
  A[1] = iftRectangular(9,9);
  
  for (int i=0; i < nimages; i++) {
    cbands[i] = iftCreateMImage(mimg[i]->xsize,mimg[i]->ysize,mimg[i]->zsize,1);
    for (int p = 0; p < cbands[i]->n; p++) {
      for (int b=0; b < mimg[0]->m; b++) { 
	cbands[i]->band[0].val[p] += weight[b]*mimg[i]->band[b].val[p];
      }
    }
    iftMImage *aux = MaxPooling(cbands[i], A[0]);
    iftDestroyMImage(&cbands[i]);
    cbands[i] = MinPooling(aux,A[1]);
    iftDestroyMImage(&aux);
  }

  iftDestroyAdjRel(&A[0]);
  iftDestroyAdjRel(&A[1]);

  return(cbands);
}

void FindBestThreshold(iftMImage **cbands, iftImage **mask, int nimages, NetParameters *nparam)
{
  nparam->threshold = 0.0;
}

iftImage **ApplyThreshold(iftMImage **cbands, int nimages, NetParameters *nparam)
{
  iftImage **bin = (iftImage **)calloc(nimages,sizeof(iftImage *));
  
  for (int i=0; i < nimages; i++) {
    bin[i] = iftCreateImage(cbands[i]->xsize,cbands[i]->ysize,cbands[i]->zsize);
    for (int p=0; p < bin[i]->n; p++) {
      if (cbands[i]->band[0].val[p] >= nparam->threshold)
  	bin[i]->val[p]=255;
    }
  }
  
  return(bin);
}

void SelectCompClosestTotheMeanWidthAndHeight(iftImage *label, float mean_width, float mean_height)
{
  int Lmax = iftMaximumValue(label);
  iftBoundingBox bb;
  float best_width=label->n, best_height=label->n;
  int closest = -1;
  for (int i=1; i <= Lmax; i++) {
    iftImage *bin = iftExtractObject(label,i);
    iftVoxel pos;
    bb = iftMinBoundingBox(bin,&pos);
    iftDestroyImage(&bin);
    float width  = (bb.end.x - bb.begin.x);
    float height = (bb.end.y - bb.begin.y);
    if (fabs(width-mean_width)+fabs(height-mean_height) <
	fabs(best_width-mean_width)+fabs(best_height-mean_height)){
      best_width  = width;
      best_height = height;
      closest = i;
    }      
  }
  
  for (int p=0; p < label->n; p++)
    if (label->val[p]!=closest)
      label->val[p]=0;
}


void PostProcess(iftImage **bin, int nimages, NetParameters *nparam)
{
  iftAdjRel *A=iftCircular(sqrtf(2.0));

  for (int i=0; i < nimages; i++) {
    iftImage *aux[2];
    iftSet *S=NULL;
    aux[0] = iftAddFrame(bin[i],15,0);
    aux[1] = iftErodeBin(aux[0],&S,8.0);
    iftDestroyImage(&aux[0]);
    aux[0] = iftDilateBin(aux[1],&S,10.0); 
    iftDestroyImage(&aux[1]);
    aux[1] = iftErodeBin(aux[0],&S,5.0);
    iftDestroyImage(&aux[0]);
    aux[0] = iftRemFrame(aux[1],15);
    iftDestroyImage(&aux[1]);
    iftDestroyImage(&bin[i]);
    iftDestroySet(&S);
    bin[i]    = iftFastLabelComp(aux[0],A);    
    iftDestroyImage(&aux[0]);
    SelectCompClosestTotheMeanWidthAndHeight(bin[i], nparam->mean_width, nparam->mean_height); 
    iftVoxel  pos, u, uo, uf;
    iftBoundingBox  bb = iftMinBoundingBox(bin[i], &pos);

    u.z=uo.z=uf.z=0;

    int xsize   = bb.end.x - bb.begin.x;
    int ysize   = bb.end.y - bb.begin.y;
    int xcenter = bb.begin.x + xsize/2;
    int ycenter = bb.begin.y + ysize/2;
      
    uo.x = iftMax(0,xcenter - nparam->mean_width/2 - 25);
    uo.y = iftMax(0,ycenter - nparam->mean_height/2 - 25);
    uf.x = iftMin(bin[i]->xsize-1,xcenter + nparam->mean_width/2 + 25);
    uf.y = iftMin(bin[i]->ysize-1,ycenter + nparam->mean_height/2 + 25);
      
    for (u.y = uo.y; u.y <= uf.y; u.y++)
      for (u.x = uo.x; u.x <= uf.x; u.x++){
	int p = iftGetVoxelIndex(bin[i],u);
	bin[i]->val[p]=255;
      }
  }
  
  iftDestroyAdjRel(&A);
}

void WriteResults(iftFileSet *fileSet, iftImage **bin)
{
  iftColor RGB, YCbCr;
  iftAdjRel *A = iftCircular(1.0), *B = iftCircular(sqrtf(2.0));
  
  RGB.val[0] = 255;
  RGB.val[1] = 100;
  RGB.val[2] = 0;
    
  YCbCr      = iftRGBtoYCbCr(RGB, 255);

  for (int i=0; i < fileSet->n; i++) {
    iftImage  *img   = iftReadImageByExt(fileSet->files[i]->path); 
    char filename[200];
    iftSList *list = iftSplitString(fileSet->files[i]->path,"_");
    iftSNode *L    = list->tail;
    sprintf(filename,"result_%s",L->elem);
    iftDrawBorders(img,bin[i],A,YCbCr,B);
    iftWriteImageByExt(img,filename);
    iftDestroyImage(&img);
    iftDestroySList(&list);
  }

  iftDestroyAdjRel(&A);
  iftDestroyAdjRel(&B);
}

