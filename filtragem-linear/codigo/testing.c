#include "ift.h"
#include "neural_net.c"
#include <stdbool.h>
#include <float.h>

iftImage *ReadMaskImage(char *pathname) {
    iftImage *mask = NULL;
    iftSList *list = iftSplitString(pathname, "_");
    iftSNode *L = list->tail;
    iftSList *other_list = iftSplitString(L->elem,".");
    L = other_list->head;
    char filename[200];
    sprintf(filename, "./imagens/placas-mask/%s.pgm", L->elem);
    mask = iftReadImageByExt(filename);
    iftDestroySList(&list);
    iftDestroySList(&other_list);
    return (mask);
}

int main(int argc, char *argv[])
{
    iftImage  **mask;
    iftMImage **mimg, **cbands;
    NetParameters *nparam;

    if (argc!=4)
        iftError("testing <testX.txt (X=1,2,3,4,5)> <kernel-bank.txt> <input-parameters.txt>","main");

    /* Read input images and kernel bank */

    iftFileSet  *testSet = iftLoadFileSetFromCSV(argv[1], false);
    mask = (iftImage **)  calloc(testSet->n,sizeof(iftImage *));
    mimg = (iftMImage **) calloc(testSet->n,sizeof(iftMImage *));
    int *kernelx = calloc(1, sizeof(int)), *kernely = calloc(1,
                                                             sizeof(int));
    iftMatrix *Kbank    = ReadMKernelBankAsMatrix(argv[2],kernelx, kernely);
    nparam                = ReadNetParameters(argv[3]);

    /* Apply NN in all test images */

    for (int i=0; i < testSet->n; i++) {
        printf("Processing file %s\n",testSet->files[i]->path);
        iftImage  *img   = iftReadImageByExt(testSet->files[i]->path);
        mask[i]          = ReadMaskImage(testSet->files[i]->path);
        mimg[i]          = SingleLayer(img,Kbank,*kernelx,*kernely);
        iftDestroyImage(&img);
    }

    /* Normalize activation values within [0,255] */

    //RemoveActivationsOutOfRegionOfPlates(mimg, testSet->n, nparam);
    NormalizeActivationValues(mimg,testSet->n,255,nparam);

    /* Combine bands */

    cbands = CombineBands(mimg, testSet->n, nparam->weight);
    RemoveActivationsOutOfRegionOfPlates(cbands, testSet->n, nparam);

    iftImage **bin = ApplyThreshold(cbands, testSet->n, nparam);

    /* Post-process binary images and write results on training set */

    PostProcess(bin,testSet->n, nparam);
    WriteResults(testSet,bin, false);

    /* Free memory */

    for (int i=0; i < testSet->n; i++) {
        iftDestroyImage(&mask[i]);
        iftDestroyImage(&bin[i]);
        iftDestroyMImage(&cbands[i]);
        iftDestroyMImage(&mimg[i]);
    }
    iftFree(mask);
    iftFree(mimg);
    iftFree(bin);
    iftFree(cbands);
    iftDestroyFileSet(&testSet);
    iftDestroyMatrix(&Kbank);
    DestroyNetParameters(&nparam);


    return(0);
}