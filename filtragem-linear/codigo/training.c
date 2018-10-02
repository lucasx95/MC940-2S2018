#include "ift.h"
#include "neural_net.c"
#include "limits.h"
#include <stdbool.h>

iftImage *ReadMaskImage(char *pathname) {
    iftImage *mask = NULL;
    iftSList *list = iftSplitString(pathname, "_");
    iftSNode *L = list->tail;
    iftSList *other_list = iftSplitString(L->elem, ".");
    L = other_list->head;
    char filename[200];
    sprintf(filename, "./imagens/placas-mask/%s.pgm", L->elem);
    mask = iftReadImageByExt(filename);
    iftDestroySList(&list);
    iftDestroySList(&other_list);
    return (mask);
}

int main(int argc, char *argv[]) {
    iftImage **mask;
    iftMImage **mimg, **regionMimg, **cbands;

    if (argc != 4)
        iftError("training <trainX.txt (X=1,2,3,4,5)> <kernel-bank.txt> <output-parameters.txt>", "main");

    /* Read input images and kernel bank */

    iftFileSet *trainSet = iftLoadFileSetFromCSV(argv[1], false);
    mask = (iftImage **) calloc(trainSet->n, sizeof(iftImage *));
    mimg = (iftMImage **) calloc(trainSet->n, sizeof(iftMImage *));
    regionMimg = (iftMImage **) calloc(trainSet->n, sizeof(iftMImage *));
    int *kernelx = calloc(1, sizeof(int)), *kernely = calloc(1,
                                                             sizeof(int));
    iftMatrix *Kbank = ReadMKernelBankAsMatrix(argv[2], kernelx, kernely);

    /* Apply the single-layer NN in all training images */

    for (int i = 0; i < trainSet->n; i++) {
        printf("Processing file %s\n", trainSet->files[i]->path);
        iftImage *img = iftReadImageByExt(trainSet->files[i]->path);
        mask[i] = ReadMaskImage(trainSet->files[i]->path);
        if ((img->xsize != 352) || (img->ysize != 240))
            printf("imagem %s ", trainSet->files[i]->path);
        mimg[i] = SingleLayer(img, Kbank, *kernelx, *kernely);
        iftDestroyImage(&img);
    }

    /* Compute plate parameters and normalize activation values within
       [0,255] */

    NetParameters *nparam = CreateNetParameters(mimg[0]->m);
    ComputeAspectRatioParameters(mask, trainSet->n, nparam);
    RegionOfPlates(mask, trainSet->n, nparam);
    for (int i = 0; i < trainSet->n; i++) {
        regionMimg[i] = iftCopyMImage(mimg[i]);
    }
    RemoveActivationsOutOfRegionOfPlates(regionMimg, trainSet->n, nparam);
    NormalizeActivationValues(regionMimg, trainSet->n, 255, nparam);

    /* Find the best kernel weights using the region cutted image */

    FindBestKernelWeights(regionMimg, mask, trainSet->n, nparam);

    /* Combine bands, find optimum threshold, and apply it */

    NormalizeActivationValues(mimg, trainSet->n, 255, nparam);
    cbands = CombineBands(mimg, trainSet->n, nparam->weight);
    RemoveActivationsOutOfRegionOfPlates(cbands, trainSet->n, nparam);
    FindBestThreshold(cbands, mask, trainSet->n, nparam);

    WriteNetParameters(nparam, argv[3]);
    iftImage **bin = ApplyThreshold(cbands, trainSet->n, nparam);

    /* Post-process binary images and write results on training set */

    PostProcess(bin, trainSet->n, nparam);
    WriteResults(trainSet, bin, true);

    /* Free memory */

    for (int i = 0; i < trainSet->n; i++) {
        iftDestroyImage(&mask[i]);
        iftDestroyImage(&bin[i]);
        iftDestroyMImage(&cbands[i]);
        iftDestroyMImage(&mimg[i]);
        iftDestroyMImage(&regionMimg[i]);
    }
    iftFree(mask);
    iftFree(mimg);
    iftFree(bin);
    iftFree(cbands);
    iftDestroyFileSet(&trainSet);
    iftDestroyMatrix(&Kbank);
    DestroyNetParameters(&nparam);

    return (0);
}