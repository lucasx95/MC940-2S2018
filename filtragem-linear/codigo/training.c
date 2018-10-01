#include "ift.h"
#include "neural_net.c"
#include "limits.h"
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

int main(int argc, char *argv[]) {
    iftImage **mask;
    iftMImage **mimg, **cbands;

    if (argc != 4)
        iftError("training <trainX.txt (X=1,2,3,4,5)> <kernel-bank.txt> <output-parameters.txt>", "main");

    /* Read input images and kernel bank */

    iftFileSet *trainSet = iftLoadFileSetFromCSV(argv[1], false);
    mask = (iftImage **) calloc(trainSet->n, sizeof(iftImage *));
    mimg = (iftMImage **) calloc(trainSet->n, sizeof(iftMImage *));
    MKernelBank *Kbank = ReadMKernelBank(argv[2]);

    /* Apply the single-layer NN in all training images */

    for (int i = 0; i < trainSet->n; i++) {
        printf("Processing file %s\n", trainSet->files[i]->path);
        iftImage *img = iftReadImageByExt(trainSet->files[i]->path);
        mask[i] = ReadMaskImage(trainSet->files[i]->path);
        if ((img->xsize != 352) || (img->ysize != 240))
            printf("imagem %s ", trainSet->files[i]->path);
        mimg[i] = SingleLayer(img, Kbank);
        iftDestroyImage(&img);
    }

    /* Compute plate parameters and normalize activation values within
       [0,255] */

    NetParameters *nparam = CreateNetParameters(mimg[0]->m);
    ComputeAspectRatioParameters(mask, trainSet->n, nparam);
    RegionOfPlates(mask, trainSet->n, nparam);
    RemoveActivationsOutOfRegionOfPlates(mimg, trainSet->n, nparam);
    NormalizeActivationValues(mimg, trainSet->n, 255, nparam);

    /* Find the best kernel weights */

    FindBestKernelWeights(mimg, mask, trainSet->n, nparam);

    /* Combine bands, find optimum threshold, and apply it */

    cbands = CombineBands(mimg, trainSet->n, nparam->weight);
    RemoveActivationsOutOfRegionOfPlates(cbands, trainSet->n, nparam);
    FindBestThreshold(cbands, mask, trainSet->n, nparam);

    WriteNetParameters(nparam, argv[3]);
    iftImage **bin = ApplyThreshold(cbands, trainSet->n, nparam);

    /* Post-process binary images and write results on training set */

    PostProcess(bin, trainSet->n, nparam);
    WriteResults(trainSet, bin);

    /* Free memory */

    for (int i = 0; i < trainSet->n; i++) {
        iftDestroyImage(&mask[i]);
        iftDestroyImage(&bin[i]);
        iftDestroyMImage(&cbands[i]);
        iftDestroyMImage(&mimg[i]);
    }
    iftFree(mask);
    iftFree(mimg);
    iftFree(bin);
    iftFree(cbands);
    iftDestroyFileSet(&trainSet);
    DestroyMKernelBank(&Kbank);
    DestroyNetParameters(&nparam);

    return (0);
}