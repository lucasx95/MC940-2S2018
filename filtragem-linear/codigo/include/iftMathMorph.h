#ifndef IFT_MATHMORPH_H_
#define IFT_MATHMORPH_H_

#ifdef __cplusplus
extern "C" {
#endif


#include "iftCommon.h"
#include "iftImage.h"
#include "iftImageMath.h"
#include "iftSeeds.h"
#include "iftGQueue.h"
#include "iftAdjacency.h"
#include "iftFiltering.h"
#include "iftCompTree.h"

/**
 * @brief Morphological dilation by using a kernel as a non-planar
 * structuring element, with the option to constrain the operation
 * within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  K     Input kernel
 * @param  mask  Input region (otherwise, NULL)
 * @return image after dilation
 */
  
  iftImage *iftDilateWithKernel(iftImage *img, iftKernel *K, iftImage *mask);

/**
 * @brief Morphological erosion by using a kernel as a non-planar
 * structuring element, with the option to constrain the operation
 * within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image.
 * @param  K     Input kernel
 * @param  mask  Input region (otherwise, NULL)
 * @return image after erosion
 */

  iftImage *iftErodeWithKernel(iftImage *img, iftKernel *K, iftImage *mask);

/**
 * @brief Morphological closing by using a kernel as a non-planar
 * structuring element, with the option to constrain the operation
 * within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image.
 * @param  K     Input kernel
 * @param  mask  Input region (otherwise, NULL)
 * @return image after closing
 */

  iftImage *iftCloseWithKernel(iftImage *img, iftKernel *K, iftImage *mask);

/**
 * @brief Morphological opening by using a kernel as a non-planar
 * structuring element, with the option to constrain the operation
 * within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image.
 * @param  K     Input kernel
 * @param  mask  Input region (otherwise, NULL)
 * @return image after opening
 */

  iftImage *iftOpenWithKernel(iftImage *img, iftKernel *K, iftImage *mask);

/**
 * @brief Morphological dilation by using adjacency relation as a
 * planar structuring element, with the option to constrain the
 * operation within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @param  mask  Input region (otherwise, NULL)
 * @return image after dilation
 */

//! swig(newobject)
  iftImage *iftDilate(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Morphological erosion by using adjacency relation as a
 * planar structuring element, with the option to constrain the
 * operation within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @param  mask  Input region (otherwise, NULL)
 * @return image after erosion
 */

//! swig(newobject)
  iftImage *iftErode(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Morphological closing by using an adjacency relation as a
 * planar structuring element, with the option to constrain the
 * operation within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @param  mask  Input region (otherwise, NULL)
 * @return image after closing
 */

//! swig(newobject)
  iftImage *iftClose(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Morphological opening by using an adjacency relation as a
 * planar structuring element, with the option to constrain the
 * operation within a binary mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @param  mask  Input region (otherwise, NULL)
 * @return image after opening
 */

//! swig(newobject)
  iftImage *iftOpen(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Alternate sequential filter by using an adjacency relation
 * as a planar structuring element, with the option to constrain the
 * operation within a binary mask. This filter applies a closing
 * operation followed by an opening operation.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @param  mask  Input region (otherwise, NULL)
 * @return filtered image
 */

  iftImage *iftAsfCO(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Alternate sequential filter by using an adjacency relation
 * as a planar structuring element, with the option to constrain the
 * operation within a binary mask. This filter applies an opening
 * operation followed by a closing operation.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @param  mask  Input region (otherwise, NULL)
 * @return filtered image
 */

  iftImage *iftAsfOC(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Exact morphological dilation for a binary image using a
 * dilation radius. The method uses the IFT-based Euclidean distance
 * transform such that the seed set is the object border. This border
 * set can be computed if the seed set is empty (NULL) or it is the
 * result of a previous erosion. You must use iftAddFrame before
 * the operation and iftRemFrame after it, using a frame size equal to
 * the radius, whenever the distance between object and image border
 * is less than the radius.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin    Input binary image
 * @param  seed   Input and output object border set (Otherwise, NULL)
 * @param  radius Input dilation radius
 * @return image after dilation
 * @note You must use iftAddFrame to extend the image domain before
 * dilation, whenever the distance between object and border of the
 * image is less than the radius. Note that, by using iftRemFrame, the
 * image returns to the original domain, but parts of the dilated
 * object might be clipped.
 */

  iftImage *iftDilateBin(iftImage *bin, iftSet **seed, float radius);

 /**
 * @brief Exact morphological erosion for a binary image using a
 * erosion radius. The method uses the IFT-based Euclidean distance
 * transform such that the seed set is the background border. This
 * border set can be computed if the seed set is empty (NULL) or it is
 * the result of a previous dilation. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin    Input binary image
 * @param  seed   Input and output background border set (Otherwise, NULL)
 * @param  radius Input erosion radius
 * @return image after erosion
 * @note You must use iftAddFrame to extend the image domain before
 * erosion, whenever the object touches the border of the image to
 * guarantee seeds in the entire background around the object.
 */

  iftImage *iftErodeBin(iftImage *bin, iftSet **seed, float radius);

/**
 * @brief Exact morphological erosion for a label image using a closing radius.
 * 
 * For each object, the functions applies the erosion by using the function iftErodeBin.
 * Since there will not be intersection between the objects after erosion, the resulting image
 * is the concatenation of all eroded objects.
 * 
 * @note See the program: ift/demo/MathMorphology/iftErodeLabelImage.c
 * 
 * @param  label_image Input Label Image with multiple objects.
 * @param  radius      Input closing radius.
 * @return             Filtered image. 
 * 
 * @author Samuel Martins
 * @date Apr 24, 2018
 */
iftImage *iftErodeLabelImage(const iftImage *label_img, float radius);

/**
 * @brief Exact morphological closing for a binary image using a
 * closing radius. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin    Input binary image
 * @param  radius Input closing radius
 * @return filtered image 
 */
  
  iftImage *iftCloseBin(iftImage *bin, float radius);


/**
 * @brief Exact morphological opening for a binary image using an
 * opening radius. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin    Input binary image
 * @param  radius Input opening radius
 * @return filtered image 
 */

  iftImage *iftOpenBin(iftImage *bin, float radius);

/**
 * @brief Alternate sequential filter by reconstruction of a binary
 * image using the decomposition of closing operation followed by
 * opening operation into binary dilations and binary erosions. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin      Input image
 * @param  radius   Input radius 
 * @return filtered image  
 */
  
  iftImage *iftAsfCOBin(iftImage *bin, float radius);

  /**
 * @brief Alternate sequential filter by reconstruction of a binary
 * image using the decomposition of opening operation followed by
 * closing operation into binary dilations and binary erosions. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin      Input image
 * @param  radius   Input radius 
 * @return filtered image  
 */

  iftImage *iftAsfOCBin(iftImage *bin, float radius);
  
/**
 * @brief Exact morphological closing by reconstruction for a binary
 * image using a closing radius. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin    Input binary image
 * @param  radius Input closing radius
 * @return filtered image 
 */

  iftImage *iftCloseRecBin(iftImage *bin, float radius);

/**
 * @brief Exact morphological opening by reconstruction for a binary
 * image using an opening radius. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  bin    Input binary image
 * @param  radius Input opening radius
 * @return filtered image 
 */

  iftImage *iftOpenRecBin(iftImage *bin, float radius);  
  
/**
 * @brief Morphological gradient by using an adjacency relation as a
 * planar structuring element. The gradient is computed by subtracting
 * the erosion of the image from the dilation of the image. In the
 * case of color images, the maximum among the gradients of the
 * components is taken. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img   Input image
 * @param  A     Input adjacency relation
 * @return (grayscale) gradient image  
 */
  
  iftImage *iftMorphGrad(iftImage *img, iftAdjRel *A);

/**
 * @brief Superior reconstruction of an image from a marker. It allows
 * to constrain the operation inside a mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img      Input image
 * @param  marker   Input marker
 * @param  mask     Input region (otherwise, NULL)
 * @return (grayscale) filtered image  
 * @note The marker image M must contain the image I. That is, M(p) >=
 * I(p) for all voxels p. In the case of color images, only the
 * luminance values are filtered.
 */
  
  iftImage *iftSuperiorRec(iftImage *img, iftImage *marker, iftImage *mask);

/**
 * @brief Inferior reconstruction of an image from a marker. It allows
 * to constrain the operation inside a mask.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img      Input image
 * @param  marker   Input marker
 * @param  mask     Input region (otherwise, NULL)
 * @return (grayscale) filtered image  
 * @note The marker image M must be contained in the image I. That is,
 * M(p) <= I(p) for all voxels p. In the case of color images, only
 * the luminance values are filtered.
 */
  
  iftImage *iftInferiorRec(iftImage *img, iftImage *marker, iftImage *mask);
  
/**
 * @brief Opening by reconstruction of an image using an adjacency
 * relation as a planar structuring element. In the case of color
 * images, only the luminance values are filtered. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img      Input image
 * @param  A        Input adjacency relation
 * @param  mask     Input region (otherwise, NULL)
 * @return (grayscale) filtered image  
 */

  iftImage *iftOpenRec(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Closing by reconstruction of an image using an adjacency
 * relation as a planar structuring element. In the case of color
 * images, only the luminance values are filtered. 
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img      Input image
 * @param  A        Input adjacency relation
 * @param  mask     Input region (otherwise, NULL)
 * @return (grayscale) filtered image  
 */

  iftImage *iftCloseRec(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Alternate sequential filter by reconstruction of an image
 * using an adjacency relation as a planar structuring element. The
 * filter applies a closing operation followed by an opening
 * operation. In the case of color images, only the luminance values
 * are filtered.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img      Input image
 * @param  A        Input adjacency relation
 * @param  mask     Input region (otherwise, NULL)
 * @return (grayscale) filtered image  
 */

  iftImage *iftAsfCORec(iftImage *img, iftAdjRel *A, iftImage *mask);

/**
 * @brief Alternate sequential filter by reconstruction of an image
 * using an adjacency relation as a planar structuring element. The
 * filter applies an opening operation followed by a closing
 * operation. In the case of color images, only the luminance values
 * are filtered.
 * @author Alexandre Falcao and Jordao Bragantini 
 * @date Feb, 2018
 * @param  img      Input image
 * @param  A        Input adjacency relation
 * @param  mask     Input region (otherwise, NULL)
 * @return (grayscale) filtered image  
 */

  iftImage *iftAsfOCRec(iftImage *img, iftAdjRel *A, iftImage *mask);


/**
 * @brief Closes basins (holes) in the input image from a given seed
 * set and inside a given mask. The default seed set (when seed ==
 * NULL) is the border of the image, when the mask is not provided
 * (mask == NULL), otherwise it is the border of the mask. Seeds must
 * be included in the mask.  
 * @author Alexandre Falcao 
 * @date Nov 10, 2016
 * @param img   Input image
 * @param seed  Input seed set (or NULL)
 * @param mask  Input region (or NULL)
 * @return image with basins (holes) closed
 */

  iftImage *iftCloseBasins(iftImage *img, iftSet *seed, iftImage *mask);

/**
 * @brief Opens domes in the input image from a given seed set and
 * inside a given mask. The default seed set (when seed is NULL) is
 * the border of the image, when the mask is not provided (when mask
 * is NULL), otherwise it is the border of the mask. Seeds must be
 * included in the mask.  
 * @author Alexandre Falcao 
 * @date Nov 10, 2016
 * @param img   Input image
 * @param seed  Input seed set (or NULL)
 * @param mask  Input region (or NULL)
 * @return image with domes opened
 */

  iftImage *iftOpenDomes(iftImage *img, iftSet *seed, iftImage *mask);

/**
 * @brief Closes basins in the input image whose depth is less than H and
 * inside a given mask.   
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param img   Input image
 * @param H     Input minimum desired depth
 * @param mask  Input region (or NULL)
 * @return filtered image
 */

  iftImage *iftHClose(iftImage *img, int H, iftImage *mask);

  /**
 * @brief Opens domes in the input image whose height is less than H and
 * inside a given mask.   
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param img   Input image
 * @param H     Input minimum desired height
 * @param mask  Input region (or NULL)
 * @return filtered image
 * @note It is implemented based on iftHClose to avoid dealing with
 * negative values in the marker image for inferior reconstruction.
 */

  iftImage *iftHOpen(iftImage *img, int H, iftImage *mask);

  /**
 * @brief Closes basins in the input image, whose area is less than a
 * threshold, by using the union-find algorithm.
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param  img   Input image
 * @param  thres Input area threshold
 * @return filtered image
 */

  iftImage *iftFastAreaClose(iftImage *img, int thres);

  /**
 * @brief Opens domes in the input image, whose area is less than a
 * threshold, by using the union-find algorithm.
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param  img   Input image
 * @param  thres Input area threshold
 * @return filtered image
 */

  iftImage *iftFastAreaOpen(iftImage *img, int thres);

  /**
 * @brief Close basins in the input image, whose area is less than a
 * threshold, by using a component tree. 
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param  img   Input image
 * @param  thres Input area threshold
 * @param  ctree Input component tree from the minima (or NULL) 
 * @return filtered image
 * @note It computes the component tree when it is not given.
 */
     
  iftImage *iftAreaClose(iftImage *img, int area_thres, iftCompTree *ctree);

    /**
 * @brief Close basins in the input image, whose volume is less than a
 * threshold, by using a component tree. 
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param  img   Input image
 * @param  thres Input volume threshold
 * @param  ctree Input component tree from the minima (or NULL) 
 * @return filtered image
 * @note It computes the component tree when it is not given.
 */

  iftImage *iftVolumeClose(iftImage *img, int volume_thres, iftCompTree *ctree);
 
  /**
 * @brief Open domes in the input image, whose area is less than a
 * threshold, by using a component tree. 
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param  img   Input image
 * @param  thres Input area threshold
 * @param  ctree Input component tree from the minima (or NULL) 
 * @return filtered image
 * @note It computes the component tree when it is not given.
 */

  iftImage *iftAreaOpen(iftImage *img, int area_thres, iftCompTree *ctree);

  /**
 * @brief Open domes in the input image, whose volume is less than a
 * threshold, by using a component tree. 
 * @author Alexandre Falcao 
 * @date Feb 2018
 * @param  img   Input image
 * @param  thres Input volume threshold
 * @param  ctree Input component tree from the minima (or NULL) 
 * @return filtered image
 * @note It computes the component tree when it is not given.
 */

  iftImage *iftVolumeOpen(iftImage *img, int volume_thres, iftCompTree *ctree);
  

#ifdef __cplusplus
}
#endif

#endif
