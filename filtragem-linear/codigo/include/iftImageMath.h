/**
 * @file iftImageMath.h
 * @brief Image Math operations.
 *
 * @note Programs:
 * * 
 */

#ifndef IFT_IMAGEMATH_H
#define IFT_IMAGEMATH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "iftCommon.h"
#include "iftImage.h"
#include "iftFImage.h"


/** @brief Adds two Images (only the Y channel).*/
iftImage *iftAdd(const iftImage *img1, const iftImage *img2);

/**
 * @brief Adds src and dst image (only the Y channel), saving the values in place in dst.
 * @param src Source Image.
 * @param dst Destination Image
 *
 * @author Samuka
 * @date Jan 16, 2017
 */
void iftAddInPlace(const iftImage *src, const iftImage *dst);


/**
 * @brief Add a scalar into all voxels of an image (only the Y channel).
 * @author Samuka
 * @date Nov 16, 2017
 */
iftImage *iftAddScalar(const iftImage *src, int scalar);


/**
 * @brief Add (in place) a scalar into all voxels of an image (only the Y channel).
 * @author Samuka
 * @date Nov 16, 2017
 */
void iftAddScalarInPlace(iftImage *dst, int scalar);




/** @brief Subtracts two Images (only the Y channel): img1 - img2.*/
iftImage *iftSub(const iftImage *img1, const iftImage *img2);

/**
 * @brief Computes the Intersection between two Label Images
 * (intersection between objects is true if they have the same label).
 * @param  label_img1 Label Image 1.
 * @param  label_img2 label_img 2.
 * @return            Resulting Intersection Image.
 *
 * @author Samuka Martins
 * @date Aug 10, 2017
 */
iftImage *iftIntersec(const iftImage *label_img1, const iftImage *label_img2);

/**
 * @brief Applies a logical AND between two Images (only the Y channel).
 *
 * For each pixel from the images, this functions gets the minimum value (Y channel from YCbCr, regardless
 * if the image is gray or colored) from two images.
 */
iftImage *iftAnd(const iftImage *img1, const iftImage *img2);


/**
 * @brief Applies a logical OR between two Images (only the Y channel).
 *
 * For each pixel from the images, this functions gets the maximum value (Y channel from YCbCr, regardless
 * if the image is gray or colored) from two images.
 */
iftImage *iftOr(const iftImage *img1, const iftImage *img2);


/** @brief Multiplies two Images (only the Y channel).*/
iftImage *iftMult(const iftImage *img1, const iftImage *img2);

/**
 * @brief Multiplies an integer image (only the Y Channel) by a float image. The images must have the same domain.
 * @param  img  Input Integer Image
 * @param  fimg Input Float Image.
 * @return      Resulting multiplied Image.
 *
 * @author Samuka Martins
 * @date Dec 26, 2017
 */
iftFImage *iftMultByFImage(const iftImage *img, const iftFImage *fimg);

/**
 * @brief Multiply an image by a scalar integer value.
 * @author Samuka Martins
 * @date Nov 30, 2017
 */
iftImage *iftMultByScalar(const iftImage *img, float scalar);


/** @brief Gets an Image with the absolute values (only the Y channel) from the Input Image.*/
iftImage *iftAbs(const iftImage *img);



/**
 * @brief Apply a xor on img1 and img2 (if img1->val[p] != img2->val[p], return true)
 * @return      Xor Binary Image.
 * 
 * @author Samuel Martins
 * @date Oct 2, 2017
 */
iftImage *iftXor(const iftImage *img1, const iftImage *img2);


/** @brief Gets the Complement (only the Y channel) from an Image.*/
iftImage *iftComplement(const iftImage *img);


/**
 * @brief Copies an image only where its mask is, i.e., only for the voxels (!= 0) from the mask.
 *
 * @note The Cb and Cr are also copied for Colored Images.
 * 
 * @param  img  Input Image.
 * @param  mask Mask that determines which voxels are copied.
 * @return      Copied Image.
 */
iftImage *iftMask(const iftImage *img, const iftImage *mask);


/**
 * @brief Copies an image only where its object mask is, i.e., only for the voxels == label.
 *
 * @note The Cb and Cr are also copied for Colored Images.
 * 
 * @param  img   Input Image.
 * @param  mask  Mask with multiple objects, where the voxels with label <label> will be copied.
 * @param  label Label from the object to be masked.
 * @return       Copied Image.
 */
iftImage *iftObjectMask(const iftImage *img, const iftImage *mask, int label);


/**
 * @brief Binarize a labeled image.
 * @author Samuka Martins
 * @date Aug 10, 2017
 */
iftImage *iftBinarize(const iftImage *label_img);


/** @brief Adds a value for all voxel only in the Y channel */
iftImage *iftAddValue(const iftImage *img, int val);


/** Transforms a Binary Image <b>bin</b> into a Binary Image where each label value has value 255 */
iftImage *iftBinaryFrom1To255(const iftImage *bin);


/** Transforms a Binary Image <b>bin</b> with object values != 1 into a Binary Image with values 1 */
iftImage *iftBinaryFrom255To1(const iftImage *bin);


/** @brief Adds a Value in an Input Image only in the voxels of a Mask (!= 0) */
iftImage *iftAddValueInRegion(const iftImage *img, int value, const iftImage *mask);


/** @brief Applies a square root into an Image */
iftFImage *iftSQRT(iftImage const *img1);


/** @brief Applies a square root into a Float Image */
iftFImage *iftFSQRT(const iftFImage *img);


/**
 * @brief Linearly combines two images given img0[x]*alpha+(1-alpha)*img1[x]. The images must have the same
 * size and must be normalized between the same minimum and maximum values.
 *
 * @author Thiago Vallin Spina
 * @date Jan 23, 2016
 *
 * @param img0 First input image.
 * @param img1 Second input image.
 *
 * @return Linearly combined image.
 */
iftImage *iftImageLinearCombination(const iftImage *img0, const iftImage *img1, double alpha);


#ifdef __cplusplus
}
#endif

#endif
