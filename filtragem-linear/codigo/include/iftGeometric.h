#ifndef IFT_GEOMETRIC_H_
#define IFT_GEOMETRIC_H_

#ifdef __cplusplus
extern "C" {
#endif

#include "iftCommon.h"
#include "iftImage.h"
#include "iftMatrix.h"
#include "iftInterpolation.h"
#include "iftDataSet.h"

iftImage  *iftRotateImage2D(iftImage *img, float theta);


/**
 * @brief Scales a 2D image by the factors sx and sy for the x-axis and y-axis, respectively.
 * @author Samuka Martins
 * @date Aug 17, 2017
 */
iftImage *iftScaleImage2D(const iftImage *img, float sx, float sy);


iftImage  *iftRotateImage(iftImage *img, float theta_x, float theta_y);


/**
 * @brief Scale an image by the factors sx, sy, and sz for the x-axis, y-axis, and z-axis, respectively.
 * @author Samuka Martins
 * @date Aug 17, 2017
 */
iftImage *iftScaleImage(const iftImage *img, float sx, float sy, float sz);



/**
 * @brief Flip an Image on a given axis (IFT_AXIS_X, IFT_AXIS_Y, or AXIS_Z)
 * @author Samuel Martins
 * @date Mar 12, 2018
 */
iftImage *iftFlipImage(const iftImage *img, char axis);


iftImage  *iftTransformImageClipping(iftImage *img, iftMatrix *InvE, int xsize, int ysize, int zsize);
iftFImage *iftTransformFImageClipping(iftFImage *img, iftMatrix *InvE, int xsize, int ysize, int zsize);
iftImage  *iftTransformImageClip(iftImage *img, iftMatrix *M);

#ifdef __cplusplus
}
#endif

#endif
