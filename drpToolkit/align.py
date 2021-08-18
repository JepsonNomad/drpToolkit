''' 
drpToolkit align module

    align: This module can be used to align a folder of time-lapse images
    to a keyframe using affine or homography tranformations.


Recommended Python 3.7.

Author: Christian John
August 2021
GNU General Public License
'''

import argparse
from datetime import datetime
import os
import glob
import numpy as np
import cv2 as cv
import pandas as pd


def estimateTransform(newImage, refImage, refMask, transModel, rRT, lRT, summarizeTransformError):
    '''
    Estimate the transformation matrix between a novel image (newImage) 
    and reference image (refImage). If desired, a mask (refImMask) can 
    be provided for the reference image. The mask should be a numpy array with 
    the same height and width as the reference image, and values should be 
    either 0 (keypoints excluded) or 255 (keypoints included). Two 
    transformation models (transModel) are available, affine and homography. 
    Homography is a more flexible transformation but in some ecology time-lapse 
    applications may be sensitive to scene variation and yield unrealistic results.
    '''
    # SIFT keypoints
    detector = cv.SIFT_create()
    keypoints1, descriptors1 = detector.detectAndCompute(newImage, None)
    keypoints2, descriptors2 = detector.detectAndCompute(refImage, refMask)
    # FLANN descriptor matcher
    matcher = cv.DescriptorMatcher_create(cv.DescriptorMatcher_FLANNBASED)
    knn_matches = matcher.knnMatch(descriptors1, descriptors2, 2)
    # Apply Lowe's ratio
    # See https://stackoverflow.com/q/56986350/5090454
    good_matches = []
    for m,n in knn_matches:
            if m.distance < lRT * n.distance:
                    good_matches.append(m)
    # Count the good matches
    nMatches =    len(good_matches)
    # print("Number of matches: ")
    # print(nMatches)
    # Only apply the warp to imagery with a decent sample of good matches
    if nMatches > 100:
        # Extract location of good matches
        points1 = np.zeros((len(good_matches), 2), dtype=np.float32)
        points2 = np.zeros((len(good_matches), 2), dtype=np.float32)
        for i, m in enumerate(good_matches):
            points1[i, :] = keypoints1[m.queryIdx].pt
            points2[i, :] = keypoints2[m.trainIdx].pt
        # Find transformation
        if transModel == "Affine":
                h, mask = cv.estimateAffinePartial2D(points1, points2, method = cv.RANSAC, ransacReprojThreshold = rRT)
        elif transModel == "Homography":
                h, mask = cv.findHomography(points1, points2, method = cv.RANSAC, ransacReprojThreshold = rRT)
        else:
                sys.exit("Invalid transModel selected. Check parameter definitions.")
        
        ## Interpret tranformation error by back-transforming points with homography matrix
        if summarizeTransformError:
            # Note discussion here: https://stackoverflow.com/questions/8600874/opencv-perspectivetransform-function-exception
            y = cv.perspectiveTransform(points1[np.newaxis], h)[0]
            # Create a vector and calculate
            dst = np.zeros(len(y))
            for i in range(0,len(y)):
                dst[i] = np.sqrt((y[i,0] - points2[i,0])**2 + (y[i,1] - points2[i,1])**2)
            projErrMean = np.mean(dst)
            projErrMedian = np.median(dst)
            # Check the distribution of reprojection squared error
            # plt.hist(dst,200)
            # plt.title("Distance between source points\nand reprojected new points")
        else:
            projErrMean = None
            projErrMedian = None
    else:
        h = None
        projErrMean = None
        projErrMedian = None      
    return h, projErrMean, projErrMedian


def applyTransform(img, h, transModel):
    '''
    Use a transformation matrix to apply a transformation to an image, using
    an affine or homography transformation model.
    '''
    height, width, channels = img.shape
    if transModel == "Affine":
        imgReg = cv.warpAffine(img, h, (width, height))
    elif transModel == "Homography":
        imgReg = cv.warpPerspective(img, h, (width, height))
    else:
        sys.exit("Invalid transModel selected. Check parameter definitions.")
    return imgReg


def generateTransformTable(keyframeFP, imageFPs, refMaskFP, transModel, rRT, lRT, summarizeTransformError, outdir):
    '''
    Generate a table with proposed transformation matrices for a set of imagery.
    Requires the filepath of a keyframe (keyframe) and a list of filepaths for
    imagery to be aligned (imageFPs). Also requires transformation estimation
    parameters passed directly to estimateTransform(). An output folder (outPath) 
    should point to a place to save aligned images, the keyframe mask, and 
    transformation table. If summarizeTransformError is set to True (the default,
    but slower), the transformation table will include mean and median distance 
    between transformed keypoint coordinates and source keypoint coordinates.
    '''
    print("Generating transformation table...")
    ## Set up transform table and initialize transformation matrix
    if transModel == "Affine":
        hCols = ["h"+str(i) for i in range(1,7)]
        hLast = np.array([[1.,0.,0.],[0.,1.,0.]])
    elif transModel == "Homography":
        hCols = ["h"+str(i) for i in range(1,10)]
        hLast = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    # Create column names
    ttColsBase = ["fileName", "transModel"]
    ttColNames = ttColsBase + hCols + ["projErrMean", "projErrMedian"]
    #print(ttColNames)
    transformTable = pd.DataFrame(columns = ttColNames)
    #print(transformTable)
    ## Import keyframe
    refImage = cv.imread(keyframeFP)
    if refMaskFP is not None:
        refMask = cv.imread(refMaskFP)[:,:,0]
    else:
        refMask = None
    imCounter = 0
    numImgs = len(imageFPs)
    for q in imageFPs:
        imCounter = imCounter + 1
        if imCounter % 50 == 0:
            print("Aligning image " + str(imCounter) + " of " + str(numImgs))
        imgBN = os.path.basename(q)
        # print("Input image: " + imgBN)
        newImage = cv.imread(q)
        h, projErrMean, projErrMedian = estimateTransform(newImage = newImage, refImage = refImage, refMask = refMask, transModel = transModel, rRT = rRT, lRT = lRT, summarizeTransformError = summarizeTransformError)
        #print("Transformation info")
        #print((h, projErrMean, projErrMedian))
        if h is not None:
            if abs(np.linalg.det(h) - 1) < 0.1:
                imgReg = applyTransform(img = newImage, h = h, transModel = transModel)
                hLast = h
                # print("Proposed transformation matrix:")
            else:
                imgReg = applyTransform(img = newImage, h = hLast, transModel = transModel)
                # print("Wonky transformation detected. Defaulting to old matrix.")
        else:
            imgReg = applyTransform(img = newImage, h = hLast, transModel = transModel)
            # print("Insufficient keypoint pairs detected. Defaulting to old transformation matrix.")

        # print(np.round(hLast,2))
        # Prep transformation matrix for export
        hFlat = hLast.flatten().tolist()
        dfRow = [imgBN, transModel] + hFlat + [projErrMean, projErrMedian]
        #print("dfRow")
        #print(dfRow)
        dfSeries = pd.Series(dfRow, index = ttColNames)
        # Append image name, transformation model, and transformation matrix to table
        transformTable = transformTable.append(dfSeries, ignore_index=True)
        ## Save transformed image
        outpath = os.path.join(outdir, imgBN)
        # print("Output image path: " + outpath)
        cv.imwrite(outpath, imgReg)
    
    ## Return the transformation table
    return transformTable    


def transformFromTable(imageFPs, transTableFP, transModel, outdir):
    '''
    Use a transformation table (*.csv) to apply a transformation to a series of 
    images, using an affine or homography transformation model. The transformation
    table should include columns titled 'h1' through 'h6' if using Affine, or 'h1'
    through 'h9' if using Homography transformations.
    '''
    tt = pd.read_csv(transTableFP)
    imCounter = 0
    numImgs = len(imageFPs)
    for q in imageFPs:
        imCounter = imCounter + 1
        if imCounter % 50 == 0:
            print("Transforming image " + str(imCounter) + " of " + str(numImgs))
        imgBN = os.path.basename(q)
        # print("Input image: " + imgBN)
        newImage = cv.imread(q)
        ttRow = tt[tt['fileName'] == imgBN]
        # Convert values from the transformation table row to an homography matrix
        if transModel == "Affine":
            hFlat = ttRow.loc[:,'h1':'h6'].to_numpy()
        elif transModel == "Homography":
            hFlat = ttRow.loc[:,'h1':'h9'].to_numpy()
    	# Reshape numpy array to useful matrix shape
        h = np.reshape(hFlat, (-1,3))
        # Transform the image
        imgReg = applyTransform(img = newImage, h = h, transModel = transModel)
        ## Save transformed image
        outpath = os.path.join(outdir, imgBN)
        # print("Output image path: " + outpath)
        cv.imwrite(outpath, imgReg)



# For command line scripting:
def getArgs():
    """
    Get args from cmd line
    """
    parser = argparse.ArgumentParser(
                                description="""align: This module can be used to align
    a folder of time-lapse images to a keyframe using affine or homography tranformations.
    DEPENDENCIES: Python: numpy, opencv, pandas.""")
	
    parser.add_argument("-i", "--imgDir",
                        required=True,
                        help="REQUIRED: The full path to a directory of images "
                        "to be aligned.")
                        
    parser.add_argument("-k", "--keyframeFP",
                        required=True,
                        help="REQUIRED: The full path to a reference image. Should "
                        "be the same dimensions as images to be aligned")
                        
    parser.add_argument("-g", "--globString",
                        required=False,
                        default="*.JPG",
                        help="OPTIONAL: A glob-friendly filename search string "
                        "for images to be aligned. By default, searches for *.JPG")
                                                                                        
    parser.add_argument("-m", "--refMaskFP",
                        required=False,
                        default=None,
                        help="OPTIONAL: The full path to a file containing "
                        "the reference image mask. The reference image mask "
                        "should be an image with dimensions = dim(keyframe) "
                        "and values equal to either 0 or 255. " 
                        "0 indicates do not select keypoints and "
                        "255 indicates to select keypoints.")
                        
    parser.add_argument("-t", "--transModel",
        required=False,
        default="Homography",
        help="OPTIONAL: Translation model to use. Select either Homography or Affine.")
                        
    parser.add_argument("-r", "--rRT",
        type = int,
        required=False,
        default=10,
        help="OPTIONAL: Ransac reprojection threshold.")
        
    parser.add_argument("-l", "--lRT",
        type = float,
        required=False,
        default=0.7,
        help="OPTIONAL: Lowe's ratio threshold.") 
        
    parser.add_argument("-s", "--summarizeTransformError",
        type = bool,
        required=False,
        default=True,
        help="OPTIONAL: Include summarized reprojection error in transformation table.") 
           
    parser.add_argument("-o", "--outdir",
        required=True,
        help="REQUIRED: The name of a subdirectory to be added to imgDir "
        "for writing aligned imagery.") 
            
    return parser.parse_args()


def main():
    # Get arguments
    args = getArgs()
    # Benchmark
    startTime = datetime.now()
    # Find image directory
    imDir = os.path.abspath(args.imgDir)
    # Find keyframe
    keyframeFullPath = os.path.abspath(args.keyframeFP)
    # Navigate to directory
    os.chdir(imDir)
    # Find and sort images
    imgFPs = glob.glob(args.globString)
    imgFPs = sorted(imgFPs)
    # Generate output directory
    outfolder = os.path.join(imDir, args.outdir)
    print("Output folder: " + outfolder)
    if not os.path.exists(outfolder):
	    os.mkdir(outfolder)
    # Identify transform
    tt = generateTransformTable(keyframeFP = keyframeFullPath, imageFPs = imgFPs, refMaskFP = args.refMaskFP, transModel = args.transModel, rRT = args.rRT, lRT = args.lRT, summarizeTransformError = args.summarizeTransformError, outdir = outfolder)
    # Save transformation table
    tt.to_csv(os.path.join(args.outdir, "transTable.csv"), index = False)
    # Final benchmark
    stopTime = datetime.now()
    print("Elapsed time: " + str(stopTime - startTime))


if __name__ == '__main__':
    main()