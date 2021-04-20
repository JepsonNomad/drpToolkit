''' 
drpToolkit align module

    align: This module can be used to align a folder of time-lapse images
    to a keyframe using affine or homography tranformations.


Recommended Python 3.7.

Author: Christian John
April 2021
GNU General Public License
'''

import argparse
from datetime import datetime
import os
import glob
import numpy as np
import cv2 as cv
import pandas as pd


def estimateTransform(newImage, refImage, refMask, transModel, rRT, lRT):
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
    else:
        h = None
    return h


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


def generateTransformTable(keyframeFP, imageFPs, refMaskFP, transModel, rRT, lRT, outdir):
    '''
    Generate a table with proposed transformation matrices for a set of imagery.
    Requires the filepath of a keyframe (keyframe) and a list of filepaths for
    imagery to be aligned (imageFPs). Also requires transformation estimation
    parameters passed directly to estimateTransform(). An output folder (outPath) 
    should point to a place to save aligned images, the keyframe mask, and 
    transformation table.
    '''
    print("Generating transformation table...")
    ## Set up transform table and initialize transformation matrix
    if transModel == "Affine":
        hCols = ["h"+str(i) for i in range(1,7)]
        hLast = np.array([[1.,0.,0.],[0.,1.,0.]])
    elif transModel == "Homography":
        hCols = ["h"+str(i) for i in range(1,10)]
        hLast = np.array([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]])
    
    ttColsBase = ["fileName", "transModel"]
    ttColNames = ttColsBase + hCols
    # print(ttColNames)
    transformTable = pd.DataFrame(columns = ttColNames)
    # print(transformTable)
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
        h = estimateTransform(newImage = newImage, refImage = refImage, refMask = refMask, transModel = transModel, rRT = rRT, lRT = lRT)
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
        dfRow = [imgBN, transModel] + hFlat
        dfSeries = pd.Series(dfRow, index = ttColNames)
        # Append image name, transformation model, and transformation matrix to table
        transformTable = transformTable.append(dfSeries, ignore_index=True)
        ## Save transformed image
        outpath = os.path.join(outdir, imgBN)
        # print("Output image path: " + outpath)
        cv.imwrite(outpath, imgReg)
    
    ## Return the transformation table
    return transformTable    


