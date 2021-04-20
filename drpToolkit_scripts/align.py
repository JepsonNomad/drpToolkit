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
import drpToolkit.align


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
    imDir = args.imgDir
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
    tt = generateTransformTable(keyframeFP = args.keyframeFP, imageFPs = imgFPs, refMaskFP = args.refMaskFP, transModel = args.transModel, rRT = args.rRT, lRT = args.lRT, outdir = outfolder)
    # Save transformation table
    tt.to_csv(os.path.join(args.outdir, "transTable.csv"), index = False)
    # Final benchmark
    stopTime = datetime.now()
    print("Elapsed time: " + str(stopTime - startTime))


if __name__ == '__main__':
    main()