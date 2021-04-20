''' 
drpToolkit extract module

    extract: This module can be used to extract and summarize data from a folder
    of aligned time-lapse images.


Recommended Python 3.7.

Author: Christian John
April 2021
GNU General Public License
'''

import argparse
from datetime import datetime
import os
import glob
import json
import math
import numpy as np
import cv2 as cv
import pandas as pd
import drpToolkit.extract
from drpToolkit.prep import prepROI


def getArgs():
    """
    Get args from cmd line
    """
    parser = argparse.ArgumentParser(
                                description="""extract: This module can be used to 
    extract and summarize data from a folder of aligned time-lapse images.
    DEPENDENCIES: Python: numpy, opencv, pandas.""")
	
    parser.add_argument("-i", "--imgDir",
                        required=True,
                        help="REQUIRED: The full path to a directory of images "
                        "that have been aligned.")
                        
    parser.add_argument("-g", "--globString",
                        required=False,
                        default="*.JPG",
                        help="OPTIONAL: A glob-friendly filename search string "
                        "for images that have been aligned. By default, searches for *.JPG")
                                                                                        
    parser.add_argument("-s", "--sitename",
        required=False,
        default="IM",
        help="OPTIONAL: Site name. Should be a string of length 2.")
        
    parser.add_argument("-p", "--plotID",
        required=False,
        default="01",
        help="OPTIONAL: Plot ID. Should be a string of length 2.")  
           
    parser.add_argument("-r", "--ROIs",
                        required=False,
                        default=None,
                        help="OPTIONAL: The full path to a .csv file containing ROI "
                        "information. If None, full image GCC is calculated. "
                        "If not None, file should have 3 columns "
                        "named 'polygon_id', 'region_shape_attributes', and "
                        "'region_attributes'. 'region_shape_attributes' and "
                        "'region_attributes' should be populated by json-formatted "
                        "strings. See example data for sample ROI table "
                        "design.")
            
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
    # Prep ROI
    roiIDs, roiSPs, roiPolys = prepROI(roiFP = args.ROIs)
    # Prep sitePlot info
    sitePlot = args.sitename + "-" + args.plotID
    # Run GCC extraction on folder
    extract_df = foldIndices(imgFPs = imgFPs, roiIDs = roiIDs, roiSPs = roiSPs, roiPolys = roiPolys, sitePlot = sitePlot)
    extract_df.to_csv("extract.csv", index = False)
    # Final benchmark
    stopTime = datetime.now()
    print("Elapsed time: " + str(stopTime - startTime))


if __name__ == "__main__":
    main()

