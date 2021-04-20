''' 
drpToolkit prep module

    prep: This module can be used to prepare a folder of time-lapse images
    for database management, image alignment, and image data extraction.


Recommended Python 3.7.

Author: Christian John
April 2021
GNU General Public License
'''

import argparse
from datetime import datetime
import os
import glob
import shutil
import exifread as ef
import cv2 as cv
import pandas as pd
import drpToolkit.prep


def getArgs():
    """
    Get args from cmd line
    """
    parser = argparse.ArgumentParser(
                                description="""prep: This module can be used to prepare 
    a folder of time-lapse images for database management, image alignment, 
    and image data extraction.
    DEPENDENCIES: Python: exifread, opencv, pandas.""")
	
    parser.add_argument("-i", "--imgDir",
                        required=True,
                        help="REQUIRED: The full path to a directory of images "
                        "to be prepped.")
                        
    parser.add_argument("-g", "--globString",
                        required=False,
                        default="*.JPG",
                        help="OPTIONAL: A glob-friendly filename search string "
                        "for images to be prepped. By default, searches for *.JPG")
           
    parser.add_argument("-s", "--sitename",
        required=False,
        default="IM",
        help="OPTIONAL: Site name. Should be a string of length 2.")
        
    parser.add_argument("-p", "--plotID",
        required=False,
        default="01",
        help="OPTIONAL: Plot ID. Should be a string of length 2.")  
           
    parser.add_argument("--xmin",
        type=int,
        required=False,
        default=None,
        help="OPTIONAL: X coordinate of top-left of align-ready image region.")  
           
    parser.add_argument("--xmax",
        type=int,
        required=False,
        default=None,
        help="OPTIONAL: X coordinate of bottom-right of align-ready image region.")  

    parser.add_argument("--ymin",
        type=int,
        required=False,
        default=None,
        help="OPTIONAL: Y coordinate of top-left of align-ready image region.")  

    parser.add_argument("--ymax",
        type=int,
        required=False,
        default=None,
        help="OPTIONAL: Y coordinate of bottom-right of align-ready image region.")  

    parser.add_argument("--width",
        type=int,
        required=False,
        default=None,
        help="OPTIONAL: Desired width in pixels of align-ready image.")  

    parser.add_argument("--height",
        type=int,
        required=False,
        default=None,
        help="OPTIONAL: Desired height in pixels of align-ready image.")  

    parser.add_argument("-o", "--outdir",
        required=True,
        help="REQUIRED: The name of a subdirectory to be added to imgDir "
        "for writing prepped imagery and metadata.") 
            
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
    # Generate temporary exif directory
    exifolder = os.path.join(imDir, "exif")
    if not os.path.exists(exifolder):
	    os.mkdir(exifolder)
    # Generate output directory
    outfolder = os.path.join(imDir, args.outdir)
    print("Output folder: " + outfolder)
    if not os.path.exists(outfolder):
        os.mkdir(outfolder)
	# Apply exif extract and image copy to folder
    exifDF = applyExifFolder(imgFPs = imgFPs, sitename = args.sitename, plotID = args.plotID, outdir = exifolder)    
    exifDF.to_csv(os.path.join(outfolder, "exif.csv"), index = False)
    # Find and sort exif-named images
    exiFPs = glob.glob(os.path.join(exifolder, args.globString))
    exiFPs = sorted(exiFPs)
    # Crop to align-ready region
    alignCrop(imgFPs = exiFPs, xmin = args.xmin, xmax = args.xmax, ymin = args.ymin, ymax = args.ymax, width = args.width, height = args.height, outdir = outfolder)
    # Remove temporary exif subdirectory
    shutil.rmtree(exifolder)
    # Final benchmark
    stopTime = datetime.now()
    print("Elapsed time: " + str(stopTime - startTime))


if __name__ == '__main__':
    main()