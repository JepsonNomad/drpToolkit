''' 
drpToolkit prep module

    prep: This module can be used to prepare a folder of time-lapse images
    for database management, image alignment, and image data extraction.


Recommended Python 3.7.

Author: Christian John
August 2021
GNU General Public License
'''

import argparse
from datetime import datetime
import os
import glob
import json
import shutil
import exifread as ef
import cv2 as cv
import numpy as np
import pandas as pd


def exifReadWrite(imgFP, sitename, plotID, outdir):
    '''
    Generate a table with exif metadata information for an image and copy that image to
    an output directory with an updated name based on metadata (SITE_PLOT_YYYY_MM_DD_HHMMSS.*).
    '''
    base, extension = os.path.splitext(imgFP)
    
    # Open imagery and extract exif info
    f = open(imgFP, 'rb')
    # Return Exif tags
    tags = ef.process_file(f)
    tags
    
    ## Access image exposure settings
    flash = tags.get('EXIF Flash').printable
    fNumber = tags.get('EXIF FNumber').printable
    apertureValue = tags.get('EXIF ApertureValue').printable
    exposureTime = tags.get('EXIF ExposureTime').printable
    isoSpeedRatings = tags.get('EXIF ISOSpeedRatings').printable
    exposureProgram = tags.get('EXIF ExposureProgram').printable
    whiteBalance = tags.get('EXIF WhiteBalance').printable
    exposureBias = tags.get('EXIF ExposureBiasValue').printable
    camModel = tags.get('Image Model').printable
    
    ## Access date-time data
    mydt = tags.get('EXIF DateTimeDigitized').printable
    # Replace special characters
    mydt_list = list(mydt)
    mydt_list[4] = "_"
    mydt_list[7] = "_"
    mydt_list[10] = "_"
    mydt_list[13] = ""
    mydt_list[16] = ""
    
    # Close connection to image
    f = None
    
    # Format filename to "newdir/HU-xx_YYYY_MM_DD_hhmmss.ext" (based on Richardson et al 2018 PhenoCam filename conventions)
    new_name = sitename+"-"+plotID+"_"+"".join(mydt_list)
    new_name_ext = new_name+extension
    
    # Generate pd.DataFrame containing the relevant image metadata
    df = pd.DataFrame(columns=["dt","filename","site","plot","camModel","fNumber", "apertureValue",
                             "exposureTime","exposureProgram","exposureBias","isoSpeedRatings",
                             "whiteBalance", "flash"])
    df.loc[0] = [mydt, new_name_ext, sitename, plotID, camModel, fNumber, apertureValue,
                 exposureTime, exposureProgram, exposureBias, isoSpeedRatings,
                 whiteBalance, flash]
    
    ## Function outputs
    # If an output directory is defined, create a name for the image and copy to directory
    if outdir is not None:
        # Specify output path for image
        new_name_abs = os.path.join(outdir, new_name_ext)
        # Finally, copy the file over to new location and naming convention
        shutil.copy(imgFP, new_name_abs)
    return df


def applyExifFolder(imgFPs, sitename, plotID, outdir):
    '''
    A foor loop to apply exifReadWrite across a folder of images and write the metadata
    to a csv file in the output directory.
    '''
    sitePlot = sitename+"_"+plotID
    exifOutPath = os.path.join(outdir,"exif.csv")

    print("Extracting exif info and copying imagery...")
    print("Site-Plot:         " + sitePlot)
    print("Site name:         " + sitename)
    print("Plot ID:           " + plotID)
    print("Image count:       " + str(len(imgFPs)))
    
    # Initialize a pandas dataframe for exif info
    exifDF = exifReadWrite(imgFP = imgFPs[0], outdir = None, sitename = sitename, plotID = plotID).iloc[0:0]
    for i in imgFPs:
        imgExif = exifReadWrite(imgFP=i, outdir=outdir, sitename=sitename, plotID=plotID)
        exifDF = exifDF.append(imgExif, ignore_index = True)
    return exifDF


def alignCrop(imgFPs, xmin, xmax, ymin, ymax, width, height, outdir):
    '''
    Crop a folder of images to a specified rectangle, and resize to a specified shape.
    Coordinates follow numpy conventions, with (xmin,ymin) at the top left of the image.
    '''
    print("Cropping and resizing imagery...")
    baseIm = cv.imread(imgFPs[0])
    baseImH, baseImW, baseImC = baseIm.shape
    if xmin is None:
        xmin = 0
    if ymin is None:
        ymin = 0
    if xmax is None:
        xmax = baseImW
    if ymax is None:
        ymax = baseImH
    if width is None:
        width = baseImW
    if height is None:
        height = baseImH
    # For each image crop and save:
    for q in imgFPs:
        imgbasename = os.path.basename(q)
        image = cv.imread(q)
        # Crop using extent parameters from above
        imNewCrop = image[ymin:ymax,xmin:xmax, :]
        # Resize using dimension parameters from above
        imNewRes = cv.resize(imNewCrop, (width, height))
        # Save image
        cv.imwrite(os.path.join(outdir,imgbasename), imNewRes)
    # Close existing connections
    hour = None
    image = None
    imNewCrop = None
    imNewRes = None


def prepROI(roiFP):
    if roiFP is None:
        ROIs_df = pd.DataFrame()
        ROIs_df["region_id"] = [1]
        ROIs_df["region_attributes"] = ["Full image"]
        ROIs_df["xvals"] = ["0,4224,4224,0,0"]
        ROIs_df["yvals"] = ["0,0,2217,2217,0"]
        roiIDs = [str(0)]
        roiSPs = ["Whole image"]
    else:
        roiDF = pd.read_csv(roiFP)
        roiIDs = [str(i) for i in roiDF['region_id'].to_numpy().astype("int")]
        roiSPs = [str(i) for i in roiDF['region_attributes'].to_numpy().astype("str")]
        roiSPs = [json.loads(i) for i in roiSPs]
        roiSPs = [i['Species'] for i in roiSPs]
        roiColumn = roiDF['region_shape_attributes'].to_numpy()
        roiInfo = [json.loads(i) for i in roiColumn]
        roiX = [np.asarray(i['all_points_x'], dtype=np.int32) for i in roiInfo]
        roiY = [np.asarray(i['all_points_y'], dtype=np.int32) for i in roiInfo]
        myROI_arr = []
        for i in range(len(roiColumn)):
            roi_vst = zip(roiX[i], roiY[i])
            roi_tup = [tuple(l) for l in roi_vst]
            roi_arr = np.array([roi_tup], dtype=np.int32)
            myROI_arr.append(roi_arr)
    return roiIDs, roiSPs, myROI_arr



# For command line scripting
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
    imDir = os.path.abspath(args.imgDir)
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