''' 
drpToolkit extract module

    extract: This module can be used to extract and summarize data from a folder
    of aligned time-lapse images.


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
import math
import numpy as np
import cv2 as cv
import pandas as pd
from drpToolkit.prep import prepROI


def imgGCC(img):
    '''
    Converts a 3-band BGR image to a 1-band GCC image
    '''
    # Lots of runtime warnings can be expected here due to black space so I suppressed
    # those for this function specifically
    np.seterr(divide='ignore', invalid='ignore')
    blu, grn, red = cv.split(img.astype('float'))
    GCC = grn / (blu + grn + red)
    return(GCC)
    

def imgNDSI(img):
    '''
    Calculates pseudo-NDSI for an image based on Hinkler et al 2002 Int J Remote Sensing.
    Input image (img) should follow opencv channel order of BGR.
    '''
    # np.seterr(divide='ignore', invalid='ignore')
    #### Define functions ----
    # RGB is the mean digital number for each pixel.
    def calcRGB(img):    
        return((img.sum(axis = 2))/3)
    # RGBhigh is a recalculated image, reflective of raw image brightness
    def calcRGBhigh(img):  
        blu, grn, red = cv.split(img.astype('float'))
        R3 = red**3    
        B3 = blu**3    
        G = grn
        RGBhigh = (B3/R3)*G
        return(RGBhigh)
    # Tau is an expression of the overall brightness of a photo. Can just use 
    # mean(RGBhigh) if a and b are not derived
    def calcTau(RGBhigh, fullTau = True, a = -0.0125, b = 1.2875):
        '''
        If fullTau = True (the default) the parameters a and b 
        are included along with the constant multiplier described in Hinkler et al 2002.
        Otherwise, Tau is set to the mean of RGBhigh.
        '''
        RGBhighmean = np.ma.masked_invalid(RGBhigh).mean()
        if fullTau == True:
            Tau = 200*(a*RGBhighmean + b)
        else:
            Tau = RGBhighmean
        return(Tau)
    # MIRreplacement is a stand-in value for the middle infrared band used in NDSI calculation
    # RGBmax is the highest RGB value in the image
    def calcMIRrep(tau, RGBmax, RGB):    
        return ((tau**4)*RGBmax)/(RGB**4)
    # RGBNDSI is a pseudo-NDSI value based on a substituted mid-IR value
    def calcNDSI(RGB, MIR):    
        NDSI = (RGB - MIR)/(RGB + MIR)    
        return(NDSI)
    #### Calculate snow cover ----
    imflt = img.astype('float')
    RGB = calcRGB(imflt)
    RGBhigh = calcRGBhigh(imflt)
    tau = calcTau(RGBhigh=RGBhigh, fullTau = False)
    RGBmax = np.max(RGB)
    MIRrep = calcMIRrep(tau, RGBmax, RGB)
    rgbNDSI = calcNDSI(RGB, MIRrep)
    return rgbNDSI


def imgIndices(imgFP, roiIDs, roiSPs, roiPolys, dt, CamID, outname = None):
    '''
    Extracts radiometric indices from the regions in an image.
    '''
    if len(roiIDs) != len(roiSPs):
        sys.exit("Number of ROI ID's does not correspond to length of ROI species list")
    if len(roiIDs) != len(roiPolys):
        sys.exit("Number of ROI ID's does not correspond to number of ROI polygons")
    # First, load image using openCV default settings (BGR)
    image = cv.imread(imgFP)
    
    gcc = imgGCC(image)
    ndsi = imgNDSI(image)
    
    # Create empty lists for GCC, ROI ID, date-time, species, and filenames
    GCCs = []
    NDSIs = []
    ROIs = []
    dts = []
    spps = []
    filenames = []
    totalPix = []
    non0Pix = []
    
    for j in range(len(roiIDs)):
        polyj = roiPolys[j]
        # print(polyj)
        # print(type(polyj))
        # Isolate a single ROI - doesn't have to be rectangular thanks to:
        # https://stackoverflow.com/a/15343106/5090454
        # Generate mask layer based on input image
        mask = np.zeros(image.shape, dtype=np.uint8)
        height, width, channels = image.shape
        ignoreMask = (255,)*channels
        cv.fillPoly(mask, polyj, ignoreMask)
        # Use mask to extract pixel values
        gcc_ext = np.extract(mask[:,:,2], gcc)
        ndsi_ext = np.extract(mask[:,:,1], ndsi)
        # Generate DataFrame with index values
        cols_ext = pd.DataFrame({"GCC":gcc_ext,"NDSI":ndsi_ext})
        # Only include values if not all 3 bands are 0 (all bands = 0 when image is realigned and ROI is no longer in frame)
        cols_INREGION = cols_ext[cols_ext.sum(axis = 1) != 0]
        # Generate nonzero pixel count for ROI
        ROI_non0Count = len(cols_ext[cols_ext.sum(axis = 1) != 0])
        # Generate total pixel count for ROI
        ROI_pixelCount = len(cols_ext)
        # Find mean index values in region
        roi_gccMean = np.mean(cols_ext.GCC.to_numpy())
        roi_ndsiMean = np.mean(cols_ext.NDSI.to_numpy())
        GCCs.append(roi_gccMean)
        NDSIs.append(roi_ndsiMean)
        ROIs.append(roiIDs[j])
        dts.append(dt)
        spps.append(roiSPs[j])
        filenames.append(os.path.basename(imgFP))
        totalPix.append(ROI_pixelCount)
        non0Pix.append(ROI_non0Count)
    # Create data.frame of date and GCC data for position time series
    df = pd.DataFrame({'filename':filenames, 'date':dts, 'GCC':GCCs, 'NDSI':NDSIs, 'regionID':ROIs, 'species':spps, 'totalPix':totalPix, 'non0Pix':non0Pix})
    df['position'] = np.repeat(CamID, len(GCCs))
    # Save as csv in designated output directory
    if outname is not None:
        df.to_csv(outname)
    return df


def foldIndices(imgFPs, roiSPs, roiIDs, roiPolys, sitePlot):
    '''
    Iterate the imgIndices function over a folder of images and aggregate the output
    into a tidy pandas DataFrame.
    '''
    GCC_df = pd.DataFrame(columns=['filename','date', 'GCC', 'NDSI', 'regionID', 'species', 'position'])
    imgCount = len(imgFPs)
    imgIndex = 0
    for i in imgFPs:
        # print(i)
        imgIndex += 1
        if imgIndex % 50 == 0:
        	print("Processing image " + str(imgIndex) + " of " + str(imgCount))
        # Absolute path of image
        # Output name for dataframe
        filename = os.path.basename(i)
        base, ext = os.path.splitext(filename)
        # Date-time stamp for image - RELIES ON FILENAME CONVENTION!
        dt = datetime(int(base[6:10]),int(base[11:13]),int(base[14:16]), 
                      int(base[17:19]),int(base[19:21]),int(base[21:23]))
        imdf = imgIndices(imgFP = i, roiIDs = roiIDs,  roiSPs = roiSPs, roiPolys = roiPolys, dt = dt, CamID = sitePlot, outname = None)
        GCC_df = GCC_df.append(imdf)
    return GCC_df


# For command line scripting
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
    imDir = os.path.abspath(args.imgDir)
    # Prep ROI
    roiIDs, roiSPs, roiPolys = prepROI(roiFP = os.path.abspath(args.ROIs))
    # Navigate to directory
    os.chdir(imDir)
    # Find and sort images
    imgFPs = glob.glob(args.globString)
    imgFPs = sorted(imgFPs)
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