''' 
drpToolkit extract module

    panelize: This module can be used to generate timelapse-friendly summary images that
    sport an aligned image alongside GCC and rgbNDSI plots according to timepoint.


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
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import gc
from drpToolkit.prep import prepROI


def imgPanelize(img, imgDate, roiPolys, indicesDF, outname = None):
    '''
    Generates a folder of time-lapse-friendly, panelized image NDSI and GCC subplots.
    Requires an input pd.DataFrame() with columns called exactly 'date', 'NDSI', and 
    'GCC'. indicesDF['date'] should be in datetime format. roiPolys should be formatted 
    as per other functions in this module, i.e. a list of arrays with coordinate pairs 
    of ROI corners.
    '''
    dateRange = [np.min(indicesDF['date']),np.max(indicesDF['date'])]
    ndsiRange = [np.min(indicesDF['NDSI'])-0.1,np.max(indicesDF['NDSI'])+0.1]
    gccRange = [np.min(indicesDF['GCC'])-0.1,np.max(indicesDF['GCC'])+0.1]
    # Cut down the dataframe to relevant dates
    indicesDR = indicesDF.loc[(indicesDF['date'] <= imgDate)]
    # Get date formatting prepared
    date_form = mdates.DateFormatter("%Y-%m")
    locator = mdates.AutoDateLocator(minticks=3, maxticks=5)
    # Use gridspec to panelize the plot
    fig = plt.figure(constrained_layout=True)
    gs = fig.add_gridspec(7, 7)
    ax1 = fig.add_subplot(gs[0:5,0:7])
    ax1.imshow(img[:,:,[2,1,0]]) # Convert to RGB for plt
    for i in range(len(roiPolys)):
        pol = roiPolys[i][0]
        pol = np.append(pol,pol[0]).reshape((2,len(pol)+1), order = 'F')
        roiXs, roiYs = zip(*pol.T)
        ax1.plot(roiXs,roiYs) 
    ax1.axis('off')
    ax2 = fig.add_subplot(gs[5:7,0:3])
    for i in range(len(roiPolys)):
        polyDF = indicesDR.loc[(indicesDR['regionID'] == i)]
        ax2.plot_date(polyDF['date'],polyDF['NDSI'], ms = 1)
    ax2.set_xlim(dateRange)
    ax2.set_ylim(ndsiRange)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(date_form)
    ax2.set_ylabel('rgbNDSI')
    ax3 = fig.add_subplot(gs[5:7,4:7])
    for i in range(len(roiPolys)):
        polyDF = indicesDR.loc[(indicesDR['regionID'] == i)]
        ax3.plot_date(polyDF['date'],polyDF['GCC'], ms = 1)
    ax3.set_xlim(dateRange)
    ax3.set_ylim(gccRange)
    ax3.xaxis.set_major_locator(locator)
    ax3.xaxis.set_major_formatter(date_form)
    ax3.set_ylabel('GCC')
    if outname is not None:
        fig.savefig(outname)
    else:
        plt.imshow(fig)
    plt.cla() 
    plt.clf() 
    plt.close('all')


def foldPanelize(imgFPs, roiPolys, indicesDF, outdir):
    '''
    Iterate the panelizeImg function over a folder of images.
    '''
    imgCount = len(imgFPs)
    # Ensure proper formatting
    indicesDF['date'] = pd.to_datetime(indicesDF['date'])
    indicesDF['regionID'] = pd.to_numeric(indicesDF['regionID'])
    indicesDF['regionID']
    imgIndex = 0
    for i in imgFPs:
        # print(i)
        imgIndex += 1
        if imgIndex % 50 == 0:
        	print("Panelizing image " + str(imgIndex) + " of " + str(imgCount))
        # Absolute path of image
        # Output name for dataframe
        filename = os.path.basename(i)
        base, ext = os.path.splitext(filename)
        # Date-time stamp for image - RELIES ON FILENAME CONVENTION!
        dt = datetime(int(base[6:10]),int(base[11:13]),int(base[14:16]), 
                      int(base[17:19]),int(base[19:21]),int(base[21:23]))
        img = cv.imread(i)
        outname = os.path.join(outdir, filename)
        imgPanelize(img = img, imgDate = dt, roiPolys = roiPolys, indicesDF = indicesDF, outname = outname)
        # Close connections
        del img
        plt.cla() 
        plt.clf() 
        plt.close('all')
        gc.collect()

  
# For command line scripting
def getArgs():
    """
    Get args from cmd line
    """
    parser = argparse.ArgumentParser(
                                description="""panelize: This module can be used to 
                                generate timelapse-friendly summary images that sport 
                                an aligned image alongside GCC and rgbNDSI plots 
                                according to timepoint.
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
                                                                                        
    parser.add_argument("-t", "--table",
                        required=True,
                        default="extract.csv",
                        help="REQUIRED: A filename for the extracted index table "
                        "generated by extract.py")

    parser.add_argument("-r", "--ROIs",
                        required=False,
                        default=None,
                        help="OPTIONAL: The full path to a .csv file containing ROI "
                        "information. If None, full image GCC is calculated. "
                        "If not None, file should have 3 columns "
                        "named 'polygon_id','region_shape_attributes', and "
                        "'region_attributes'. See example data for sample ROI table "
                        "design.")
            
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
    # Import indices DataFrame
    dfPath = args.table
    extract_df = pd.read_csv(dfPath)
    # Prep ROI
    roiIDs, roiSPs, roiPolys = prepROI(roiFP = args.ROIs)
    # Generate output plots
    panelFolder = os.path.join(imDir, "panelized")
    if not os.path.exists(panelFolder):
	    os.mkdir(panelFolder)
    foldPanelize(imgFPs = imgFPs, roiPolys = roiPolys, indicesDF = extract_df, outdir = panelFolder)
    # Final benchmark
    stopTime = datetime.now()
    print("Elapsed time: " + str(stopTime - startTime))


if __name__ == "__main__":
    main()
