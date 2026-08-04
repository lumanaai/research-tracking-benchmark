import glob
import json
import requests
import shutil
# import cv2
import os.path
from PIL import ImageFont, ImageDraw, Image
import numpy as np


def prepareForHota(readFolder, writeFolder, groundTruth, ): #(location of the detected/tracking results, location of the MOTA train eval, groundtruth location)
    #count = 0
    seqmaps = open(writeFolder+"/gt/mot_challenge/seqmaps/MOT15-train.txt", "w")
    seqmaps.write("name\n")
    for file in glob.iglob(readFolder+"**/*.txt", recursive = True):
        #seqmaps creation
        folderName = file[file.rfind("/")+ 1:file.rfind(".")]
        seqmaps.write(folderName+"\n")
        
        #Make the folder and gt folder
        os.mkdir(writeFolder+"/gt/mot_challenge/MOT15-train/"+folderName)
        os.mkdir(writeFolder+"/gt/mot_challenge/MOT15-train/"+folderName+"/gt")
        
        #Make seqinfo.ini file
        gtFile = open(groundTruth, "r")
        array = []
        for line in gtFile:
            array.append(int(line.split(",")[0]))
        seqinfo = open(writeFolder+"/gt/mot_challenge/MOT15-train/"+folderName+"/seqinfo.ini", "w")
        seqinfo.write("[Sequence]\n")
        seqinfo.write("name="+folderName+"\n")
        seqinfo.write("imDir=img1\n")
        seqinfo.write("frameRate=25\n")
        seqinfo.write("seqLength="+str(max(array))+"\n")
        seqinfo.write("imWidth=1920\n")
        seqinfo.write("imHeight=1080\n")
        seqinfo.write("imExt=.jpg\n")
        seqinfo.close()
        
        #Copy ground truth to gt
        original = groundTruth
        target = writeFolder+"/gt/mot_challenge/MOT15-train/"+folderName+"/gt/gt.txt"
        shutil.copyfile(original, target)
        
        #Make trackers
        trackerTxt = writeFolder + "/trackers/mot_challenge/MOT15-train/MPNTrack/data/" + folderName + ".txt"
        shutil.copyfile(file,trackerTxt)
        
        
    seqmaps.close()
    


#clear gt/mot_challenge/MOT15-train location
#location of the detected/tracking results, location of the MOTA train eval, groundtruth location
# prepareForHota("D:/Pintel_Projects/TrackEval/TrackEval/tracker_test/tracker_byte", 
#                "D:/Pintel_Projects/TrackEval/TrackEval/data",
#                 "D:/Pintel_Projects/TrackEval/TrackEval/tracker_test/tracker_gt/TownCentre-groundtruth.txt")

prepareForHota("/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/TrackEval/TrackEval/tracker_test/tracker_day_right", 
               "/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/TrackEval/TrackEval/data",
               "/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/TrackEval/TrackEval/tracker_test/tracker_day_right/gt.txt")
    
# prepareForHota("/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/TrackEval/TrackEval/tracker_test/tracker_byte", 
#             "/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/TrackEval/TrackEval/data",
#             "/media/ezhdeha/B2CA8AACCA8A6C83/Pintel_Projects/TrackEval/TrackEval/tracker_test/tracker_gt/TownCentre-groundtruth.txt")


