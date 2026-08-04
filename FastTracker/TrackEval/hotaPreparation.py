import glob
import json
import requests
import shutil
import cv2
import os.path
from PIL import ImageFont, ImageDraw, Image
import numpy as np
import argparse
import os

def make_parser():
    parser = argparse.ArgumentParser("MOTA-HOTA Preparation")
    parser.add_argument("-d", "--data", type=str, default=".\\TrackEval\\tracker_test\\tracker_mot17_signle\\data\\MOT17-02-DPM.txt")
    parser.add_argument("-g", "--gt", type=str, default=".\\TrackEval\\tracker_test\\tracker_mot17_signle\\gt\\gt.txt")
    parser.add_argument("-p", "--path", type=str, default=".\\TrackEval\\data")
    return parser

def prepareForHota(file, writeFolder, groundTruth, ): #(location of the detected/tracking results, location of the MOTA train eval, groundtruth location)
    #count = 0
    seqmaps_path = os.path.join(writeFolder, "gt", "mot_challenge", "seqmaps", "MOT15-train.txt")
    if os.path.exists(seqmaps_path):
        os.remove(seqmaps_path)
    seqmaps = open(seqmaps_path, "w")
    seqmaps.write("name\n")

    #seqmaps creation
    folderName = file[file.rfind("\\")+ 1:file.rfind(".")]
    seqmaps.write(folderName+"\n")
    
    #Make the folder and gt folder
    target_dir = os.path.join(writeFolder, "gt", "mot_challenge", "MOT15-train", folderName)
    # Remove the folder if it already exists
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
    os.mkdir(writeFolder+"gt\\mot_challenge\\MOT15-train\\"+folderName)
    os.mkdir(writeFolder+"gt\\mot_challenge\\MOT15-train\\"+folderName+"\\gt")
    
    #Make seqinfo.ini file
    gtFile = open(groundTruth, "r")
    array = []
    for line in gtFile:
        array.append(int(line.split(",")[0]))
    seqinfo = open(writeFolder+"gt\\mot_challenge\\MOT15-train\\"+folderName+"\\seqinfo.ini", "w")
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
    target = writeFolder+"gt\\mot_challenge\\MOT15-train\\"+folderName+"\\gt\\gt.txt"
    shutil.copyfile(original, target)
    
    
    # Delete all files in the folder (but not the folder itself)
    target_dir = os.path.join(writeFolder, "trackers", "mot_challenge", "MOT15-train", "MPNTrack", "data")
    if os.path.exists(target_dir):
        for filename in os.listdir(target_dir):
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
    #Make trackers
    trackerTxt = writeFolder + "trackers\\mot_challenge\\MOT15-train\\MPNTrack\\data\\" + folderName + ".txt"
    shutil.copyfile(file,trackerTxt)
        
        
    seqmaps.close()
    

if __name__ == "__main__":
    args = make_parser().parse_args()

    prepareForHota(args.data,
                args.path,
                args.gt)


