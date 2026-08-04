import glob
import json
import requests
import shutil
import cv2
import os.path
from PIL import ImageFont, ImageDraw, Image
import numpy as np


def prepareForHota(groundTruth, writeFolder ): #(location of the detected/tracking results, location of the MOTA train eval, groundtruth location)
    #count = 0
    gt = open(groundTruth, "r")
    gt = [line.rstrip().split(",") for line in gt]
    gt = [[int(i) if index < 8 else float(i) for index, i in enumerate(line)] for line in gt]
    gt = [[i if index !=6  else 1 for index, i in enumerate(line)] for line in gt]
    gt = [[i if index !=7  else -1 for index, i in enumerate(line)] for line in gt]
    gt = [[i if index !=8  else -1 for index, i in enumerate(line)] for line in gt]
    gt.sort(key = lambda x: x[0])
    seqmaps = open(writeFolder, "w")
    for line in gt:
        for index, elem in enumerate(line):
            if (index != 8):
                if (index == 1):
                    seqmaps.write(str(elem -1 )+ ",")
                else:
                    seqmaps.write(str(elem )+ ",")
            else:
                seqmaps.write(str(elem)+ "\n")
    seqmaps.close()

prepareForHota("D:/PINTEL_Projects/Dataset/MOT20/MOT20/train/MOT20-05/gt/gt.txt",
                "D:/PINTEL_Projects/Dataset/MOT20/MOT20/train/MOT20-05/gt/preparedgt.txt")