# Need to set up out dir and also how it is being split. Bounding boxes?

# Some basic setup
# Setup detectron2 logger
from sys import argv
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import matplotlib.pyplot as plt
import numpy as np
import cv2
from glob import glob
import subprocess
from shlex import quote
import csv

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.structures import BoxMode #I added this
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import statistics 

import random
from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg
import os
import traceback

numdir = argv[1]
album = argv[2]

# Set Up Models
# the cfg object here is an instantiation of the model. The merge_from_file function gets arguments from a default YAML 
# file to configure the model. The functions that follow update certain arguments that were set to default from the YAML file.
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = "/home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/mexico_5_column_weights.pth" #SET UP WEIGHTS HERE
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 5 # 5 classes (5 columns in this instance, but you may have more depending on what you are doing)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set the testing threshold for this model
predictor = DefaultPredictor(cfg)

#FUNCTIONS
#This function returns a list of vertical lines found within the image passed to the function. 
def get_vertical_lines(img, width=385, line_height=2000, circle = 155): #this function takes as parameter and image and default integers. It returns a list. 
  ys=[]
  keepers=[]
  n=0
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
  edges = ~cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,circle,2)
  kernel = np.ones((3, 3), np.uint8)
  th2 = cv2.erode(edges, kernel, iterations=1)
  kernel = np.ones((1, 7), np.uint8)
  th3 = cv2.dilate(th2, kernel, iterations=1)
  lines = cv2.HoughLines(th3,1,np.pi/180, line_height) 
  for line in range(len(lines)):
      if lines[line][0][1]>-.1 and lines[line][0][1]<.1:
          keepers.append(lines[line])
          n+=1
  for line2 in range(n):
      for rho,theta in keepers[line2]:
          b = np.sin(theta)
          y0 = b*rho
          a = np.cos(theta)
          x0 = a*rho
          x1 = int(x0 + 30*(-b))
          y1 = int(y0 + 30*(a))
          x2 = int(x0 - 30*(-b))
          y2 = int(y0 - 30*(a))
          slope = (y2-y1) / (x2-x1)
          intercept = y1 - (slope * x1)
          side = slope * width + intercept
          ys.append(intercept)
          ys.append(side)
  return ys

#This function returns a list of horizontal lines found in the image passed into the function. 
def get_horizontal_lines(img, width=385, line_width=150, circle = 155): #this function takes as parameter and image and default integers. It returns a list. 
  ys=[]
  keepers=[]
  n=0
  gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) #converts image to grayscale
  edges = ~cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,circle,2)#applies threshold on image
  kernel = np.ones((3, 3), np.uint8)
  th2 = cv2.erode(edges, kernel, iterations=1)
  kernel = np.ones((7, 1), np.uint8)
  th3 = cv2.dilate(th2, kernel, iterations=1)
  lines = cv2.HoughLines(th3,1,np.pi/180, line_width)
  for line in range(len(lines)):
      if lines[line][0][1]>1.45 and lines[line][0][1]<1.7:
          keepers.append(lines[line])
          n+=1
  for line2 in range(n):
      for rho,theta in keepers[line2]:
          b = np.sin(theta)
          y0 = b*rho
          a = np.cos(theta)
          x0 = a*rho
          x1 = int(x0 + 30*(-b))
          y1 = int(y0 + 30*(a))
          x2 = int(x0 - 30*(-b))
          y2 = int(y0 - 30*(a))
          slope = (y2-y1) / (x2-x1)
          intercept = y1 - (slope * x1)
          side = slope * width + intercept
          ys.append(intercept)
          ys.append(side)
  return ys


def crop_bot(img, width = 385, line_width_crop = 300):
    temp=img[-50:,0:width]
    try:
      ys = get_horizontal_lines(temp, line_width = line_width_crop)
      return img[:img.shape[0]-50+int(np.mean(ys)),0:width]
    except:
      return img


def make_snippets(img, ys, rows = 50, pixels_per_row = 60, pixels_on_either_side = 15, file_path = "", column = "lit", add_to_end = 0):
  start = 0
  for y in range(rows):
    finish = start + pixels_per_row
    x_check = start - pixels_on_either_side
    x_check2 = start + pixels_on_either_side
    y_check = finish - pixels_on_either_side
    y_check2 = finish + pixels_on_either_side
    newlist = [x for x in ys if (x > x_check) & (x < x_check2)]
    newlist2 = [x for x in ys if (x > y_check) & (x < y_check2)]
    if len(newlist)!=0:
      start = round(statistics.median(newlist))
    if len(newlist2)!=0:
      finish = round(statistics.median(newlist2))
    if y==rows-1:
      snippet=img[start:]
    elif y!=rows-1:
      snippet=img[start:finish]
    start = finish
    cv2.imwrite(file_path + "_" + column + "_row_" + str(y+1) + ".jpg", snippet)

# CODE THAT DOES THE SEGMENTATION
bad=[]
files = os.listdir()
#files = random.sample(os.listdir(), 4)
for d in files:
    if d[-4:] == ".jpg":
      try:
        out_dir = "/home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/segments/snippets/{}".format(numdir + "/" + album)
        im = cv2.imread(d)
        outputs = predictor(im)
        objects = outputs["instances"].pred_classes
        boxes = outputs["instances"].pred_boxes
        masks = outputs["instances"].pred_masks
        boxes_np = boxes.tensor.cpu().numpy()
        obj_np = objects.cpu().numpy()
        masks_np = masks.cpu().numpy()
        m = 0
        for box in range(len(boxes_np)):
          left = int(boxes_np[box][0])
          top = int(boxes_np[box][1])
          right = int(boxes_np[box][2])
          bottom = int(boxes_np[box][3])
          cropped_array = im[top:bottom,left:right]
          mask = masks_np[m][top:bottom,left:right]
          h , w = mask.shape
          tl = int(np.argwhere(mask[200]==True)[0])
          bl = int(np.argwhere(mask[h-200]==True)[0])
          white1 = np.zeros([h,w,3],dtype=np.uint8)
          white1.fill(255)
          white2 = np.zeros([h,w,3],dtype=np.uint8)
          white2.fill(255)
          change = (tl-bl)/h
          white3= (cropped_array * mask[..., None]) + (white1 * ~mask[..., None]) 
          for i in range(h):
            start = int(tl - i*change)
            if len(np.argwhere(mask[i]==True))>0:
              last = int(np.argwhere(mask[i]==True)[-1])
            elif len(np.argwhere(mask[i]==True))==0:
              last = w-start
            white2[i][0:last-start] = white3[i][start:last]
          if obj_np[m] == 0:
              white3=white2[:,0:60]
              outputs2 = predictor2(white3)
              boxes2 = outputs2["instances"].pred_boxes
              boxes_np2 = boxes2.tensor.cpu().numpy()
              bottom2 = int(boxes_np2[0][3])
              no_top=white3[bottom2:,:]
              no_bot_or_top = crop_bot(no_top, width = 60, line_width_crop= 45)
              no_bot_or_top = cv2.resize(no_bot_or_top,(60,3000))
              ys = get_horizontal_lines(no_bot_or_top,width=60, line_width=45)
              make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,  pixels_on_either_side = 15, file_path = out_dir + "/" + d[:-4], column= 'lit1')
          elif obj_np[m] == 1:
              white3=white2[:,0:60]
              outputs2 = predictor2(white3)
              boxes2 = outputs2["instances"].pred_boxes
              boxes_np2 = boxes2.tensor.cpu().numpy()
              bottom2 = int(boxes_np2[0][3])
              no_top=white3[bottom2:,:]
              no_bot_or_top = crop_bot(no_top, width = 60, line_width_crop= 45)
              no_bot_or_top = cv2.resize(no_bot_or_top,(60,3000))
              ys = get_horizontal_lines(no_bot_or_top,width=60, line_width=45)
              make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,  pixels_on_either_side = 15, file_path = out_dir + "/" + d[:-4], column= 'lit2')
          elif obj_np[m] == 2:
              white3=white2[:,0:60]
              outputs2 = predictor2(white3)
              boxes2 = outputs2["instances"].pred_boxes
              boxes_np2 = boxes2.tensor.cpu().numpy()
              bottom2 = int(boxes_np2[0][3])
              no_top=white3[bottom2:,:]
              no_bot_or_top = crop_bot(no_top, width = 60, line_width_crop= 45)
              no_bot_or_top = cv2.resize(no_bot_or_top,(60,3000))
              ys = get_horizontal_lines(no_bot_or_top,width=60, line_width=45)
              make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60,  pixels_on_either_side = 15, file_path = out_dir + "/" + d[:-4], column= 'lang1')
          elif obj_np[m] == 3:
              white3=white2[:,0:350]
              outputs2 = predictor2(white3)
              boxes2 = outputs2["instances"].pred_boxes
              boxes_np2 = boxes2.tensor.cpu().numpy()
              bottom2 = int(boxes_np2[0][3])
              no_top=white3[bottom2:,:]
              no_bot_or_top = crop_bot(no_top, line_width_crop=265)
              no_bot_or_top = cv2.resize(no_bot_or_top,(350,3000))
              ys = get_horizontal_lines(no_bot_or_top,width=350, line_width=265)
              make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60, pixels_on_either_side = 15, file_path = out_dir + "/" + d[:-4], column= 'lang2')
          elif obj_np[m] == 4:
              white3=white2[:,0:225]
              outputs2 = predictor2(white3)
              boxes2 = outputs2["instances"].pred_boxes
              boxes_np2 = boxes2.tensor.cpu().numpy()
              bottom2 = int(boxes_np2[0][3])  
              no_top=white3[bottom2:,:]
              no_bot_or_top = crop_bot(no_top, line_width_crop=300)
              no_bot_or_top = cv2.resize(no_bot_or_top,(225,3000))
              ys = get_horizontal_lines(no_bot_or_top,width=225, line_width=150)
              make_snippets(no_bot_or_top, ys, rows=50, pixels_per_row=60, pixels_on_either_side = 15, file_path = out_dir + "/" + d[:-4], column= 'rel')
          m += 1   
      except:
        bad.append(d)
        traceback.print_exc() 
        print("image failed: " + d)
        pass

print("Percent Error: " + str(len(bad)/len(files)))
print(bad)
with open(f'/home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/error_img/mexico_error_{numdir}.csv', 'a') as output:
 # /home/jmorri33/fsl_groups/fslg_census/compute/projects/Mexico_Census/error_img/mexico_error_62.csv
# ../../../../error_img
    writer = csv.writer(output, delimiter=',')
    writer.writerow(bad)

