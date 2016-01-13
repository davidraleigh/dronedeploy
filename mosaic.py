import cv2
import numpy as np
import glob
import csv
import os
import math

EXAMPLE_DIR = './example'
IMAGE_DETAILS_FILE = './image_details.txt'

with open(IMAGE_DETAILS_FILE, 'r') as image_details_csv:
    details_reader = csv.DictReader(image_details_csv)
    for row in details_reader:
        # important columns X est,Y est,Z est,Yaw est,Pitch est,Roll est
        img = cv2.imread(os.path.join(EXAMPLE_DIR, row['Filename']))
        height, width = img.shape[:2]
        
        yaw_radians = math.radians(row['Yaw est'])
        pitch_radians = math.radians(row['Pitch est'])
        roll_radians = math.radians(row['Roll est'])

        print float(row['X est'])
        print float(row['Y est'])
        print float(row['Z est'])


test_files = glob.glob(EXAMPLE_DIR + '*.jpg')
