import cv2
import numpy as np
import glob
import csv

EXAMPLE_DIR = './example/'
IMAGE_DETAILS_FILE = './image_details.txt'

with open(IMAGE_DETAILS_FILE, 'r') as image_details_csv:
    details_reader = csv.DictReader(image_details_csv)
    for row in details_reader:
        # important columns X est,Y est,Z est,Yaw est,Pitch est,Roll est
        print row['Filename']
        print row['X est']
        print row['Y est']
        print row['Z est']
        print row['Yaw est']
        print row['Pitch est']
        print row['Roll est']
        print float(row['X est'])
        print float(row['Y est'])
        print float(row['Z est'])
        print float(row['Yaw est'])
        print float(row['Pitch est'])
        print float(row['Roll est'])


test_files = glob.glob(EXAMPLE_DIR + '*.jpg')

for image in test_files:
    print image