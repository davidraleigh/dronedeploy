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
        Maffine = np.float32([[1, 0, -width/2], [0, 1, -height/2]])
        dst = cv2.warpAffine(img, Maffine, (width, height))


        
        
        yaw_radians = math.radians(float(row['Yaw est']))
        pitch_radians = math.radians(float(row['Pitch est']))
        roll_radians = math.radians(float(row['Roll est']))

        # # yaw rotation
        RotZ = np.matrix([[math.cos(yaw_radians), -math.sin(yaw_radians), 0],
                          [math.sin(yaw_radians), math.cos(yaw_radians) , 0],
                          [0                    , 0                     , 1]])

        RotY = np.matrix([[math.cos(pitch_radians) , 0, math.sin(pitch_radians)],
                          [0                       , 1, 1                      ],
                          [-math.sin(pitch_radians), 0, math.cos(pitch_radians)]])

        RotX = np.matrix([[1, 0                     , 0                      ],
                          [0, math.cos(roll_radians), -math.sin(roll_radians)],
                          [0, math.sin(roll_radians), math.cos(roll_radians) ]])

        Rot = RotZ #* RotY * RotX

        dst = cv2.warpPerspective(dst, Rot, (width, height))

        Maffine = np.float32([[1, 0, width/2], [0, 1, height/2]])
        dst = cv2.warpAffine(dst, Maffine, (width, height))

        dst = cv2.resize(dst, (width/4, height/4), interpolation = cv2.INTER_CUBIC)
        cv2.imshow('img',dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        print float(row['X est'])
        print float(row['Y est'])
        print float(row['Z est'])


test_files = glob.glob(EXAMPLE_DIR + '*.jpg')
