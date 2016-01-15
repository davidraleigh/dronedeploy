import cv2
import numpy as np
import glob
import csv
import os
import math
from scipy import ndimage
from scipy import misc

#testing 
import matplotlib.pyplot as plt

EXAMPLE_DIR = './example'
IMAGE_DETAILS_FILE = './image_details.txt'

def order_points(pts):
    # initialzie a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype = "float32")
 
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis = 1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
 
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis = 1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
 
    # return the ordered coordinates
    return rect

def pitch_roll_pts(height, width, RotY, RotX):
    pts = np.matrix([[0,      0,  width,  width], 
                    [height,  0,  0,      height],
                    [1,       1,  1,      1]])

    print pts
    pts = RotY * RotX * pts

    coords = np.array([(pts[0,0], pts[1, 0]), 
              (pts[0,1], pts[1, 1]), 
              (pts[0,2], pts[1, 2]), 
              (pts[0,3], pts[1, 3])])

    return order_points(coords)

def create_src_rect(height, width):
    pts = np.matrix([[0,      0,  width,  width], 
                    [height,  0,  0,      height],
                    [1,       1,  1,      1]])

    coords = np.array([(pts[0,0], pts[1, 0]), 
              (pts[0,1], pts[1, 1]), 
              (pts[0,2], pts[1, 2]), 
              (pts[0,3], pts[1, 3])])

    return order_points(coords)

def four_point_transform(image, RotY, RotX):
    height, width = img.shape[:2]

    src_rect = create_src_rect(height, width)
    # obtain a consistent order of the points and unpack them
    # individually
    dst_rect = pitch_roll_pts(height, width, RotY, RotX)
    (tl, tr, br, bl) = dst_rect
 
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
 
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
 
    # compute the perspective transform matrix and then apply it
    print src_rect
    print dst_rect
    M = cv2.getPerspectiveTransform(dst_rect, src_rect)
    print M
    M = cv2.findHomography(src_rect, dst_rect, cv2.RANSAC, 5.0)
    print M
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

with open(IMAGE_DETAILS_FILE, 'r') as image_details_csv:
    details_reader = csv.DictReader(image_details_csv)


    for row in details_reader:
        # important columns X est,Y est,Z est,Yaw est,Pitch est,Roll est
        img = cv2.imread(os.path.join(EXAMPLE_DIR, row['Filename']))
        height, width = img.shape[:2]
        print height
        print width

        yaw_degrees = float(row['Yaw est']) 
        pitch_radians = math.radians(float(row['Pitch est']))
        roll_radians = math.radians(float(row['Roll est']))

        # yaw rotation (perpendicular to ground)
        # RotZ = np.matrix([[math.cos(yaw_radians), -math.sin(yaw_radians), 0],
        #                   [math.sin(yaw_radians), math.cos(yaw_radians) , 0],
        #                   [0                    , 0                     , 1]])

        # pitch rotation (axis through wing perpendicular to flight path)
        RotY = np.matrix([[math.cos(pitch_radians) , 0, math.sin(pitch_radians)],
                          [0                       , 1, 1                      ],
                          [-math.sin(pitch_radians), 0, math.cos(pitch_radians)]])

        # roll rotation (axis through direction of flight path)
        RotX = np.matrix([[1, 0                     , 0                      ],
                          [0, math.cos(roll_radians), -math.sin(roll_radians)],
                          [0, math.sin(roll_radians), math.cos(roll_radians) ]])

        warped_image = four_point_transform(img, RotY, RotX)
        height, width = warped_image.shape[:2]
        print height
        print width
        misc.imsave('face.png', warped_image)

        dst = cv2.resize(warped_image, (width/4, height/4), interpolation = cv2.INTER_CUBIC)



        rotate_image= ndimage.rotate(warped_image, yaw_degrees, (1, 0))
        height, width = rotate_image.shape[:2]
        print height
        print width



        cv2.imshow('img', dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # print pitch_roll_pts(height, width, RotY, RotX)
        # Rot = RotZ #* RotY * RotX

        # dst = cv2.warpPerspective(dst, Rot, (width, height))

        # Maffine = np.float32([[1, 0, width/2], [0, 1, height/2]])
        # dst = cv2.warpAffine(dst, Maffine, (width, height))

        # dst = cv2.resize(dst, (width/4, height/4), interpolation = cv2.INTER_CUBIC)
        # cv2.imshow('img',dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        # print float(row['X est'])
        # print float(row['Y est'])
        # print float(row['Z est'])


test_files = glob.glob(EXAMPLE_DIR + '*.jpg')
