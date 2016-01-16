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

# http://forum.dji.com/thread-28597-1-1.html
CCD_WIDTH_MM = 6.16
CCD_HEIGHT_MM = 4.62

FOCAL_LENGTH_35_MM = 20.0
# https://support.pix4d.com/hc/en-us/articles/202557469-Step-1-Before-Starting-a-Project-1-Designing-the-Images-Acquisition-Plan-b-Computing-the-Flight-Height-for-a-Given-GSD
FOCAL_LENGTH_MM = (FOCAL_LENGTH_35_MM * CCD_WIDTH_MM) / 34.6

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
    pts = np.matrix([[0, 0, width, width], [height, 0, 0, height],[0, 0, 0, 0]])
    print pts

    centerOffset = np.matrix([[width/2,width/2,width/2,width/2],[height/2,height/2,height/2,height/2],[0, 0, 0, 0]])
    # shift coordinates so that origin is the center of the image
    pts = pts - centerOffset
    print pts
    # rotate image about the center and add the offset back into the coordinates
    pts = RotY * RotX * pts + centerOffset
    print pts
    coords = np.array([(pts[0,0], pts[1, 0]), (pts[0,1], pts[1, 1]), (pts[0,2], pts[1, 2]), (pts[0,3], pts[1, 3])])
    return order_points(coords)

def create_src_rect(height, width):
    #pts = np.matrix([[0, 0, width, width], [height, 0, 0, height], [1, 1, 1, 1]])
    #coords = np.array([(pts[0,0], pts[1, 0]), (pts[0,1], pts[1, 1]), (pts[0,2], pts[1, 2]), (pts[0,3], pts[1, 3])])
    coords = np.array([(0, height), (0, 0), (width, 0), (width, height)])
    return order_points(coords)

def four_point_transform(img, RotY, RotX):
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
    # M, mask = cv2.findHomography(src_rect, dst_rect, cv2.RANSAC, 5.0) return same result for M
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    print M
    
    print M
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def calculateDronePositionPixel(img, pitch_radians, roll_radians):
    height, width = img.shape[:2]

    shift_x = FOCAL_LENGTH_MM * math.tan(roll_radians) * (width / CCD_WIDTH_MM)
    shift_y = FOCAL_LENGTH_MM * math.tan(pitch_radians) * (height / CCD_HEIGHT_MM)
    
    center_x = width / 2.0
    center_y = height / 2.0

    return int(center_y + shift_y), int(center_x + shift_x)

def cmPerPixel(img, pitch_radians, roll_radians, elevation):
    height, width = img.shape[:2]

    pitch_offset = elevation * math.tan(pitch_radians)
    roll_offset = elevation * math.tan(roll_radians)

    distance = math.sqrt(math.sqrt(pitch_offset**2 + roll_offset**2) + elevation**2)
    print distance
    return (CCD_WIDTH_MM * distance * 100)/(FOCAL_LENGTH_MM * width)

#http://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    print padX
    padY = [img.shape[0] - pivot[1], pivot[1]]
    print padY
    #imgP = np.pad(img, [padY, padX], 'constant')
    imgP = np.pad(img, [padY, padX, [0, 0]], 'constant')
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR
    # return imgR[padY[0] : -padY[1], padX[0] : -padX[1]]

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
        elevation_meters = float(row['Z est'])

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
        drone_pixel_x, drone_pixel_y = calculateDronePositionPixel(warped_image, pitch_radians, roll_radians)
        height, width = warped_image.shape[:2]
        print "warped dimensions"
        print height
        print width
        GSD = cmPerPixel(warped_image, pitch_radians, roll_radians, elevation_meters)

        #rotate_image = rotateImage(warped_image, yaw_degrees, (drone_pixel_x, drone_pixel_y))
        rotate_image= ndimage.rotate(warped_image, yaw_degrees, (1, 0))
        cv2.imwrite(os.path.join(EXAMPLE_DIR, row['Filename']) + '.tif', rotate_image)
        
        height, width = rotate_image.shape[:2]
        print height
        print width

        # dst = cv2.resize(rotate_image, (width/4, height/4), interpolation = cv2.INTER_CUBIC)

        # cv2.imshow('img', dst)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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
