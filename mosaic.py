import cv2
import numpy as np
import glob
import csv
import os
import math
from scipy import ndimage
from scipy import misc
import gdal
import osr
import urllib2
import zipfile

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

# http://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/
# order the envelope points so that they are clockwise starting at upper left
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
    # pts represent the corners of the image that will be tilted about the center
    pts = np.matrix([[0, 0, width, width], [height, 0, 0, height],[0, 0, 0, 0]])

    # the offset is needed to tilt the pts about the center instead of about the origin
    centerOffset = np.matrix([[width/2,width/2,width/2,width/2],[height/2,height/2,height/2,height/2],[0, 0, 0, 0]])
    # shift coordinates so that origin is the center of the image
    pts = pts - centerOffset

    # rotate image about the center and add the offset back into the coordinates
    pts = RotY * RotX * pts + centerOffset

    coords = np.array([(pts[0,0], pts[1, 0]), (pts[0,1], pts[1, 1]), (pts[0,2], pts[1, 2]), (pts[0,3], pts[1, 3])])
    return order_points(coords)

def create_src_rect(height, width):
    coords = np.array([(0, height), (0, 0), (width, 0), (width, height)])
    return order_points(coords)

def four_point_transform(img, RotY, RotX):
    height, width = img.shape[:2]

    # the from pts to be warped
    src_rect = create_src_rect(height, width)
    # the rotated destination pts
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
    # M, mask = cv2.findHomography(src_rect, dst_rect, cv2.RANSAC, 5.0) return same result for M
    M = cv2.getPerspectiveTransform(src_rect, dst_rect)
    warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight))
 
    # return the warped image
    return warped

def calculateDronePositionPixel(img, pitch_rad, roll_rad):
    height, width = img.shape[:2]

    shift_x = FOCAL_LENGTH_MM * math.tan(roll_rad) * (width / CCD_WIDTH_MM)
    shift_y = FOCAL_LENGTH_MM * math.tan(pitch_rad) * (height / CCD_HEIGHT_MM)
    
    center_x = width / 2.0
    center_y = height / 2.0

    return int(center_y + shift_y), int(center_x + shift_x)

def cmPerPixel(img, pitch_rad, roll_rad, elevation_m):
    height, width = img.shape[:2]

    pitch_offset = elevation_m * math.tan(pitch_rad)
    roll_offset = elevation_m * math.tan(roll_rad)

    distance_m = math.sqrt(math.sqrt(pitch_offset**2 + roll_offset**2) + elevation_m**2)
    return (CCD_WIDTH_MM * distance_m * 100)/(FOCAL_LENGTH_MM * width)

#http://stackoverflow.com/questions/25458442/rotate-a-2d-image-around-specified-origin-in-python
def rotateImage(img, angle, pivot):
    padX = [img.shape[1] - pivot[0], pivot[0]]
    padY = [img.shape[0] - pivot[1], pivot[1]]
    print padX
    print padY
    print img.shape[:2]
    #imgP = np.pad(img, [padY, padX], 'constant')
    imgP = np.pad(img, [padY, padX, [0, 0]], 'constant')
    print imgP.shape[:2]
    print pivot
    imgR = ndimage.rotate(imgP, angle, reshape=False)
    return imgR

def coordsFromAziDistance(lat1_deg, lon1_deg, azimuth_deg, distance_cm):
    radius_km = 6378.1 #Radius of the Earth

    azimuth_rad = math.radians(azimuth_deg)
    distance_km = distance_cm / 100000
    lat1_rad = math.radians(lat1_deg) #Current lat point converted to radians
    lon1_rad = math.radians(lon1_deg) #Current long point converted to radians

    lat2_rad = math.asin( math.sin(lat1_rad)*math.cos(distance_km/radius_km) + math.cos(lat1_rad)*math.sin(distance_km/radius_km)*math.cos(azimuth_rad))
    lon2_rad = lon1_rad + math.atan2(math.sin(azimuth_rad)*math.sin(distance_km/radius_km)*math.cos(lat1_rad), math.cos(distance_km/radius_km)-math.sin(lat1_rad)*math.sin(lat2_rad))

    lat2_deg = math.degrees(lat2_rad)
    lon2_deg = math.degrees(lon2_rad)

    return lon2_deg, lat2_deg 


def envelopeFromImage(img, lon1_deg, lat1_deg, x_pixel, y_pixel, GSD):
    height, width = img.shape[:2]
    
    #               up               right                       down                        left
    azimuths_deg = [0.0,            90.0,                       180.0,                      270.0]
    distances_cm = [y_pixel * GSD,  (width - x_pixel) * GSD,    (height - y_pixel) * GSD,   x_pixel * GSD]

    lon_null, lat_up = coordsFromAziDistance(lat1_deg, lon1_deg, azimuths_deg[0], distances_cm[0])
    lon_right, lat_null = coordsFromAziDistance(lat1_deg, lon1_deg, azimuths_deg[1], distances_cm[1])
    lon_null, lat_down = coordsFromAziDistance(lat1_deg, lon1_deg, azimuths_deg[2], distances_cm[2])
    lon_left, lat_null = coordsFromAziDistance(lat1_deg, lon1_deg, azimuths_deg[3], distances_cm[3])

    #ulx uly lrx lry
    return lon_left, lat_up, lon_right, lat_down

def retrieveExampleData():
    print 'retriving data'
    response = urllib2.urlopen('https://s3.amazonaws.com/drone.deploy.map.engine/example.zip')
    zipcontent= response.read()
    with open("example.zip", 'w') as f:
        f.write(zipcontent)
    with zipfile.ZipFile("./example.zip") as zf:
        zf.extractall("./example")

if (os.path.exists(EXAMPLE_DIR) == False):
    retrieveExampleData()
    
with open(IMAGE_DETAILS_FILE, 'r') as image_details_csv:
    details_reader = csv.DictReader(image_details_csv)

    for row in details_reader:
        # important columns X est,Y est,Z est,Yaw est,Pitch est,Roll est
        jpg_filename = os.path.join(EXAMPLE_DIR, row['Filename'])

        # open file for opencv
        img = cv2.imread(jpg_filename)

        # get yaw, pitch, roll and elevation
        yaw_deg = float(row['Yaw est']) 
        pitch_rad = math.radians(float(row['Pitch est']))
        roll_rad = math.radians(float(row['Roll est']))
        elevation_meters = float(row['Z est'])

        # pitch rotation (axis through wing perpendicular to flight path)
        RotY = np.matrix([[math.cos(pitch_rad) , 0, math.sin(pitch_rad)],
                          [0                       , 1, 1                      ],
                          [-math.sin(pitch_rad), 0, math.cos(pitch_rad)]])

        # roll rotation (axis through direction of flight path)
        RotX = np.matrix([[1, 0                     , 0                      ],
                          [0, math.cos(roll_rad), -math.sin(roll_rad)],
                          [0, math.sin(roll_rad), math.cos(roll_rad) ]])

        # use rotation matrices to adjust image for pitch and roll errors. Rotate about the center of image.
        warped_image = four_point_transform(img, RotY, RotX)

        # get the pixel directly below the drone (not the center of the image)
        drone_pixel_x, drone_pixel_y = calculateDronePositionPixel(warped_image, pitch_rad, roll_rad)
        print drone_pixel_x
        print drone_pixel_y
        drone_pixel_x, drone_pixel_y = calculateDronePositionPixel(img, pitch_rad, roll_rad)
        print drone_pixel_x
        print drone_pixel_y


        # Ground sampling distance calculations using
        GSD = cmPerPixel(warped_image, pitch_rad, roll_rad, elevation_meters)

        rotate_image = rotateImage(warped_image, -yaw_deg, (drone_pixel_x, drone_pixel_y))
        #rotate_image= ndimage.rotate(warped_image, -yaw_deg, (1, 0))
        temp_filename = jpg_filename + '.jpg'

        cv2.imwrite(temp_filename, rotate_image)

        src_ds = gdal.Open(temp_filename)

        lon_deg = float(row['X est'])
        lat_deg = float(row['Y est'])
        height, width = rotate_image.shape[:2]
        ulx, uly, lrx, lry = envelopeFromImage(rotate_image, lon_deg, lat_deg, width / 2, height / 2, GSD)


        env = "-a_ullr " + str(ulx) + " " + str(uly) + " " + str(lrx) + " " + str(lry) + " "
        gdal_tif_filename = jpg_filename + 'gdal.tif'
        os.system("gdal_translate -co compress=LZW -of GTiff -a_srs EPSG:4326 -a_nodata 0 " + env + temp_filename + " " +  gdal_tif_filename)

        # cleanup temp file
        os.remove(temp_filename)
        

os.remove('out.tif')
gdal_merge = "gdal_merge.py -v -n 0 -o out.tif "
tif_files = glob.glob(EXAMPLE_DIR + '/*.tif')
for tif in tif_files:
    gdal_merge = gdal_merge + " " + tif

os.system(gdal_merge)

# for tif in tif_files:
#     os.remove(tif)

