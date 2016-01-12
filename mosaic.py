import cv2
import numpy as np
import glob

EXAMPLES_DIR = ('./example/')

test_files = glob.glob(EXAMPLES_DIR + '*.jpg')

for image in test_files:
	print image