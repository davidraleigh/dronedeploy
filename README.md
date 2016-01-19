# dronedeploy

## required libraries:
```bash
$ pip install geopy
$ brew install gdal
$ brew install opencv3
$ brew install scipy
```

## running the script:
To run the script you must have installed the above libraries. This script has only been tested with python 2.7. The script will download the examples data. If there is no internet connection, you must put the 'examples' directory at the same level as the 'mosaic.py' script.
```bash
$ python mosaic.py
```


### Notes on yaw, pitch and roll
![Drone coordinate system](https://www.av8n.com/physics/img48/yaw-pitch-roll.png)

Summary of coordinate system for drone position
- yaw will be positive in the NE direction and negative in the NW direction
- pitch will be positive in the upward direction
- roll will be positive as the left wing lifts and right wing dips

### Notes on image coordinate system in relation to yaw, pitch and roll
![OpenCV image coordinate system](https://tspp.files.wordpress.com/2009/10/cvcoordinate.png?w=1000)

In the image, if the camera is oriented with the top of the photo in the direction of the flight path:
- positive pitch means that the position of the drone will be lower in the image
  - this means a *higher* y pixel value for the drone's position relative to center of image
  - center_y = height / 2.0
  - shift_y = FOCAL_LENGTH_MM * math.tan(roll_radians) * (height / CCD_HEIGHT_MM)
  - drone_y = center_y + shift_y
- negative pitch means that the position of the drone will be higher in the image
  - this means a *lower* y pixel value for the drone's position relative to center of image

- positive roll means that the position of the drone will be to the right in the image
  - this means a *higher* x pixel value for the drone's position relative to center of image
  - center_x = width / 2.0
  - shift_x = FOCAL_LENGTH_MM * math.tan(roll_radians) * (width / CCD_WIDTH_MM)
  - center_x + shift_x
- negative roll means that the position of the drone will be to the left in the image
  - this means a *lower* x pixel value for the drone's position relative to center of image
