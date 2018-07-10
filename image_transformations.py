import cv2
import numpy as np

# Private functions
def _common_scale(image, scale, x, y):
  if scale <= 1:
    return image
  height, width, channels = image.shape
  resized_image = cv2.resize(image,None,fx=scale, fy=scale, interpolation = cv2.INTER_CUBIC)
  croped_image = resized_image[ y:height+y, x:width+x ]
  return croped_image

  
# Available transformations
def get_negative(image) :
  return (255-image)

def get_posterize(image) :
  Z = image.reshape((-1,3))
  # convert to np.float32
  Z = np.float32(Z)

  # define criteria, number of clusters(K) and apply kmeans()
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
  K = 4
  ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

  # Now convert back into uint8, and make original image
  center = np.uint8(center)
  res = center[label.flatten()]
  posterized_image = res.reshape((image.shape))
  return posterized_image

def get_normalize(image):
  image = image.astype(np.float32)
  image = np.multiply(image, 1.0 / 255.0) # C'est un choix, ce qui compte c'est d'avoir toujours la même chose : soit [0-255] ou [0-1]
  return image

def get_centre_scale(image, scale):
  height, width, channels = image.shape
  x = int( ((scale - 1) / 2) * width )
  y = int( ((scale - 1) / 2) * height )
  return _common_scale(image, scale, x, y)
  
def get_top_left_scale(image, scale):
  return _common_scale(image, scale, 0, 0)
  
def get_bottom_left_scale(image, scale):
  height, width, channels = image.shape
  y = int( (scale - 1) * height ) - 1
  return _common_scale(image, scale, 0, y)

def get_top_right_scale(image, scale):
  height, width, channels = image.shape
  x = int( (scale - 1) * width ) - 1
  return _common_scale(image, scale, x, 0)
  
def get_bottom_right_scale(image, scale):
  height, width, channels = image.shape
  x = int( (scale - 1) * width ) - 1
  y = int( (scale - 1) * height ) - 1
  return _common_scale(image, scale, x, y)
  
def get_rotate(image,angle):
  angle = 10 if angle >= 0 else -10
  scale = 1.2
  height, width, channels = image.shape
  x = int( ((scale - 1) / 2) * width )
  y = int( ((scale - 1) / 2) * height )
  rotation_matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
  img_rotation = cv2.warpAffine(image, rotation_matrix, (width, height))
  return _common_scale(img_rotation, scale, x, y)
  
def get_flip(image, horizontal = 1, vertical = 0):
  if horizontal == 0 and vertical == 0:
    return image
  if horizontal == 1 and vertical == 1:
    return cv2.flip( image, -1 )
  if horizontal == 1:
    return cv2.flip( image, 0 )
  if vertical == 1:
    return cv2.flip( image, 1 )
  
# tools
def get_supported_transformations():
  return {
    'negative' : get_negative,
    'posterize' : get_posterize,
    'normalize' : get_normalize,
    'centre_scale' : get_centre_scale,
    'top_left_scale' : get_top_left_scale,
    'bottom_left_scale' : get_bottom_left_scale,
    'top_right_scale' : get_top_right_scale,
    'bottom_right_scale' : get_bottom_right_scale,
    'rotate' : get_rotate,
    'flip' : get_flip,
  }
  
  