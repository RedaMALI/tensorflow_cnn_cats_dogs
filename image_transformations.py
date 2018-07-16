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

def get_gaussian_noise(image, var):
  row,col,ch= image.shape
  mean = 0
  sigma = var**0.5
  gauss = np.random.normal(mean,sigma,(row,col,ch))
  gauss = gauss.reshape(row,col,ch)
  noisy = image + gauss
  return noisy

def get_salt_pepper_noise(image, s_vs_p, amount):
  row,col,ch = image.shape
  out = np.copy(image)
  # Salt mode
  num_salt = np.ceil(amount * image.size * s_vs_p)
  coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
  out[coords] = 1
  # Pepper mode
  num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
  coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
  out[coords] = 0
  return out

def get_poisson_noise(image):
  vals = len(np.unique(image))
  vals = 2 ** np.ceil(np.log2(vals))
  noisy = np.random.poisson(image * vals) / float(vals)
  return noisy

def get_speckle_noise(image):
  row,col,ch = image.shape
  gauss = np.random.randn(row,col,ch)
  gauss = gauss.reshape(row,col,ch)        
  noisy = image + image * gauss
  return noisy

def get_brightness(image, value):
  hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
  h, s, v = cv2.split(hsv)
  v += value
  final_hsv = cv2.merge((h, s, v))
  return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

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
    'gaussian_noise' : get_gaussian_noise,
    'salt_pepper_noise' : get_salt_pepper_noise,
    'poisson_noise' : get_poisson_noise,
    'speckle_noise' : get_speckle_noise,
    'brightness' : get_brightness,
  }
  
  