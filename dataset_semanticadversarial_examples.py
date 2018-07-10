# import scripts
import image_transformations

# import libs
import cv2
import os
import glob
from sklearn.utils import shuffle
import numpy as np

# utils
def get_classes(train_path):
  return [ name for name in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, name)) ];
  

  
# Modèle du dataset
class DataSetWithTransform(object):

  def __init__(self, images_no_transform, images_negative, images_posterized, labels, img_names, cls):
    self._num_examples = images_no_transform.shape[0]

    self._images_no_transform = images_no_transform
    self._images_negative = images_negative
    self._images_posterized = images_posterized
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images_no_transform(self):
    return self._images_no_transform
  
  @property
  def images_negative(self):
    return self._images_negative
    
  @property
  def images_posterized(self):
    return self._images_posterized

  @property
  def labels(self):
    return self._labels

  @property
  def img_names(self):
    return self._img_names

  @property
  def cls(self):
    return self._cls

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def epochs_done(self):
    return self._epochs_done

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""
    start = self._index_in_epoch
    self._index_in_epoch += batch_size

    if self._index_in_epoch > self._num_examples:
      # After each epoch we update this
      self._epochs_done += 1
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch

    return self._images_no_transform[start:end], self._images_negative[start:end], self._images_posterized[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

# Chargement des données en mémoire
def load_train_with_transform(train_path, image_size, classes):
  images_no_transform = []
  images_negative = []
  images_posterized = []
  labels = []
  img_names = []
  cls = []

  print('Going to read training images')
  for fields in classes:   
    index = classes.index(fields)
    print('Now going to read {} files (Index: {})'.format(fields, index))
    path = os.path.join(train_path, fields, '*g')
    files = glob.glob(path)
    for fl in files:
      image = cv2.imread(fl)
      
      image_no_transform = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
      image_negative = get_negative_image(image_no_transform)
      image_posterized = get_posterized_image(image_no_transform)
        
      image_no_transform = get_normalized_image(image_no_transform)
      image_negative = get_normalized_image(image_negative)
      image_posterized = get_normalized_image(image_posterized)
    
      images_no_transform.append(image_no_transform)
      images_negative.append(image_negative)
      images_posterized.append(image_posterized)
        
      label = np.zeros(len(classes))
      label[index] = 1.0
      labels.append(label)
      flbase = os.path.basename(fl)
      img_names.append(flbase)
      cls.append(fields)

  images_no_transform = np.array(images_no_transform)
  images_negative = np.array(images_negative)
  images_posterized = np.array(images_posterized)

  labels = np.array(labels)
  img_names = np.array(img_names)
  cls = np.array(cls)

  # images_* : data [image_size, image_size, color]
  # labels : array with len(classes) elmts > all at zero except to index of the image class equal to one
  # img_names : file name
  # cls : class name
  return images_no_transform, images_negative, images_posterized, labels, img_names, cls

# Préparer un dataset
def read_train_sets_with_transform(train_path, image_size, classes, validation_size):
  class DataSets(object):
    pass

  images_no_transform, images_negative, images_posterized, labels, img_names, cls = load_train_with_transform(train_path, image_size, classes)
  
  images_no_transform, images_negative, images_posterized, labels, img_names, cls = shuffle(images_no_transform, images_negative, images_posterized, labels, img_names, cls) 

  # Same size for all transformations > Use one
  if isinstance(validation_size, float):
    validation_size = int(validation_size * images_no_transform.shape[0])

  validation_images_no_transform = images_no_transform[:validation_size]
  validation_images_negative = images_negative[:validation_size]
  validation_images_posterized = images_posterized[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images_no_transform = images_no_transform[validation_size:]
  train_images_negative = images_negative[validation_size:]
  train_images_posterized = images_posterized[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]
  
  data_set = DataSets()

  data_set.train = DataSetWithTransform(train_images_no_transform, train_images_negative, train_images_posterized, train_labels, train_img_names, train_cls)
  data_set.valid = DataSetWithTransform(validation_images_no_transform, validation_images_negative, validation_images_posterized, validation_labels, validation_img_names, validation_cls)

  return data_set

# END  
