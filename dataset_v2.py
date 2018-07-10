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

def _prepare_image(image_to_prepare, file_path, class_index, class_name, images, classes, labels, img_names, cls):
  image = image_to_prepare.astype(np.float32)
  image = np.multiply(image, 1.0 / 255.0) # C'est un choix, ce qui compte c'est d'avoir toujours la même chose : soit [0-255] ou [0-1]
  images.append(image)
  label = np.zeros(len(classes))
  label[class_index] = 1.0
  labels.append(label)
  flbase = os.path.basename(file_path)
  img_names.append(flbase)
  cls.append(class_name)
  return image

  
# Modèle du dataset
class DataSet(object):

  def __init__(self, images, labels, img_names, cls):
    self._num_examples = images.shape[0]

    self._images = images
    self._labels = labels
    self._img_names = img_names
    self._cls = cls
    self._epochs_done = 0
    self._index_in_epoch = 0

  @property
  def images(self):
    return self._images

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

    return self._images[start:end], self._labels[start:end], self._img_names[start:end], self._cls[start:end]

# Chargement des données en mémoire
def load_train(train_path, image_size, classes, transform_list = False):
  images = []
  labels = []
  img_names = []
  cls = []
  
  transformations = image_transformations.get_supported_transformations()

  print('Going to read training images')
  for fields in classes:   
    index = classes.index(fields)
    print('Now going to read {} files (Index: {})'.format(fields, index))
    path = os.path.join(train_path, fields, '*g')
    files = glob.glob(path)
    for fl in files:
      image = cv2.imread(fl)
      image = cv2.resize(image, (image_size, image_size), interpolation = cv2.INTER_LINEAR)
      
      # Préparer l'image d'origine
      _prepare_image(image, fl, index, fields, images, classes, labels, img_names, cls)
      
      # Vérifier s'il faut transformer les images
      if not transform_list: continue
      
      # Traiter les transformation demandée
      for transform in transform_list:
        type = transform['type']
        transform_args = transform['args']
        if type not in transformations.keys():
          continue
        transform_args['image'] = image
        transformed_image = transformations[type](**transform_args)
        del transform_args['image']
        _prepare_image(transformed_image, fl+type, index, fields, images, classes, labels, img_names, cls)
        
  images = np.array(images)
  labels = np.array(labels)
  img_names = np.array(img_names)
  cls = np.array(cls)

  # images : data [image_size, image_size, color]
  # labels : array with len(classes) elmts > all at zero except to index of the image class equal to one
  # img_names : file name
  # cls : class name
  return images, labels, img_names, cls

# Préparer un dataset
def read_train_sets(train_path, image_size, classes, validation_size, transform_list = False):
  class DataSets(object):
    pass
  data_sets = DataSets()

  images, labels, img_names, cls = load_train(train_path, image_size, classes, transform_list)
  images, labels, img_names, cls = shuffle(images, labels, img_names, cls)  

  if isinstance(validation_size, float):
    validation_size = int(validation_size * images.shape[0])

  validation_images = images[:validation_size]
  validation_labels = labels[:validation_size]
  validation_img_names = img_names[:validation_size]
  validation_cls = cls[:validation_size]

  train_images = images[validation_size:]
  train_labels = labels[validation_size:]
  train_img_names = img_names[validation_size:]
  train_cls = cls[validation_size:]

  data_sets.train = DataSet(train_images, train_labels, train_img_names, train_cls)
  data_sets.valid = DataSet(validation_images, validation_labels, validation_img_names, validation_cls)

  return data_sets

# END
