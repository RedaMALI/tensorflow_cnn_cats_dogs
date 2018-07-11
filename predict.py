from train_logging import TrainLogging

import tensorflow as tf
import numpy as np
import os,glob,cv2
import sys,argparse

import os

# Set ID
train_id = '20180710084514'

# Settings
categories = ['cats','dogs']
image_size = 128
num_channels = 3
testing_data_path = './testing_data/'

# Prepare
results_path = './dogs-cats-results/'
checkpoint_path = results_path+train_id+'/'
model_path = checkpoint_path + 'model.ckpt.meta'
categories_size = len(categories)

# Logging
logging = TrainLogging(results_path, train_id)

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph(model_path)
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint(checkpoint_path))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

global_results = []
global_data_count = []

# testing_data_path contains a folder for each category
for index, category in enumerate(categories):
  
  category_testing_data_path = os.path.join(testing_data_path, category)
  if not os.path.isdir(category_testing_data_path):
    continue
    
  test_images = [f for f in os.listdir(category_testing_data_path) if os.path.isfile(os.path.join(category_testing_data_path, f))]
  print( "Testing size in {} : {}".format(category, len(test_images)) )

  categories_rs = []
  for x in range(0, categories_size):
    categories_rs.append(0)

  for filename in test_images:
    filename = os.path.join(category_testing_data_path, filename)
    
    images = []
    # Reading the image using OpenCV
    image = cv2.imread(filename)
    # Resizing the image to our desired size and preprocessing will be done exactly as done during training
    image = cv2.resize(image, (image_size, image_size), cv2.INTER_LINEAR)
    images.append(image)
    images = np.array(images, dtype=np.uint8)
    images = images.astype('float32')
    images = np.multiply(images, 1.0/255.0) 
    
    #The input to the network is of shape [None image_size image_size num_channels]. Hence we reshape.
    x_batch = images.reshape(1, image_size,image_size,num_channels)

    # Now, let's get hold of the op that we can be processed to get the output.
    # In the original network y_pred is the tensor that is the prediction of the network
    y_pred = graph.get_tensor_by_name("y_pred:0")

    ## Let's feed the images to the input placeholders
    x= graph.get_tensor_by_name("x:0") 
    y_true = graph.get_tensor_by_name("y_true:0")
    is_training = graph.get_tensor_by_name("is_training:0")
    
    y_test_images = np.zeros((1, categories_size)) 

    ### Creating the feed_dict that is required to be fed to calculate y_pred 
    feed_dict_testing = {x: x_batch, y_true: y_test_images, is_training: False}
    result=sess.run(y_pred, feed_dict=feed_dict_testing)
    # result is of this format [probabiliy_of_rose probability_of_sunflower]

    # reduce
    prediction=tf.argmax(result,1)
    rs = sess.run(prediction)

    categories_rs[int(rs)] += 1
  
  print("Result : {} / {} ({})".format( categories_rs[index], len(test_images),  category))
  global_results.append(categories_rs[index])
  global_data_count.append(len(test_images))

logging.log_results(categories, global_data_count, global_results)