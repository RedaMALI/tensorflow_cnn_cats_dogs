# import scripts
import dataset_v2 as dataset
from cnn_layers import create_weights
from cnn_layers import create_biases
from cnn_layers import create_convolutional_layer
from cnn_layers import create_flatten_layer
from cnn_layers import create_fc_layer
from train_logging import TrainLogging

# import libs
import tensorflow as tf
import time
import datetime
import math
import os
import random
import numpy as np
import json
import sys

# -----------------------------------------------------------------------------------------------------------
# RMA
# Network Architecture (Code : AR001)
# Image --> CNN --> Flatten Layer --> FC --> end
# -----------------------------------------------------------------------------------------------------------

#Adding Seed so that random initialization is consistent >> Get the same random values in each execution
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

# Folders settings
results_path = './dogs-cats-results/'
train_path = 'training_data'

# Prepare input data
classes = ['cats','dogs']
num_classes = len(classes)

# Read params.json
params_file = open('./params.json')
json_str = params_file.read()
params_list = json.loads(json_str)

for params in params_list['params'] :
  settings = params['settings']
  conv_layers_params = params['conv_layers_params']
  fc_layers_params = params['fc_layers_params']
  transformations = params['transformations']

  # Get current timestamp, will be used as id
  now = datetime.datetime.now()
  train_id = now.strftime("%Y%m%d%H%M%S")
  
  # Set folders
  current_run_path = os.path.join(results_path, train_id)
  model_path = os.path.join(current_run_path, 'model.ckpt')

  # Prepare folders and init logging
  if not os.path.exists(current_run_path):
    os.makedirs(current_run_path)
  logging = TrainLogging(results_path, train_id)

  # We shall load all the training and validation images and labels into memory using openCV and use that during training
  data = dataset.read_train_sets(train_path, settings['img_size'], classes, validation_size=settings['validation_size'], transform_list=transformations)

  print("Complete reading input data. Will Now print a snippet of it")
  print("Number of files in Training-set:\t\t{}".format(len(data.train.labels)))
  print("Number of files in Validation-set:\t{}".format(len(data.valid.labels)))

  # Log settings before start session
  logging.log_settings(classes, settings, conv_layers_params, fc_layers_params, transformations, data)

  session = tf.Session()

  is_training = tf.placeholder(tf.bool, shape=[], name='is_training')

  # Create a var that can be feeded with data : [ANY,img_size,img_size,num_channels]
  x = tf.placeholder(tf.float32, shape=[None, settings['img_size'],settings['img_size'],settings['num_channels']], name='x')

  ## labels
  y_true = tf.placeholder(tf.float32, shape=[None, num_classes], name='y_true')
  y_true_cls = tf.argmax(y_true, dimension=1)

  # Start with Conv Layers
  current_layer = x
  if settings['input_layer_dropout'] > 0:
    current_layer = tf.layers.dropout(inputs=current_layer, rate=settings['input_layer_dropout'], training=is_training)

  for conv_layer in conv_layers_params:
    current_layer = create_convolutional_layer(input=current_layer,
                num_input_channels=conv_layer['num_input_channels'],
                conv_filter_size=conv_layer['filter_size'],
                num_filters=conv_layer['num_filters'])
    if conv_layer['dropout'] > 0:
      current_layer = tf.layers.dropout(inputs=current_layer, rate=conv_layer['dropout'], training=is_training)

  # Flatten the CNNs output
  current_layer = create_flatten_layer(current_layer)
  if settings['flatten_layer_dropout'] > 0:
    current_layer = tf.layers.dropout(inputs=current_layer, rate=settings['flatten_layer_dropout'], training=is_training)

  # Next : Fully Connected Layers
  for fc_layer in fc_layers_params:
    current_layer = create_fc_layer(input=current_layer,
                      num_inputs=current_layer.get_shape()[1:4].num_elements(),
                      num_outputs=fc_layer['num_outputs'],
                      use_relu=fc_layer['use_relu'])
    if fc_layer['dropout'] > 0:
      current_layer = tf.layers.dropout(inputs=current_layer, rate=fc_layer['dropout'], training=is_training)

  # Compute
  y_pred = tf.nn.softmax(current_layer,name='y_pred')

  y_pred_cls = tf.argmax(y_pred, dimension=1)
  session.run(tf.global_variables_initializer())
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=current_layer,
                                                      labels=y_true)
  cost = tf.reduce_mean(cross_entropy)
  optimizer = tf.train.AdamOptimizer(learning_rate=settings['learning_rate']).minimize(cost)
  correct_prediction = tf.equal(y_pred_cls, y_true_cls)
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


  session.run(tf.global_variables_initializer())
  saver = tf.train.Saver()
  total_iterations = 0

  def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0:04d} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))
    logging.log_epoch(epoch, acc, val_acc, val_loss)

  def train(num_iteration):
    global total_iterations
    
    batch_size = settings['batch_size']
    
    for i in range(total_iterations,
                total_iterations + num_iteration):

      x_batch, y_true_batch, _, _ = data.train.next_batch(batch_size)
      x_valid_batch, y_valid_batch, _, _ = data.valid.next_batch(batch_size)

      
      feed_dict_tr = {x: x_batch,
                        y_true: y_true_batch, is_training: True}
      feed_dict_val = {x: x_valid_batch,
                            y_true: y_valid_batch, is_training: False}

      session.run(optimizer, feed_dict=feed_dict_tr)

      if i % int(data.train.num_examples/batch_size) == 0: 
        val_loss = session.run(cost, feed_dict=feed_dict_val)
        epoch = int(i / int(data.train.num_examples/batch_size))    
        
        show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
        saver.save(session, model_path) 
    total_iterations += num_iteration

  train(num_iteration=settings['num_iteration'])
