import os
import json

# Gestion des logs
class TrainLogging(object):

  def __init__(self, current_results_path, train_id):
    # set paths
    self._current_results_path = current_results_path
    self._train_id = train_id
    self._settings_file_name = 'settings.csv'
    self._epochs_file_name = 'epochs.csv'
    self._results_file_name = 'results.csv'
    # init setting file
    log_path = self.get_settings_path()
    if not os.path.exists(log_path):
      file_handler = open(log_path, "w")
      header = TrainLogging._get_settings_header()
      file_handler.write(header+"\n")
      file_handler.close()
    # init log file
    log_path = self.get_epochs_path()
    if not os.path.exists(log_path):
      file_handler = open(log_path, "w")
      header = TrainLogging._get_epochs_header()
      file_handler.write(header+"\n")
      file_handler.close()
    # init results file
    log_path = self.get_results_path()
    if not os.path.exists(log_path):
      file_handler = open(log_path, "w")
      header = TrainLogging._get_results_header()
      file_handler.write(header+"\n")
      file_handler.close()
  
  @staticmethod
  def _get_settings_header():
    return "classes_count;classes;validation_size;img_size;num_channels;learning_rate;num_iteration;batch_size;input_layer_dropout;flatten_layer_dropout;conv_layers_count;conv_layers_params;fc_layers_count;fc_layers_params;transformations_count;transformations;training_data;validation_data"
  
  @staticmethod
  def _get_epochs_header():
    return "training_epoch;training_accuracy;validation_accuracy;validation_loss"
    
  @staticmethod
  def _get_results_header():
    return "class_count;data_count;correct_predictions"

  def get_epochs_path(self):
    return os.path.join(self._current_results_path, self._train_id, self._epochs_file_name)

  def get_settings_path(self):
    return os.path.join(self._current_results_path, self._settings_file_name)

  def get_results_path(self):
    return os.path.join(self._current_results_path, self._results_file_name)
  
  def log_settings(self, classes, settings, conv_layers_params, fc_layers_params, transformations, data):
    log_path = self.get_settings_path()
    file_handler = open(log_path, "a")
    line = "{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{};{}".format(
      len(classes),
      json.dumps(classes),
      settings['validation_size'],
      settings['img_size'],
      settings['num_channels'],
      settings['learning_rate'],
      settings['num_iteration'],
      settings['batch_size'],
      settings['input_layer_dropout'],
      settings['flatten_layer_dropout'],
      len(conv_layers_params),
      json.dumps(conv_layers_params),
      len(fc_layers_params),
      json.dumps(fc_layers_params),
      len(transformations),
      json.dumps(transformations),
      len(data.train.labels),
      len(data.valid.labels)
    )
    file_handler.write(line+"\n")
    file_handler.close()
    
  def log_epoch(self, epoch, acc, val_acc, val_loss):
    log_path = self.get_epochs_path()
    file_handler = open(log_path, "a")
    line = "{};{};{};{}".format(
      epoch+1,
      acc,
      val_acc,
      val_loss,
    )
    file_handler.write(line+"\n")
    file_handler.close()

  def log_results(self,classes, data_counts, results):
    log_path = self.get_results_path()
    
    file_handler = open(log_path, "a")
    
    for index, class_name in enumerate(classes) :
      line = "{};{};{}".format(
        class_name,
        data_counts[index],
        results[index]
      )
      file_handler.write(line+"\n")
    
    file_handler.close()