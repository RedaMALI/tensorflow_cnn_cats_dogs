import os
import shutil

images_file = 'D:\\Users\\Reda\\Doctorat\\ENSA-El-Jadida\\Test-2017-08\\bcnn\\data\\cub\\images.txt'
classes_file = 'D:\\Users\\Reda\\Doctorat\\ENSA-El-Jadida\\Test-2017-08\\bcnn\\data\\cub\\image_class_labels.txt'
labels_name_file = 'D:\\Users\\Reda\\Doctorat\\ENSA-El-Jadida\\Test-2017-08\\bcnn\\data\\cub\\classes.txt'
images_path = 'D:\\Users\\Reda\\Doctorat\\ENSA-El-Jadida\\Test-2017-08\\bcnn\\data\\cub\\images'
test_split_file = 'D:\\Users\\Reda\\Doctorat\\ENSA-El-Jadida\\Test-2017-08\\bcnn\\data\\cub\\train_test_split.txt'

current_folder = os.path.dirname( os.path.realpath(__file__) )
training_folder = os.path.join(current_folder, 'cub_training_data')
testing_folder = os.path.join(current_folder, 'cub_testing_data')
print("training_folder : " + training_folder)
print("testing_folder : " + testing_folder)

data_split = []
data_classes = []
data_labels = []

with open(test_split_file) as split_p:
	for line in split_p:
		tokens = line.split()
		data_split.append(int(tokens[1]) == 1)
		
with open(classes_file) as classes_p:
	for line in classes_p:
		tokens = line.split()
		data_classes.append( int(tokens[1]) )
		
with open(labels_name_file) as labels_p:
	for line in labels_p:
		tokens = line.split()
		data_labels.append( tokens[1] )

print("data_split size : " + str( len(data_split) ))

# Clear training_folder
if os.path.exists(training_folder):
	shutil.rmtree(training_folder, ignore_errors=True)
# Clear testing_folder
if os.path.exists(testing_folder):
	shutil.rmtree(testing_folder, ignore_errors=True)

# Create training_folder
if not os.path.exists(training_folder):
    os.makedirs(training_folder)
	
# Create testing_folder
if not os.path.exists(testing_folder):
    os.makedirs(testing_folder)
		
with open(images_file) as data_p:
	for num, line in enumerate(data_p, 1):
		tokens = line.split()
		if data_split[num-1]:
			dest_path = os.path.join(testing_folder, data_labels[ data_classes[num-1]-1 ])
			if not os.path.exists(dest_path):
				os.makedirs(dest_path)
			shutil.copy2(os.path.join(images_path, tokens[1]), dest_path)
		else:
			dest_path = os.path.join(training_folder, data_labels[ data_classes[num-1]-1 ])
			if not os.path.exists(dest_path):
				os.makedirs(dest_path)
			shutil.copy2(os.path.join(images_path, tokens[1]), dest_path)