import csv
import numpy as np
import tensorflow as tf
import cv2
import os 

#import keras 
#print(keras.__version__)
#print(tf.__version__)

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D
from keras.utils import to_categorical
from keras.layers.pooling import MaxPooling2D
from keras.preprocessing.image import load_img, img_to_array
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD
from PIL import Image
import skimage.transform
#print(skimage.__version__)
#from copy import deepcopy
from sklearn.utils import shuffle
from sklearn import metrics
import matplotlib.pyplot as plt


fieldnames = ["image_name", "label"]
path = "/home/student/Desktop/CarND-Capstone-master/imgs/traffic_lights/"

image_names = []
labels = []

with open(path+"labels_final.csv") as f:
	reader=csv.reader(f)
	next(reader)
	for row in reader:
		image_names.append(row[0])
		lab = int(row[1])
		if lab>2:
			labels.append(3)		
		else:
			labels.append(lab)
print("imagenames read")

#this loop deletes old training images
for(dirpath, dirnamens, filenames) in os.walk(path+"imgs/"):
	for f in filenames:
		if(f[:-4] not in image_names):
			print(f[:-4])
			os.remove(path+"imgs/"+f)


image_names, labels = shuffle(image_names, labels)

print(set(labels))

#labels = 5*labels
#labels = 3*labels
labelsonehot = to_categorical(labels)
#labelsonehot = to_categorical(labels[0:100])

images = []
for img in image_names:
#for img in image_names[0:100]:
	#print(path+"imgs/"+img+".jpg")	
	#i = cv2.imread(path+"imgs/"+img+".jpg", cv2.IMREAD_COLOR)
	#print(i.shape)	
	#i = np.array(i / 255.0 - 0.5)
	#images.append(i)
	p = path+"imgs/"+img+".jpg"
	#i = load_img(p, grayscale=False, color_mode="rgb", target_size=(60,80), interpolation="nearest")
	i = load_img(p, grayscale=False, target_size=(60,80))
	i = img_to_array(i)
	#print(np.shape(i))
	i = np.array(i / 255.0)
	#print(np.shape(i))
	images.append(i)
	#print(np.max(i))

	#rot1 = skimage.transform.rotate(i, angle=-10, resize=False)
	#images.append(rot1)

	#rot2 = skimage.transform.rotate(i, angle=10, resize=False)
	#images.append(rot2)
	
	#rot3 = skimage.transform.rotate(i, angle=-5, resize=False)
	#images.append(rot3)

	#rot4 = skimage.transform.rotate(i, angle=5, resize=False)
	#images.append(rot4)

	#print(np.shape(rot1))
	#print()

#cv2.imwrite("/home/student/Desktop/CarND-Capstone-master/imgs/traffic_lights/first.jpg", images[0])
for k in range(10):
	plt.imshow(images[k])
	plt.savefig("/home/student/Desktop/CarND-Capstone-master/imgs/traffic_lights/"+str(k)+"_train.jpg")

print("images read")

#X_train = np.array([img for img in images])
X_train = np.array(images)
y_train = np.array(labelsonehot)
#print(y_train)

print(np.shape(X_train))
print(np.shape(y_train))

model = Sequential()

#model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(600, 800, 3)))
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=(60, 80, 3)))
model.add(MaxPooling2D(pool_size=(2,2)))
#model.add(Dropout(0.5))
model.add(Dropout(0.75))
model.add(Activation('relu'))
model.add(Flatten())

model.add(Dense(128))
model.add(Activation('relu'))

model.add(Dense(64))
model.add(Activation('relu'))

model.add(Dense(32))
model.add(Activation('relu'))

model.add(Dense(16))
model.add(Activation('relu'))

model.add(Dense(4))
model.add(Activation('softmax'))
print("model constructed")

#cw = 10.
#class_weight = {0:cw, 1:cw, 2:cw, 3:0.5}

cw = 10.
class_weight = {0:cw, 1:2., 2:8., 3:0.5}

#cw = 8.
#class_weight = {0:cw, 1:cw, 2:cw, 3:1.}
print("cw: "+str(cw))

cb = EarlyStopping(monitor='val_acc', patience=3, restore_best_weights=True, verbose=1)


#opt = SGD(lr=0.01)
#opt = SGD(lr=0.03)
model.compile('adam', 'categorical_crossentropy', ['accuracy'])
#model.compile(opt, 'categorical_crossentropy', ['accuracy'])

history = model.fit(X_train, y_train, epochs=20, validation_split=0.125, verbose=2, class_weight=class_weight, callbacks=[cb])
#history = model.fit(X_train, y_train, epochs=10, validation_split=0.15, verbose=2, class_weight=class_weight)
print("model fitted")

model.save("/home/student/Desktop/CarND-Capstone-master/ros/src/tl_detector/light_classification/clf.h5")
print("model saved")

acc = model.evaluate(X_train, y_train, verbose=1)
print("model evaluated")
print(acc)

y_pred = model.predict(X_train)
matrix = metrics.confusion_matrix(y_train.argmax(axis=1), y_pred.argmax(axis=1))
print("confusion matrix generated")

print(matrix)



