from styx_msgs.msg import TrafficLight
import cv2
from datetime import datetime
import csv
import os
import numpy as np
import tensorflow as tf
import keras
from keras.backend import resize_images
from keras.preprocessing.image import img_to_array
from scipy.misc import imresize
import rospy
import matplotlib.pyplot as plt

class TLClassifier(object):
    def __init__(self):
	#self.model = keras.models.load_model("/home/student/Desktop/CarND-Capstone-master/ros/src/tl_detector/light_classification/clf_best")
	self.model = keras.models.load_model("/home/student/Desktop/CarND-Capstone-master/ros/src/tl_detector/light_classification/clf.h5")
	self.model._make_predict_function()
	#self.counter = 0

    def get_classification(self, image, label=TrafficLight.UNKNOWN):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light
	    the label can be given as optional argument - 
	    if so the image and its label are saved as training images

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        
	#saves training images
	if label != TrafficLight.UNKNOWN:
		now = datetime.now()		
		name = now.strftime("%Y_%m_%d_%H_%M_%S_%f")
		path = "/home/student/Desktop/CarND-Capstone-master/imgs/traffic_lights/"
		cv2.imwrite(path+"imgs/"+name+".jpg", image)
		fieldnames = ["image_name", "label"]
		if os.path.isfile(path+"labels.csv"):
					
			with open(path+"labels.csv", "a") as f:
				writer=csv.DictWriter(f, fieldnames=fieldnames)				
				writer.writerow({"image_name":name,\
				"label":str(label)})
				
		else:
			with open(path+"labels.csv", "w") as f:
				writer=csv.DictWriter(f, fieldnames=fieldnames)
				writer.writeheader()				
				writer.writerow({"image_name":name,\
				"label":str(label)})
			
			
		
		return(label)
	else:
		img = imresize(image, (60, 80))
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)	
		img = np.array(img / 255.0)
		#if self.counter<10:
		#	plt.imshow(img)
		#	plt.savefig("/home/student/Desktop/CarND-Capstone-master/imgs/traffic_lights/"+str(self.counter)+"_run.jpg")
		#	self.counter+=1


		img = img.reshape(-1, 60, 80, 3)		

		y = self.model.predict(img)
		#rospy.logwarn("tl state y: "+str(y))
		state = y.argmax(axis=1)[0]
		#rospy.logwarn("tl state before: "+str(state))
		if state == 3:
			state = 4
		#rospy.logwarn("tl state: "+str(state))
		#rospy.logwarn("")



		return(state)
		#return(TrafficLight.UNKNOWN)










