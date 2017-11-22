from styx_msgs.msg import TrafficLight
import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import threading
import time

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from threading import Lock
from datetime import datetime


class TLClassifier(object):
    def __init__(self):

        # it takes time tu run the clasifyer so it is best to just return the
        # last known classification. Homework for later: bilt a faster model
        self.lock = Lock()

        #initializing tensorflow
        dir_path = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_MODEL = dir_path + '/06train-faster_rcnn_inception_v2_coco_2017_11_08.pb'
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.FastGFile(PATH_TO_MODEL, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
            self.d_boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
            self.d_scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.d_classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_d = self.detection_graph.get_tensor_by_name('num_detections:0')
        self.sess = tf.Session(graph=self.detection_graph)

    def get_classification(self, image):
        print("Start clasification")
        with self.detection_graph.as_default():
            print ("time to classify a inamge ")
            #Expand dimension since the model expects image to have shape [1, None, None, 3].
            img_expanded = np.expand_dims(image, axis=0)
            ( scores, classes ) = self.sess.run(
                [ self.d_scores, self.d_classes],
                feed_dict={self.image_tensor: img_expanded})

            print("classes: ", classes)
            print("scores: ", scores)

        return TrafficLight.UNKNOWN
    # def get_classification(self, image):
    #     thread = threading.Thread(target=self.clasify_in_background, args=(image,))
    #     thread.daemon = True
    #     thread.start()


# clasifier = TLClassifier()
# image = Image.open('image3.png')
#
# start_time = str(datetime.now())
# for i in range (4):
#     print ("Item: ", i)
#     clasifier.clasify_in_background(image)
# end_time = str(datetime.now())
# print ("Start time: ", start_time)
# print ("End time  : ", end_time)
