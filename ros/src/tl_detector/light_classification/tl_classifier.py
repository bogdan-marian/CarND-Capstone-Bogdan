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


class TLClassifier(object):
    def __init__(self):

        # it takes time tu run the clasifyer so it is best to just return the
        # last known classification. Homework for later: bilt a faster model
        self.lock = Lock()
        self.lastLight = TrafficLight.UNKNOWN

        #initializing tensorflow
        dir_path = os.path.dirname(os.path.realpath(__file__))
        PATH_TO_MODEL = dir_path + '/frozen_inference_graph.pb'
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

    def clasify_in_background(self, image):

        if not self.lock.acquire(False):
            print("Failed to lock resource")
        else:
            try:
                print("Working on something")
                with self.detection_graph.as_default():
                    print ("time to classify a inamge ")
                    #Expand dimension since the model expects image to have shape [1, None, None, 3].
                    img_expanded = np.expand_dims(image, axis=0)
                    ( scores, classes ) = self.sess.run(
                        [ self.d_scores, self.d_classes],
                        feed_dict={self.image_tensor: img_expanded})
                    print("classes: ", classes)
                    print("scores: ", scores)
            finally:
                self.lock.release()



    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """


        thread = threading.Thread(target=self.clasify_in_background, args=(image,))
        thread.daemon = True
        thread.start()

        return self.lastLight


        # if self.working:
        #     return self.lastLight
        # else:
        #     thread = threading.Thread(target=self.clasify_in_background, args=(image,))
        #     #thread.daemon = True
        #     thread.start()
        #     return TrafficLight.UNKNOWN

        #self.clasify_in_background(image)
        #return TrafficLight.UNKNOWN
