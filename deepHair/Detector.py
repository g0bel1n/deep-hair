import os
from typing import Any
import cv2
from matplotlib import pyplot as plt
import numpy as np
import yaml


def format_yolo(source):
    """
    Given an image, resize it to the largest dimension and fill the rest with zeros
    
    :param source: The image to be transformed
    :return: a blob from the image.
    """

    col, row, _ = source.shape
    _max = max(col, row)
    resized = np.zeros((_max, _max, 3), np.uint8)
    resized[0:col, 0:row] = source

    return cv2.dnn.blobFromImage(resized, 1/255.0, (640, 640), swapRB=True)

class Detector:

    def __init__(self,config: dict, model = '', threshold = 0.4) -> None:
        """
        Reads the weights and config file of the model and loads the model
        
        :param model: The model to be used
        :param threshold: This is the confidence threshold
        """
        absolutePath  = f"{config['model_path']}/yolov4{model}."
        self.MODEL = cv2.dnn.readNet(f'{absolutePath}weights', f'{absolutePath}cfg')
        self.THRESHOLD = threshold
        self.CLASSES = []
        with open("models/coco-names.txt", "r") as f:
            self.CLASSES = [line.strip() for line in f.readlines()]


    def evaluate(self, img) -> bool:
        """
        Given an image, it will return True if a person is detected in the image, and False otherwise
        
        :param img: the image to be classified
        :return: A boolean value. True if the object is detected, False otherwise.
        """

        formatted_img = format_yolo(img)
        plt.imshow(img)
        self.MODEL.setInput(formatted_img)
        output = self.MODEL.forward()
        #output = output[output[:,5]>self.THRESHOLD]

        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            #print(self.CLASSES[class_id], confidence)
            if class_id == 0 and confidence > self.THRESHOLD : 
                return True
        return False


if __name__=='__main__':
    print (os.getcwd())
    obj = Detector()