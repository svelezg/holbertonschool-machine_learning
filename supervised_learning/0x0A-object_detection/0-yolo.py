#!/usr/bin/env python3
"""contains the Yolo class"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """
    uses the Yolo v3 algorithm to perform object detection
    """

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        class constructor
        :param self:
        :param model_path: path to where a Darknet Keras model is stored
        :param classes_path: path to where the list of class names used
            for the Darknet model, listed in order of index, can be found
        :param class_t: float representing the box score threshold
            for the initial filtering step
        :param nms_t: float representing the IOU threshold
            for non-max suppression
        :param anchors: numpy.ndarray of shape (outputs, anchor_boxes, 2)
            containing all of the anchor boxes:
            outputs is the number of outputs (predictions)
                made by the Darknet model
            anchor_boxes is the number of anchor boxes used for each prediction
            2 => [anchor_box_width, anchor_box_height]
        """
        self.model = K.models.load_model(model_path)
        with open(classes_path, 'r') as f:
            self.class_names = [line.strip() for line in f]
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors
