"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import print_function

from keras_retinanet.utils.visualization import draw_detections

from .overlap import compute_overlap
from .visualization import draw_masks

import numpy as np
import os

import cv2

def _masks_to_rles(masks,threshold=.5):
    for mask in masks:
        _mask_to_rle(mask,threshold)

def _mask_to_rle(mask,threshold=.5):
    line = np.where(x.T.flatten() > threshold)[0]

    run_lengths = []
    prev = -2

    for b in line:
        if (b > prev+1): run_lengths.extend((b+1,0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths

def _get_detections(generator, model, score_threshold=0.05, max_detections=100, save_file=None):
    """ Get the detections from the model using the generator.

    The result is a list of lists such that the size is:
        all_detections[num_images][num_classes] = detections[num_detections, 4 + num_classes]

    # Arguments
        generator       : The generator used to run images through the model.
        model           : The model to run on the images.
        score_threshold : The score confidence threshold to use.
        max_detections  : The maximum number of detections to use per image.
        save_path       : The path to save the images with visualized detections to.
    # Returns
        A list of lists containing the detections for each image in the generator.
    """
    all_detections = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
    all_masks      = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

    for i in range(generator.size()):
        raw_image    = generator.load_image(i)
        image        = generator.preprocess_image(raw_image.copy())
        image, scale = generator.resize_image(image)
        image_name = generator.image_names[i]

        # run network
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes  = outputs[-4]
        scores = outputs[-3]
        labels = outputs[-2]
        masks  = outputs[-1]

        # correct boxes for image scale
        boxes /= scale

        # select indices which have a score above the threshold
        indices = np.where(scores[0, :] > score_threshold)[0]

        # select those scores
        scores = scores[0][indices]

        # find the order with which to sort the scores
        scores_sort = np.argsort(-scores)[:max_detections]

        # select detections
        image_boxes      = boxes[0, indices[scores_sort], :]
        image_scores     = scores[scores_sort]
        image_labels     = labels[0, indices[scores_sort]]
        image_masks      = masks[0, indices[scores_sort], :, :, image_labels]
        image_detections = np.concatenate([image_boxes, np.expand_dims(image_scores, axis=1), np.expand_dims(image_labels, axis=1)], axis=1)

        if save_file is not None:
            rles = masks_to_rles(image_masks)
            with open(save_file,"w") as outFile:
                outFile.write("ImageId,EncodedPixels\n")
                for rle in rles:
                    outFile.write(image_name + "," + " ".join(rle) + "\n")

        # copy detections to all_detections
        for label in range(generator.num_classes()):
            all_detections[i][label] = image_detections[image_detections[:, -1] == label, :-1]
            all_masks[i][label]      = image_masks[image_detections[:, -1] == label, ...]

        print('{}/{}'.format(i + 1, generator.size()), end='\r')

    return all_detections, all_masks
