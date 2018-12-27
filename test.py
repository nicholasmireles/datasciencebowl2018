import keras 
import tensorflow as tf

from keras_maskrcnn import models
from keras_maskrcnn.utils.visualization import draw_mask
from keras_retinanet.utils.visualization import draw_box, draw_caption, draw_annotations
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.colors import label_color
from keras_retinanet.preprocessing.csv_generator import CSVGenerator

import numpy as np
import os

import cv2
from argparse import ArgumentParser

from time import time

class ImageGenerator:
    def __init__(self, image_dir):
        self.image_dir = image_dir
        self.image_paths = os.listdir(image_dir)
        self.num_images = len(self.image_paths)

        print("Found %i images." % self.num_images)

    def load_images(self):

        for i in range(self.num_images):
            path = self.image_paths[i]
            image_name = os.path.join(self.image_dir, path, "images", path + ".png")
            if not os.path.exists(image_name):
                continue

            image = read_image_bgr(image_name)
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            image = preprocess_image(image)
            image, scale = resize_image(image)

            yield i, path, image, draw, scale


def _masks_to_rles(masks, threshold=.5):
    results = [None for _ in range(len(masks))]
    for i, mask in enumerate(masks):
        results[i] = _mask_to_rle(mask,threshold=threshold)
    return results


def _mask_to_rle(mask, threshold=.5):
    line = np.where(mask.T.flatten() > threshold)[0]
    run_lengths = []
    prev = -2

    for b in line:
        if b > prev + 1: run_lengths.extend((b + 1, 0))
        run_lengths[-1] += 1
        prev = b
    return run_lengths


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def get_model(model_path):
    model = models.load_model(model_path, backbone_name='resnet50')
    return model


def get_detections(generator, model, out_dir, score_threshold=0.5):
    all_results = [None for _ in range(generator.num_images)]

    start = time()

    print("Processing %i images." % generator.num_images)

    for i, path, image, draw, scale in generator.load_images():
        outputs = model.predict_on_batch(np.expand_dims(image, axis=0))
        boxes = outputs[-4][0]
        scores = outputs[-3][0]
        labels = outputs[-2][0]
        masks = outputs[-1][0]
        boxes /= scale

        for box, score, label, mask in zip(boxes, scores, labels, masks):
            if score < score_threshold:
                break
            b = box.astype(int)
            mask = mask[:, :, label]
            draw_mask(draw, b, mask, color=label_color(label))

        out_name = os.path.join(out_dir, path + '.png')
        cv2.imwrite(out_name, draw)

        rles = _masks_to_rles(masks, threshold = score_threshold)
        all_results[i] = (path, boxes, rles)

    end = time()
    print("Processed images in %is"% (end - start))

    return all_results


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument('image_dir', help="The directory for the test images.")
    p.add_argument('out_dir', help="The directory to output the test images.")
    p.add_argument('model_path', help="The trained model to test.")
    p.add_argument('--threshold', default=.5, help="Threshold at which to filter the masks.")

    args = p.parse_args()

    generator = ImageGenerator(args.image_dir)

    keras.backend.tensorflow_backend.set_session(get_session())

    model = get_model(args.model_path)

    results = get_detections(generator, model, args.out_dir, score_threshold= float(args.threshold))

    valid = set()
    with open("results.csv", "w") as outFile:
        outFile.write("ImageId,EncodedPixels\n")
        for result in results:
            missing = 0
            for rle in result[2]:
                 if len(rle) > 0:
                    outFile.write(result[0] + "," + " ".join([str(i) for i in rle]) + "\n")
                 else:
                    missing += 1
            if missing < len(result[2]):
                valid.add(result[0])

    missing_paths = set(generator.image_paths) - valid

    for p in missing_paths:
        print("Missing RLEs for %s" % path)
