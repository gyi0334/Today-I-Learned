import os
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from lib.interface import waitKey
import argparse
import numpy as np
import sys
import time
import cv2
import imutils
from keras.models import load_model

# Model path
face_detection_model_path = 'models/haarcascade_frontalface_default.xml'
emotion_model_path = 'models/xception_batch_20.hdf5'

# Face detect and emotion classifier
face_detection = cv2.CascadeClassifier(face_detection_model_path)
emotion_classfier = load_model(emotion_model_path, compile=False)
EMOTIONS = ["normal", "abnormal"]

# Input camera or video
camera = cv2.VideoCapture(0)
# cap = cv2.VideoCapture('video/test_vedeo_20_emotion_10_01.mp4')

class getPulseApp():
    def __init__(self, args):
        self.cameras = []
        self.pressed = 0
        self.frame_in = np.zeros((10,10))
        self.data_buffer = []
        self.times = []
        self.trained = False
        self.samples = []
        self.face_rect = [1, 1, 2, 2]
        self.last_center = np.array([0, 0])
        self.t0 = time.time()
        self.idx = 1
        self.bpm = 0
        self.fft = []

    # Camera close
    def key_handler(self):
        self.pressed = waitKey(10) & 255
        if self.pressed == 27:  # ESC
            for cam in self.cameras:
                cam.cam.release()
            sys.exit()

    # Find forehead
    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]