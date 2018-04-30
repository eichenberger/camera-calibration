import cv2
import numpy as np

class OrbDescriptors:
    def __init__(self, image):
        self._image = image

    def extract(self):
        """docstring for extract"""
        orb = cv2.ORB_create(2000, 1.4, 8, fastThreshold=20)
        kp = orb.detect(self._image, None)
        kp, descriptors = orb.compute(self._image, kp)
        self._kp = kp
        self._descriptors = descriptors

    def match_descriptors(self, trainingset):
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = matcher.match(self._descriptors, trainingset)

        sorted_matches = sorted(matches, key=lambda match: match.distance)
        matches = list(map(lambda match: [match.queryIdx,
                                          match.trainIdx,
                                          match.distance], sorted_matches))

        return matches

    def get_key_points(self):
        return self._kp

