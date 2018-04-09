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

    def match_descriptors(self, trainingset, maxdistance):
        matcher = cv2.BFMatcher()
        matches = matcher.match(self._descriptors, trainingset)

        good_kps = []
        good_matches = []
        kp = self._kp
        # ratio test as per Lowe's paper
        for i, match in enumerate(list(matches)):
            if match.distance < maxdistance:
                good_kps.append(kp[i])
                good_matches.append(matches[i])

        self._matches = matches
        self._good_matches = good_matches
        self._good_kps = good_kps

        return good_kps, good_matches

    def get_points(self, points3d):
        good_kps = self._good_kps
        good_matches = self._good_matches

        good_points2d = [[0, 0]]*len(good_matches)
        for i, kp in enumerate(good_kps):
            good_points2d[i] = [kp.pt[0], kp.pt[1]]

        good_points3d = [[0, 0, 0]]*len(good_matches)
        for i, match in enumerate(good_matches):
            good_points3d[i] = points3d[match.trainIdx]

        return good_points2d, np.asarray(good_points3d)


