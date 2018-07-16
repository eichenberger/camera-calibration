"""
File: camera-calibration.py
Author: Stefan Eichenberger
Email: eichest@gmail.com
Github: eichenberger
Description: This is a proof of concept for point cloud base camera calibration
"""
import math
import sys
import io
import json

import numpy as np
import cv2
from matplotlib import pyplot as plt
import multiprocessing as mp

#from pointmodel import PointModel
from model import Model
from orbdescriptors import OrbDescriptors
from cameramodelestimator import CameraModelEstimator
from cameraparams import CameraParams

def get_multiplier(n):
    if n == 0:
        return -1
    else:
        return 1

def estimate(cme):
    return cme.estimate()

def do_estimate(resolution, points2d, points3d):
    cme = CameraModelEstimator(resolution, points2d, points3d)
    return cme.estimate()

def plot_matches(image, image_descriptors, image_kps, pointcloud_matches,
                 keyframefile, keyframe_infos):

    keyframe = cv2.imread(keyframefile)

    with io.open(keyframe_infos, 'r') as f:
        keyframe_infos = json.JSONDecoder().decode(f.read())

    indexes = pointcloud_matches[:,0].astype(int)
    image_descriptors_subset = image_descriptors[indexes]
    image_kps_subset = np.asarray(image_kps)[indexes]

    keyframe_descriptors = np.asarray(keyframe_infos['descriptors'], dtype=np.uint8)
    keyframe_positions = np.asarray(keyframe_infos['position'])
    kf_matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    kf_matches = kf_matcher.match(keyframe_descriptors, image_descriptors_subset)

    kf_kps = np.asarray(list(map(lambda point: cv2.KeyPoint(point[0], point[1], 10),
                                 keyframe_positions)))

    sorted_matches = sorted(kf_matches, key=lambda match: match.distance)
    out_image = image.copy()
    out_image = cv2.drawMatches(keyframe, kf_kps, image, image_kps_subset, sorted_matches, out_image)
    plt.imshow(out_image)
    plt.show()

def array_string(array, space):
    output = ""
    for entry in array:
        entry = str(entry)
        output = output + entry
        output = output + " "*(space - len(entry))

    return output

# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    print("Start camera calibration")

    pointcloudfile = sys.argv[1]
    imagefile = sys.argv[2]
    keyframefile = None
    keyframe_infos = None
    if len(sys.argv) > 3:
        keyframefile = sys.argv[3]
        keyframe_infos = sys.argv[4]
    # Workaround so that orb compute doesn't crash
    cv2.ocl.setUseOpenCL(False)

    pointcloud = Model(pointcloudfile)
    image = cv2.imread(imagefile, 0)

    print("Extract descriptors")
    descriptors = OrbDescriptors(image)
    descriptors.extract()

    print("Match keypoints")
    matches = descriptors.match_descriptors(pointcloud.descriptors)
    matches = np.asarray(matches)

    if keyframefile:
        plot_matches(image, descriptors._descriptors, descriptors.get_key_points(),
                     matches, keyframefile, keyframe_infos)


    kps =  descriptors.get_key_points()
    kps = np.asarray(kps)

    kp_indexes = matches[:, 0].astype(int)
    matched_kps = kps[kp_indexes]
    tmp_points2d = np.array(list(map(lambda kp: np.asarray(kp.pt), matched_kps)))
    points2d = np.ones((tmp_points2d.shape[0],3))
    points2d[:, 0:2] = tmp_points2d

    kp_image = image.copy()
    kp_image = cv2.drawKeypoints(image, matched_kps, kp_image)
    plt.imshow(kp_image)
    plt.show()

    p3d_indexes = matches[:,1].astype(int)
    points3d = pointcloud.points3d[p3d_indexes,:]

    resolution = [image.shape[1], image.shape[0]]
    print("Optimize camera model")
    params, res = do_estimate(resolution, points2d, points3d)
    params = CameraParams(params)

    #print("res: {}\nis:       {}".format(res, np.round(res.x, 2).tolist()))
    print("Found camera parameters:")
    print(array_string([" ", "fx", "fy", "cx", "cy", "thetax", "thetay", "thetaz",
                        "tx", "ty", "tz", "k1", "k2", "k3", "p1", "p2"], 8))
    print("values: " + params.get_string(8, 2))

    image2 = cv2.drawKeypoints(image, matched_kps, None)
    image2 = cv2.undistort(image2, np.asarray([[params.fx, 0, params.cx],
                                               [0, params.fy, params.cy],
                                               [0, 0, 1]]),
                           np.asarray([params.k1, params.k2,
                                       params.p1, params.p2]))

    plt.imshow(image2)
    plt.show()

    print("Exit program")

if __name__ == '__main__':
    main()
