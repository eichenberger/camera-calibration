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
from cameramodel import CameraModel
from cameramodelestimator_rand import CameraModelEstimator

def get_multiplier(n):
    if n == 0:
        return -1
    else:
        return 1

def create_x0(resolution, i):
    m_x = get_multiplier((i>>0)&0x1)
    m_y = get_multiplier((i>>1)&0x1)
    m_z = get_multiplier((i>>2)&0x1)

    x0 = [1333, 1339,
        629, 362,
        m_x*math.pi/2, m_y*math.pi/2, m_z*math.pi/2,
        0, 0, 0,
        0.3, -2, 6.6,
        0 , 0]
    return x0

def estimate(cme):
    return cme.estimate()

def do_single(resolution, points2d, points3d):
    cme = CameraModelEstimator(resolution, points2d, points3d)
    x0 = create_x0(resolution, 0)
    cme.set_x0(x0)
    return cme.estimate(True)

def show_res(res):
    print("fx {}".format(res.x[0]))
    print("fy {}".format(res.x[1]))
    print("cx {}".format(res.x[2]))
    print("cy {}".format(res.x[3]))
    print("thetax {}".format(res.x[4]))
    print("thetay {}".format(res.x[5]))
    print("thetaz {}".format(res.x[6]))
    print("tx {}".format(res.x[7]))
    print("ty {}".format(res.x[8]))
    print("tz {}".format(res.x[9]))
    print("k1 {}".format(res.x[10]))
    print("k2 {}".format(res.x[11]))
    print("k3 {}".format(res.x[12]))
    print("p1 {}".format(res.x[13]))
    print("p2 {}".format(res.x[14]))


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
    points2d = np.array(list(map(lambda kp: np.asarray(kp.pt), matched_kps)))

    kp_image = image.copy()
    kp_image = cv2.drawKeypoints(image, matched_kps, kp_image)
    plt.imshow(kp_image)
    plt.show()

    p3d_indexes = matches[:,1].astype(int)
    points3d = pointcloud.points3d[p3d_indexes,:]

    resolution = [image.shape[1], image.shape[0]]
    print("Optimize camera model")
    res = do_single(resolution, points2d, points3d)

    #print("res: {}\nis:       {}".format(res, np.round(res.x, 2).tolist()))
    show_res(res)

    fx_est = res.x[0]
    fy_est = res.x[1]
    cx_est = res.x[2]
    cy_est = res.x[3]
    thetax_est = res.x[4]
    thetay_est = res.x[5]
    thetaz_est = res.x[6]
    tx_est = res.x[7]
    ty_est = res.x[8]
    tz_est = res.x[9]
    k1_est = res.x[10]
    k2_est = res.x[11]
    k3_est = res.x[12]
    p1_est = res.x[13]
    p2_est = res.x[14]

    image2 = cv2.drawKeypoints(image, matched_kps, None)
    image2 = cv2.undistort(image2, np.asarray([[fx_est, 0, cx_est],[0, fy_est, cy_est], [0, 0, 1]]), np.asarray([k1_est, k2_est, p1_est, p2_est]))

    plt.imshow(image2)
    plt.show()

    print("Exit program")

if __name__ == '__main__':
    main()
