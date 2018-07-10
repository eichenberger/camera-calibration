"""
File: poc.py
Author: Stefan Eichenberger
Email: eichest@gmail.com
Github: eichenberger
Description: This is a proof of concept for point cloud base camera calibration
"""
import math
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp

#from pointmodel import PointModel
from pointmodel2 import PointModel2
from cameramodel import CameraModel
from cameramodelestimator2 import CameraModelEstimator

# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    print("Start program")
    mod = PointModel2()
    mod.create_points(20)

    resolution = [800, 800]

    fx = 1000
    fy = 1000
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    thetax = 0.01
    thetay = 0
    thetaz = 0
    tx = -0.5
    ty = -0.5
    tz = 3
    k1 = 0.8
    k2 = 0.3
    k3 = 0.2
    p1 = 0.3
    p2 = 0.1

    cm = CameraModel(resolution, [fx, fy], [cx, cy])
    cm.add_distortion([0, 0, 0], [0, 0])
    cm.create_extrinsic([thetax, thetay, thetaz], [tx, ty, tz])
    cm.update_point_cloud(mod.points)
    image1 = cm.get_image()

    cm = CameraModel(resolution, [fx, fy], [cx, cy])
    cm.add_distortion([k1, k2, k3], [p1, p2])
    cm.create_extrinsic([thetax, thetay, thetaz], [tx, ty, tz])
    cm.update_point_cloud(mod.points)
    image2 = cm.get_image()

    plt.subplot(121)
    plt.imshow(image1)
    plt.subplot(122)
    plt.imshow(image2)
    plt.show()

if __name__ == '__main__':
    main()
