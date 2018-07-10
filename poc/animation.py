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
import cv2

#from pointmodel import PointModel
from pointmodel2 import PointModel2
from cameramodel import CameraModel
from cameramodelestimator2_animation import CameraModelEstimator

# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    print("Start program")
    mod = PointModel2()
    mod.create_points(10)

    resolution = [800, 800]

    fx = 1000
    fy = 1000
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    thetax = math.pi/10
    thetay = -math.pi/10
    thetaz = -math.pi/10
    tx = -0.4
    ty = -0.4
    tz = 2.5
    k1 = -2
    k2 = 0.0
    k3 = 0.0
    p1 = 0.0
    p2 = 0.0
    noise = 0
    noise_z = 0
    miss_classified = 0

    cm = CameraModel(resolution, [fx, fy], [cx, cy])
    cm.add_distortion([k1, k2, k3], [p1, p2])
    cm.create_extrinsic([thetax, thetay, thetaz], [tx, ty, tz])
    cm.update_point_cloud(mod.points)
    image = cm.get_image()

    points2d = cm.points2d
    points3d = np.ones((len(mod.points),4))
    points3d[:,0:3] = mod.points

    n = points3d.shape[0]
    # Add noise in z direction
    points3d[:,2] = points3d[:,2] + np.random.randn(n)*noise_z
    # Add noise
    points2d[:,0:2] = points2d[:,0:2] + np.random.randn(len(points2d), 2)*noise

    add = math.floor(np.random.rand()*50)
    i = add
    # Do some missmatching
    for j in range(0, miss_classified):
        i = (i + add) % len(points2d)
        new_x = math.floor(np.random.rand()*resolution[0])
        new_y = math.floor(np.random.rand()*resolution[1])
        points2d[i,0:2] = [new_x, new_y]
        print("Missclassify: {}".format(i))

#    cm.points2d = points2d
#    image2 = cm.get_image()
#    plt.figure()
#    plt.imshow(image2)

    max_iter = 30
    cme = CameraModelEstimator(resolution, points2d, points3d)
    cme._max_iter = max_iter
    res = cme.estimate(False)
    for i in range(1,21):
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

        print(res.x)

        cm_est = CameraModel(resolution, [fx_est, fy_est], [cx_est, cy_est])
        cm_est.add_distortion([k1_est, k2_est, k3_est], [p1_est, p2_est])
        cm_est.create_extrinsic([thetax_est, thetay_est, thetaz_est], [tx_est, ty_est, tz_est])
        cm_est.update_point_cloud(mod.points)
        image_est = cm_est.get_image()

        image1 = np.zeros((800,800,3))
        image2 = np.zeros((800,800,3))
        kernel = np.ones((5,5),np.uint8)
        image1[:,:,0] = image_est*255
        image2[:,:,1] = image*255

        diff_img = image1 + image2
        diff_img = cv2.dilate(diff_img, kernel, iterations=1)

        cv2.imwrite("animation/img" + str(i) + ".png", diff_img)
        res = cme.continue_est()


    print("Exit program")

if __name__ == '__main__':
    main()
