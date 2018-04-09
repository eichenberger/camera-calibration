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
from cameramodelestimator import CameraModelEstimator

def get_multiplier(n):
    if n == 0:
        return -1
    else:
        return 1

def create_x0(resolution, i):
    m_x = get_multiplier((i>>0)&0x1)
    m_y = get_multiplier((i>>1)&0x1)
    m_z = get_multiplier((i>>2)&0x1)

    x0 = [200, 200,
        resolution[0]/2, resolution[1]/2,
        m_x*math.pi/2, m_y*math.pi/2, m_z*math.pi/2,
        -0.5, -0.5, 1,
        100, 10, 1,
        10 , 1]
    return x0

def estimate(cme):
    return cme.estimate()

def do_parallel(resolution, points2d, points3d):
    cme_list = []
    for i in range(0,7):
        x0 = create_x0(resolution, i)
        cme = CameraModelEstimator(resolution, points2d, points3d)
        cme.set_x0(x0)
        cme_list.append(cme)

    pool = mp.Pool()
    res_list = pool.map(estimate, cme_list)
    res = res_list[0]
    # search min, update with real min
    for r in res_list:
        if r.fun < res.fun:
            res = r

    return res

def do_single(resolution, points2d, points3d):
    cme = CameraModelEstimator(resolution, points2d, points3d)
    x0 = create_x0(resolution, 0)
    cme.set_x0(x0)
    return cme.estimate(True)


# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    print("Start program")
    mod = PointModel2()
    mod.create_points(8)

    resolution = [640, 800]

    fx = 1150
    fy = 1100
    cx = resolution[0] / 2
    cy = resolution[1] / 2
    thetax = -math.pi/10
    thetay = math.pi/10
    thetaz = math.pi/8
    tx = -0.5
    ty = -0.5
    # How can we get rid of this stupid tz... (Distance from camera)
    tz = 6
    k1 = -0.2
    k2 = -0.01
    k3 = 0.001
    p1 = 0.01
    p2 = 0.001

    noise = 1
    miss_classified = 3

    cm = CameraModel(resolution, [fx, fy], [cx, cy])
    cm.add_distortion([k1, k2, k3], [p1, p2])
    cm.create_extrinsic([thetax, thetay, thetaz], [tx, ty, tz])
    cm.update_point_cloud(mod.points)
    image = cm.get_image()


    plt.imshow(image)

    points2d = cm.points2d
    points3d = mod.points

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

    cm.points2d = points2d
    image2 = cm.get_image()
    plt.figure()
    plt.imshow(image2)

#    res = do_parallel(resolution, points2d, points3d)
    res = do_single(resolution, points2d, points3d)


    should = [fx, fy, cx, cy, thetax, thetay, thetaz, tx, ty, tz, k1, k2, k3, p1, p2]
    print("res: {}\nis:       {}\nshould:   {}".format(res, np.round(res.x, 2).tolist(), should))

    fx_est = res.x[0]
    fy_est = res.x[1]
    cx_est = res.x[2]
    cy_est = res.x[3]
    thetax_est = res.x[4]
    thetay_est = res.x[5]
    thetaz_est = res.x[6]
    tx_est = res.x[7]
    ty_est = res.x[8]
    # How can we get rid of this stupid tz... (Distance from camera)
    tz_est = res.x[9]
    k1_est = res.x[10]
    k2_est = res.x[11]
    k3_est = res.x[12]
    p1_est = res.x[13]
    p2_est = res.x[14]

    cm_est = CameraModel(resolution, [fx_est, fy_est], [cx_est, cy_est])
    cm_est.add_distortion([k1_est, k2_est, k3_est], [p1_est, p2_est])
    cm_est.create_extrinsic([thetax_est, thetay_est, thetaz_est], [tx_est, ty_est, tz_est])
    cm_est.update_point_cloud(mod.points)
    image_est = cm_est.get_image()


    print("distance: {}".format(np.round((np.array(res.x)-np.array(should)), 2).tolist()))
    print("tot-distance: {}".format(np.linalg.norm(np.array(res.x)-np.array(should))))

    diff_img = image_est - image

    plt.figure()
    plt.imshow(diff_img)
    plt.show()

    print("Exit program")

if __name__ == '__main__':
    main()