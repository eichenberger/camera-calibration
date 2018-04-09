"""
File: camera-calibration.py
Author: Stefan Eichenberger
Email: eichest@gmail.com
Github: eichenberger
Description: This is a proof of concept for point cloud base camera calibration
"""
import math
import sys
import numpy as np
import cv2
from matplotlib import pyplot as plt
import multiprocessing as mp

#from pointmodel import PointModel
from model import Model
from orbdescriptors import OrbDescriptors
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



# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    print("Start camera calibration")

    pointcloudfile = sys.argv[1]
    imagefile = sys.argv[2]

    # Workaround so that orb compute doesn't crash
    cv2.ocl.setUseOpenCL(False)

    pointcloud = Model(pointcloudfile)
    image = cv2.imread(imagefile, 0)

    descriptors = OrbDescriptors(image)
    descriptors.extract()

    print("Extract descriptors")
    kps, matches = descriptors.match_descriptors(pointcloud.descriptors, 150)
    points2d_tmp, points3d = descriptors.get_points(pointcloud.points3d)
    points2d = np.ones((len(points2d_tmp), 3))
    points2d[:,0:2] = points2d_tmp

#    image2 = cv2.drawKeypoints(image, kps, None)
#    plt.imshow(image2)
#    plt.show()

    resolution = [image.shape[1], image.shape[0]]
#    res = do_parallel(resolution, points2d, points3d)
    print("Optimize camera model")
    valid_points, res = do_single(resolution, points2d, points3d)

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

    valid_kps = np.asarray(kps)[valid_points]
    image2 = cv2.drawKeypoints(image, valid_kps, None)
    image2 = cv2.undistort(image2, np.asarray([[fx_est, 0, cx_est],[0, fy_est, cy_est], [0, 0, 1]]), np.asarray([k1_est, k2_est, p1_est, p2_est]))

#    cm_est = CameraModel(resolution, [fx_est, fy_est], [cx_est, cy_est])
#    cm_est.add_distortion([k1_est, k2_est, k3_est], [p1_est, p2_est])
#    cm_est.create_extrinsic([thetax_est, thetay_est, thetaz_est], [tx_est, ty_est, tz_est])
#    cm_est.update_point_cloud(mod.points)
#    image_est = cm_est.get_image()


#    print("distance: {}".format(np.round((np.array(res.x)-np.array(should)), 2).tolist()))
#    print("tot-distance: {}".format(np.linalg.norm(np.array(res.x)-np.array(should))))
#
#    diff_img = image_est - image
#
#    plt.figure()
#    plt.imshow(diff_img)
#    plt.show()

    plt.imshow(image2)
    plt.show()

    print("Exit program")

if __name__ == '__main__':
    main()
