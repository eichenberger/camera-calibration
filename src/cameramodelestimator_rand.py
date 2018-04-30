import math
import numpy as np
import scipy.optimize as opt
from cameramodel import CameraModel
import cv2

class CameraModelEstimator:
    """docstring for CameraModelEstimator"""
    def __init__(self, resolution, points2d, points3d):
        self._points2d = points2d
        self._points3d = points3d
        self._points2d_reduced = []
        self._points3d_reduced = []
        self._resolution = resolution
        self._cm = CameraModel(resolution)
        self._x0 = [200, 200,
                    self._resolution[0]/2, self._resolution[1]/2,
                    math.pi/2, math.pi/2, math.pi/2,
                    -0.5, -0.5, 1,
                    0, 0, 0,
                    0 ,0]

    def _loss_lm(self, x):
        fx = x[0]
        fy = x[1]
        cx = x[2]
        cy = x[3]
        thetax = x[4]
        thetay = x[5]
        thetaz = x[6]
        tx = x[7]
        ty = x[8]
        tz = x[9]
        k1 = x[10]
        k2 = x[11]
        k3 = x[12]
        p1 = x[13]
        p2 = x[14]

        self._cm.set_c([cx, cy])
        self._cm.set_f([fx, fy])

        self._cm.create_extrinsic([thetax, thetay, thetaz], [tx, ty, tz])
        self._cm.add_distortion([k1, k2, k3], [p1, p2])

        self._cm.update_point_cloud(self._points3d_reduced)
        points2d_est = self._cm.points2d

        dists = points2d_est[:, 0:2] - self._points2d_reduced
        res = list(map(lambda dist: np.dot(dist, dist), dists))

        return res

    def set_x0(self, x0):
        self._x0 = x0

    def _guess_pose(self):
        camera_matrix = np.array([[200.0, 0.0, self._resolution[0]/2],
                         [0.0, 200.0, self._resolution[1]/2],
                         [0.0, 0.0, 1.0]])
        dist_coeff = np.array([])

        points3d = np.array([self._points3d])
        points2d = np.array([self._points2d[:,0:2]])

        res = cv2.solvePnPRansac(points3d, points2d, camera_matrix, dist_coeff)
        # return the pose, the translation and all valid points
        return res[1], res[2], res[3].transpose()[0]

    def estimate(self, guess_pose = False):
        self._points2d_reduced = self._points2d
        self._points3d_reduced = self._points3d
        # If we guess the pose via pnp
        if guess_pose:
            rvec, tvec, valid_points = self._guess_pose()
            # TODO: remove creepy conversion
            self._x0[4] = rvec[0][0]
            self._x0[5] = rvec[1][0]
            self._x0[6] = rvec[2][0]
            self._x0[7] = tvec[0][0]
            self._x0[8] = tvec[1][0]
            self._x0[9] = tvec[2][0]
            # Throwaway points that are probably missmatches
            #self._points2d_reduced = self._points2d[valid_points]
            #self._points3d_reduced = self._points3d[valid_points]

        x0 = [0]*15
        min_res = None
        for i in range(0, 500):
            x0[0] = np.random.rand()*2000+10
            x0[1] = np.random.rand()*2000+10
            x0[2] = self._resolution[0]/2
            x0[3] = self._resolution[0]/2
            x0[4] = 2*np.random.rand()*math.pi
            x0[5] = 2*np.random.rand()*math.pi
            x0[6] = 2*np.random.rand()*math.pi
            x0[7] = np.random.rand()*10-5
            x0[8] = np.random.rand()*10-5
            x0[9] = np.random.rand()*10-5
            x0[10] = np.random.rand()*5
            x0[11] = np.random.rand()*3
            x0[12] = np.random.rand()*5
            x0[13] = np.random.rand()
            x0[14] = np.random.rand()

            res = opt.least_squares(self._loss_lm, x0,
                                    method = 'lm', max_nfev = 200)
            if min_res is None or res.optimality <= min_res.optimality:
                min_res = res

        print("Done best result: {}".format(min_res.optimality))

        x0 = min_res.x
        res = opt.least_squares(self._loss_lm, x0,
                                method = 'lm')
        return res

