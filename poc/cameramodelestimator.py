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

    def _loss(self, x):
        fx = x[0]
        fy = x[1]
        cx = x[2]
        cy = x[3]
        thetax = x[4]
        thetay = x[5]
        thetaz = x[6]
        tx = x[7]
        ty = x[8]
        # Fix it to 1 we can't get the scale
        tz = 1
        k1 = x[9]
        k2 = x[10]
        k3 = x[11]
        p1 = x[12]
        p2 = x[13]

        self._cm.set_c([cx, cy])
        self._cm.set_f([fx, fy])

        self._cm.create_extrinsic([thetax, thetay, thetaz], [tx, ty, tz])
        self._cm.add_distortion([k1, k2, k3], [p1, p2])

        self._cm.update_point_cloud(self._points3d_reduced)
        points2d_est = self._cm.points2d

        dists = points2d_est - self._points2d_reduced

        # Calculate the squared distance over all points as loss function
        res = 0
        for i, dist in enumerate(dists):
            d = dist[0]**2+dist[1]**2
            res = res + d

        # It works better with log than without log... Don't know why yet:(
        return math.log(res)
        #return res

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

        dists = points2d_est - self._points2d_reduced
        res = np.zeros(len(dists))
        # Calculate the squared distance over all points as loss function
        for i, dist in enumerate(dists):
            res[i] = np.dot(dist, dist)

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
        bounds_lb = [(1.0, 1000.0),(1.0, 1000.0),
                  (self._resolution[0]/2 - 20, self._resolution[0]/2 + 20),
                  (self._resolution[1]/2 - 20, self._resolution[1]/2 + 20),
                  (-3.15, 3.15), (-3.15, 3.15), (-3.15/2, 3.15/2),
                  (-100.0, 100.0), (-100.0, 100.0),
                  (-1.0, 1.0), (-0.5, 0.5), (-0.2, 0.2),
                  (-1.0, 1.0), (-0.5, 0.5)]


        bounds_lm = ([1,1,
                   0,0,
                   0,0,0,
                   -100, -100,
                   -3, -3, -3,
                   -3, -3],
                  [1000, 1000,
                   self._resolution[0], self._resolution[1],
                   2*math.pi, 2*math.pi, 2*math.pi,
                   100, 100,
                   3, 3, 3,
                   3, 3])

        scale = [1e-4, 1e-4,
                 1e-6, 1e-6,
                 1e-8, 1e-8, 1e-8,
                 1e-8, 1e-8,
                 1e-8, 1e-8, 1e-8,
                 1e-8, 1e-8]

#        return opt.minimize(self._loss, x0,
#                            bounds = bounds,
#                            method = 'L-BFGS-B')
#
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
            # Throwaway points that could be missmatches
            self._points2d_reduced = self._points2d[valid_points]
            self._points3d_reduced = self._points3d[valid_points]

#                            tol = 1e-15,
#                            bounds = bounds_lb,
#                            eps = scale,
#                                     'ftol': 1e-15,
#                                     'gtol': 1e-15
                            #method = 'L-BFGS-B',

        # Use least squares minimization with levenberg-marquardt algorithm
        return opt.least_squares(self._loss_lm, self._x0,
                                 method = 'lm')


#        return opt.minimize(self._loss, self._x0,
#                            method = 'BFGS',
#                            options={'eps': scale})



