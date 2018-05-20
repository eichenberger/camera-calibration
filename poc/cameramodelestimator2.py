import math
import numpy as np
import scipy.optimize as opt
from cameramodel import CameraModel
import cv2
import random

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
        if len(x) == 13:
            k1 = x[10]
            k2 = x[11]
            k3 = 0
            p1 = x[12]
            p2 = 0
        else:
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

    def _rq(self, M):
        # User algorithm for RQ from QR decomposition from
        # https://math.stackexchange.com/questions/1640695/rq-decomposition
        P = np.fliplr(np.diag(np.ones(len(M))))
        Mstar = np.matmul(P, M)
        Qstar, Rstar = np.linalg.qr(np.transpose(Mstar))
        Q = np.matmul(P, np.transpose(Qstar))
        R = np.matmul(P, np.matmul(np.transpose(Rstar), P))

        # Now make the diagonal of R positiv. This can be done, because
        # we know that f, and c is allways positive
        T = np.diag(np.sign(np.diag(R)))

        R = np.matmul(R, T)
        Q = np.matmul(Q, T)

        return R, Q

    def _convert_for_equation(self, XYZ, uv):
        n = XYZ.shape[0]
        # XYZ and uv must be in rows not in columns!
        A = np.zeros((2*n, 11))
        uv = np.mat(uv)
        # This is triky we need the form
        # [[C11*X, C12*Y, C13*Z, C14, 0, 0, 0, 0, -u*C31*X, -u*C32*Y, -uC33*Z],
        #  [0, 0, 0, 0, C21*X, C22*y, C23*Z, C24, -u*C31*X, -u*C32*Y, -uC33*Z]]
        # The following black magic will do the trick
        A[0::2] = np.concatenate((XYZ, np.ones((n, 1)), np.zeros((n,4)),
                                  -np.multiply(np.multiply(np.ones((n,3)),uv[:,0]),XYZ)),
                                 axis=1)
        A[1::2] = np.concatenate((np.zeros((n, 4)), XYZ, np.ones((n,1)),
                                  -np.multiply(np.multiply(np.ones((n,3)),uv[:,1]),XYZ)),
                                 axis=1)

        # This is simple, we need all uvs flattened to a column vector
        B = np.reshape(uv, (2*n, 1))
        return A, B

    def _select_points(self, points3d, points2d):
        POINTS = 4
        # Random sample gives back unique values
        sel = random.sample(range(0,len(points3d)), POINTS)
        sel3d = points3d[sel,:]
        sel2d = points2d[sel,:]
        return sel3d, sel2d


    def _guess_transformation(self, points3d, points2d):
#        points3d_cal = np.ones((points3d.shape[0], 4))
#        points3d_cal[:,0:3] = points3d[:,0:3]
#
#        points2d_cal = np.ones((points2d.shape[0], 3))
#        points2d_cal[:,0:2] = points2d[:,0:2]

        # We can't do a direct linear least square, we first need to create
        # the lineare equation matrix see robotics and control 332 and camcald.m
        # from the corresponding Matlab toolbox
        A, B = self._convert_for_equation(points3d[:,0:3], points2d[:,0:2])
        # least square should solve the problem for us
        res = np.linalg.lstsq(A, B, rcond=None)
        # Now we have all unknown parameters and we have to bring it to
        # the normal 3x4 matrix. The last parameter C34 is 1!
        C = np.reshape(np.concatenate((res[0], [[1]])), (3, 4))
        print("Found transformation matrix: {}".format(C))

        # Make rq matrix decomposition, transformation is not taken into account
        intrinsic, extrinsic = self._rq(C[0:3,0:3])
        intrinsic = np.multiply(intrinsic, 1/intrinsic[2,2])
        print("Intrinsic: {}\nExtrinsic: {}".format(intrinsic, extrinsic))

        self._C = C
        self._intrinsic = np.asarray(np.abs(intrinsic))
        t = np.linalg.lstsq(intrinsic, C[:,3], rcond=None)[0]
        self._extrinsic = np.asarray(np.concatenate((extrinsic, t), axis=1))

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

    def estimate(self, reduce_dist_param=False):
        self._points2d_reduced = self._points2d
        self._points3d_reduced = self._points3d

#        # Do ransac to find outliers, throwaway other points
#        rvec, tvec, valid_points = self._guess_pose()
#        # Throwaway points that could be missmatches
#        self._points2d_reduced = self._points2d[valid_points]
#        self._points3d_reduced = self._points3d[valid_points]

        # Guess initial intrinsic and extrinsic mat with linear least squares
        self._guess_transformation(self._points3d, self._points2d)

        # If we don't use the full intrinsic and extrensic matrix,
        # we can safe a lot of parameters to optimize! We pay that with
        # the overhead of calculate sin/cos/tan
        fx = self._intrinsic[0,0]
        fy = self._intrinsic[1,1]
        cx = self._intrinsic[0,2]
        cy = self._intrinsic[1,2]
        # Calculate angles
        rot_x = np.arctan2(self._extrinsic[1,2], self._extrinsic[2,2])
        rot_y = np.arctan2(self._extrinsic[0,2], np.sqrt(self._extrinsic[1,2]**2 + self._extrinsic[2,2]**2))
        rot_z = np.arctan2(self._extrinsic[0,1], self._extrinsic[0,0])
        tx = self._extrinsic[0,3]
        ty = self._extrinsic[1,3]
        tz = self._extrinsic[2,3]

        # Create an array from the intrinsic, extrinsic and k0-k2, p0-p1
        if reduce_dist_param:
            dist_params = [0, 0, 0]
        else:
            dist_params = [0, 0, 0, 0, 0]

        x0 = np.concatenate(([fx, fy, cx, cy],
                             [rot_x, rot_y, rot_z, tx, ty, tz],
                             dist_params))
        # Use least squares minimization with levenberg-marquardt algorithm
        return opt.least_squares(self._loss_lm, x0,
                                 method = 'lm')

