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
        self._points2d_inliers = []
        self._points3d_inliers = []
        self._inliers = []
        self._resolution = resolution
        self._reduce_dist_param = False
        self._cm = CameraModel(resolution)
        self._max_iter = None

    def _loss_dist(self, x):
        k1 = x[0]
        k2 = x[1]
        k3 = x[2]
        p1 = x[3]
        p2 = x[4]

        self._cm.add_distortion([k1, k2, k3], [p1, p2])

        self._cm.update_point_cloud(self._points3d_inliers)
        points2d_est = self._cm.points2d

        dists = points2d_est - self._points2d_inliers

        return dists.flatten()

    def _loss_full(self, x):
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

        self._cm.update_point_cloud(self._points3d_inliers)
        points2d_est = self._cm.points2d

        dists = points2d_est - self._points2d_inliers

        return dists.flatten()


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

    def _select_points(self, points3d, points2d, n=4):
        # Random sample gives back unique values
        sel = random.sample(range(0,len(points3d)), n)
        sel3d = points3d[sel,:]
        sel2d = points2d[sel,:]
        return sel3d, sel2d

    def _guess_transformation(self, points3d, points2d):
        max_matches = 0
        transformation_mat = None
        best_reprojection_error = None
        inliers = None
        points2d_est = None
        # Try to find the best match within n tries
        for i in range(0, 1000):
            sel3d, sel2d = self._select_points(points3d, points2d, 10)
            # We can't do a direct linear least square, we first need to create
            # the lineare equation matrix see robotics and control 332 and camcald.m
            # from the corresponding Matlab toolbox
            A, B = self._convert_for_equation(sel3d[:,0:3], sel2d[:,0:2])
            # least square should solve the problem for us
            res = np.linalg.lstsq(A, B, rcond=None)
            # Now we have all unknown parameters and we have to bring it to
            # the normal 3x4 matrix. The last parameter C34 is 1!
            C = np.reshape(np.concatenate((res[0], [[1]])), (3, 4))
            points2d_transformed = np.transpose(np.matmul(C, np.transpose(points3d)))
            points2d_transformed = points2d_transformed/points2d_transformed[:,2]
            points2d_diff = points2d - points2d_transformed
            reprojection_error = list(map(lambda diff: np.linalg.norm(diff), points2d_diff))
            inliers = [i for i,err in enumerate(reprojection_error) if err < 16]
            matches = len(inliers)
            if matches > max_matches:
                intrinsic, extrinsic = self._rq(C[0:3,0:3])
                intrinsic = np.multiply(intrinsic, 1/intrinsic[2,2])

                if math.fabs(intrinsic[0,1]) > 10:
                    continue

                self._C = C
                self._intrinsic = np.asarray(intrinsic)
                t = np.linalg.lstsq(intrinsic, C[:,3], rcond=None)[0]
                self._extrinsic = np.asarray(np.concatenate((extrinsic, t), axis=1))
                max_matches = matches
                best_reprojection_error = reprojection_error
                self._inliers = inliers
                points2d_est = points2d_transformed
#        print("i: " + str(i))
#        print("Max matches: " + str(max_matches))
#        print("Match percentage: " + str((max_matches/len(points2d))*100))
#        print("Found transformation matrix: {}".format(C))
#
#        # Make rq matrix decomposition, transformation is not taken into account
#        print("Intrinsic: {}\nExtrinsic: {}".format(intrinsic, extrinsic))

        return points2d_est

    def _guess_distortion_lin(self, points2d_are, points2d_est, f, c):
        # uv = [u, v]
        uv = np.asarray((points2d_est[:,0:2]) - c)
        xy = (uv)/f
        radius = np.linalg.norm(xy, axis=(1))
        x = xy[:,0]
        y = xy[:,1]
        # Because uv[:,0] is a onedimensional array we need to transpose M1 so
        # that we have column vectors
        M1 = np.transpose(np.array([x*radius**2, x*radius**4, x*radius**6,
                                    2*x*y, radius**2+2*x**2]))
        M2 = np.transpose(np.array([y*radius**2, y*radius**4, y*radius**6,
                                    radius**2+2*y**2, 2*y*x]))
        M = np.zeros((M1.shape[0]*2,M1.shape[1]))
        M[0::2] = M1*f[0]
        M[1::2] = M2*f[1]
        delta = points2d_are[:,0:2] - points2d_est[:,0:2]
        delta = delta.reshape(delta.shape[0]*2, 1)
        distortion = np.linalg.lstsq(M, delta, rcond=-1)
        print("Distortion: {}".format(distortion[0]))
        return distortion


    def estimate(self):
        self._points2d_inliers = self._points2d
        self._points3d_inliers = self._points3d

        # Guess initial intrinsic and extrinsic mat with linear least squares
        points2d_est = self._guess_transformation(self._points3d, self._points2d)
        self._points2d_inliers = self._points2d[self._inliers]
        self._points3d_inliers = self._points3d[self._inliers]

        # If we don't use the full intrinsic and extrensic matrix,
        # we can safe a lot of parameters to optimize! We pay that with
        # the overhead of calculate sin/cos/tan
        fx = self._intrinsic[0,0]
        fy = self._intrinsic[1,1]
        cx = self._intrinsic[0,2]
        cy = self._intrinsic[1,2]

        points2d_est_inliers = points2d_est[self._inliers]
        # Does not work
#        self._guess_distortion_lin(self._points2d,
#                                   points2d_est,
#                                   [fx, fy], [cx, cy])
#
        # Calculate angles
        rot_x = np.arctan2(self._extrinsic[1,2], self._extrinsic[2,2])
        rot_y = np.arctan2(self._extrinsic[0,2], np.sqrt(self._extrinsic[1,2]**2 + self._extrinsic[2,2]**2))
        rot_z = np.arctan2(self._extrinsic[0,1], self._extrinsic[0,0])
        tx = self._extrinsic[0,3]
        ty = self._extrinsic[1,3]
        tz = self._extrinsic[2,3]

        # Create an array from the intrinsic, extrinsic and k0-k2, p0-p1
        x0 = [0, 0, 0, 0, 0]

        self._cm.set_c([cx, cy])
        self._cm.set_f([fx, fy])
        self._cm.create_extrinsic([rot_x, rot_y, rot_z], [tx, ty, tz])
        # Use least squares minimization with levenberg-marquardt algorithm
        res = opt.least_squares(self._loss_dist, x0,
                                 method = 'lm',
                                 max_nfev = self._max_iter)
        x0 = [fx, fy, cx, cy, rot_x, rot_y, rot_z, tx, ty, tz,
                 res.x[0], res.x[1], res.x[2], res.x[3], res.x[4]]

        res = opt.least_squares(self._loss_full, x0,
                                 method = 'lm',
                                 max_nfev = self._max_iter)
        cameraparams = {}
        cameraparams['f'] = [res.x[0], res.x[1]]
        cameraparams['c'] = [res.x[2], res.x[3]]
        cameraparams['theta'] = [res.x[4], res.x[5], res.x[6]]
        cameraparams['t'] = [res.x[7], res.x[8], res.x[9]]
        cameraparams['k'] = [res.x[10], res.x[11], res.x[12]]
        cameraparams['p'] = [res.x[13], res.x[14]]

        return cameraparams, res
