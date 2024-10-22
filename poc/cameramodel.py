import logging
import numpy as np
from math import sin, cos, floor, sqrt
from cameraparams import CameraParams


class CameraModel:
    """The sample camera model to use"""
    def __init__(self, resolution, cameraparams=None, logger = None):
        self._extrinsic_mat = np.mat([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        self._resolution = resolution
        if cameraparams:
            self._k = [cameraparams.k1, cameraparams.k2, cameraparams.k3]
            self._p = [cameraparams.p1, cameraparams.p2]

            self._f = [cameraparams.fx, cameraparams.fy]
            self._c = [cameraparams.cx, cameraparams.cy]
            self._update_intrinsic()
            self.create_extrinsic([cameraparams.thetax,
                                cameraparams.thetay,
                                cameraparams.thetaz],
                                [cameraparams.tx,
                                cameraparams.ty,
                                cameraparams.tz])
        else:
            self._f = [0]*2
            self._c = [0]*2
            self._k = [0]*3
            self._p = [0]*2
            self._update_intrinsic()

        self._transformation_mat = None
        if logger == None:
            self.logger = logging.getLogger("cameramodel")
            self.logger.setLevel(logging.WARNING)
        else:
            self.logger = logger


    @staticmethod
    def _rot_mat_x(angle):
        return np.mat([[1, 0, 0],
                       [0, cos(angle), -sin(angle)],
                       [0, sin(angle), cos(angle)]])

    @staticmethod
    def _rot_mat_y(angle):
        return np.mat([[cos(angle), 0, sin(angle)],
                       [0, 1, 0],
                       [-sin(angle), 0, cos(angle)]])

    @staticmethod
    def _rot_mat_z(angle):
        return np.mat([[cos(angle), -sin(angle), 0],
                       [sin(angle), cos(angle), 0],
                       [0, 0, 1]])

    def _update_intrinsic(self):
        self._intrinsic_mat = np.mat([[self._f[0], 0, self._c[0]],
                                      [0, self._f[1], self._c[1]],
                                      [0, 0, 1]])

    def set_f(self, f):
        self._f = f
        self._update_intrinsic()

    def set_c(self, c):
        self._c = c
        self._update_intrinsic()

    def create_extrinsic(self, pose, translation):
        rot_x = self._rot_mat_x(pose[0])
        rot_y = self._rot_mat_y(pose[1])
        rot_z = self._rot_mat_z(pose[2])

        rot = np.matmul(rot_x, rot_y)
        rot = np.matmul(rot, rot_z)

        translation = np.mat(translation).transpose()
        self._extrinsic_mat = np.concatenate((rot, translation), axis=1)

    def _distortion(self, point):
        newp = [0,0,0]

        # Get point coordinates from center of image
        point_center = point[0:2] - self._c
        x = point_center[0] / self._f[0]
        y = point_center[1] / self._f[1]

        # We need the radius of the point from the principal component
        r = sqrt(x**2 + y**2)

        # Radial distortion
        newp[0] = x * (self._k[0]*r**2+self._k[1]*r**4+self._k[2]*r**6)
        newp[1] = y * (self._k[0]*r**2+self._k[1]*r**4+self._k[2]*r**6)

        # Tangential distortion
        newp[0] = newp[0] + 2*self._p[0]*x*y + self._p[1]*(r**2+2*x**2)
        newp[1] = newp[1] + 2*self._p[1]*x*y + self._p[0]*(r**2+2*y**2)

        # Calculate pixels again
        newp[0] = self._f[0]*newp[0]
        newp[1] = self._f[1]*newp[1]

        point = point + newp

        return point

    def set_transfomration_mat(self, transformation_mat):
        self._transformation_mat = transformation_mat


    def update_point_cloud(self, point_cloud):
        """ Get an image of a point cloud """
        if self._transformation_mat is not None:
            cam_mat = self._transformation_mat
        else:
            self.logger.debug("Intrinsic:")
            self.logger.debug(self._intrinsic_mat)
            self.logger.debug("Extrinsic:")
            self.logger.debug(self._extrinsic_mat)
            cam_mat = np.matmul(self._intrinsic_mat, self._extrinsic_mat)
            self.logger.debug("C:")
            self.logger.debug(cam_mat)

        n = point_cloud.shape[0]
        points2d = np.asarray(np.matmul(cam_mat, np.transpose(point_cloud)))
        # normalize third dimension to 1
        points2d = np.mat(points2d.transpose())
        points2d = np.asarray(points2d/points2d[:, 2])
        self.points2d = np.array(list(map(lambda point: self._distortion(point), points2d)))

    def get_image(self):
        image = np.zeros((self._resolution[0], self._resolution[1]))
        """ Get an image of a point cloud """
        for point in self.points2d:
            if point[0] < self._resolution[0] and \
                    point[1] < self._resolution[1] and\
                    point[0] >= 0 and point[1] >= 0:
                x = floor(point[0])
                y = floor(point[1])
                image[x, y] = 1

        return image

    def add_distortion(self, k, p):
        """ Add distortion to the camera model """
        self._k = k
        self._p = p


