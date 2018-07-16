import math
import numpy as np

from pointmodel import PointModel
from cameramodel import CameraModel
from cameramodelestimator import CameraModelEstimator
from cameraparams import CameraParams

class ProofOfConcept:
    def __init__(self, args):
        self.mod = PointModel(args.pointmodel)
        self.mod.create_points(args.npoints)

        self._read_cameramodel(args.cameramodel)

        self.noise = args.noise
        self.missclassified = args.missclassified

    def _read_cameramodel(self, cameramodel_file):
        import io
        import json
        with io.open(cameramodel_file, 'r') as fd:
            cameramodel = json.load(fd)

        self.resolution = cameramodel['resolution']
        self.cameraparams = CameraParams(cameramodel)

    def _get_cameramodel(self, noise=True):
        params = self.cameraparams
        cameramodel = CameraModel(self.resolution, params)
        cameramodel.update_point_cloud(self.mod.points)
        points2d = cameramodel.points2d

        if noise:
            points2d = self._add_noise(points2d)
            points2d = self._do_missclassifcation(points2d)
            cameramodel.points2d = points2d
        return cameramodel


    def get_image(self, noise=True):
        cameramodel = self._get_cameramodel(noise)
        return cameramodel.get_image()

    def _add_noise(self, points2d):
        points2d[:,0:2] = points2d[:,0:2] + np.random.randn(len(points2d), 2)*self.noise
        return points2d

    def _do_missclassifcation(self, points2d):
        # Do missclassifciation with random skip
        add = math.floor(np.random.rand()*50)
        i = add
        # Do some missmatching
        for j in range(0, self.missclassified):
            i = (i + add) % len(points2d)
            # random new position
            new_x = math.floor(np.random.rand()*self.resolution[0])
            new_y = math.floor(np.random.rand()*self.resolution[1])
            points2d[i,0:2] = [new_x, new_y]
        return points2d

    def estimate(self):
        cameramodel = self._get_cameramodel(True)
        cme = CameraModelEstimator(self.resolution, cameramodel.points2d,
                                   self.mod.points)
        cameraparams, res = cme.estimate()
        # convert it to more common if
        cameraparams = CameraParams(cameraparams)
        return cameraparams, res, cme

