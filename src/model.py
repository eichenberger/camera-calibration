import io
import json
import numpy as np
import multiprocessing as mp

class Model:
    def __init__(self, filename):
        with io.open(filename, 'r') as f:
            self.pointmap = json.JSONDecoder().decode(f.read())

        points3d = list(map(lambda point: point['position'], self.pointmap))
        points3d = np.asarray(points3d)
        self.points3d = np.ones((points3d.shape[0], 4))
        self.points3d[:, 0:3] = points3d


        self.descriptors = np.array(
            list(map(lambda point: point['descriptors'], self.pointmap)),
            dtype = np.uint8)

    def _show(self):
        from mayavi import mlab
        points = self._position
        mlab.points3d(points[:,0], points[:,1], points[:,2], mode='axes',
                      color=(0,1,0), scale_mode='none', scale_factor=0.1)
        mlab.show()

    def show(self):
        plot_proc = mp.Process(target=self._show)
        plot_proc.start()
        self.plot_proc = plot_proc

    def wait(self):
        if self.plot_proc:
            self.plot_proc.join()

