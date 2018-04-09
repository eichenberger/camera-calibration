import numpy as np
import multiprocessing as mp

class PointModel:
    def __init__(self):
        self.points = np.loadtxt('model3.txt')

    def _show_model(self):
        from mayavi import mlab
        points = self.points
        mlab.points3d(points[:,0], points[:,1], points[:,2], mode='axes',
                      color=(0,1,0), scale_mode='none', scale_factor=0.1)
        mlab.show()

    def show_model(self):
        plot_proc = mp.Process(target=self._show_model)
        plot_proc.start()
        self.plot_proc = plot_proc

    def wait_model(self):
        if self.plot_proc:
            self.plot_proc.join()

