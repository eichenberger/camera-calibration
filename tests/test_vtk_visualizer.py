import vtk_visualizer
import numpy as np

m=np.mat("100,100,100; \
       200,200,200; \
       300,300,300; \
       400,400,400; \
       500,500,500; \
       600,600,600")

vtk_visualizer.plotxyz(m)

