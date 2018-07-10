import numpy as np
from slp_map import SlpMap
from cameramodelestimator3 import CameraModelEstimator
from cameramodel import CameraModel

def main():
    import sys
    map_file = sys.argv[1]
    keyframe = int(sys.argv[2])
    reduced = False
    if len(sys.argv) == 4:
        reduced = True if int(sys.argv[3]) > 0 else False
    slpmap = SlpMap(map_file)
    slpmap.load_map()
    slpmap.get_points_kf(keyframe)

    points2d = np.ones((len(slpmap.points2d),3))
    points2d[:,0:2] = np.array(slpmap.points2d)[:,0:2]
    points3d = np.ones((len(slpmap.points3d),4))
    points3d[:,0:3] = np.array(slpmap.points3d)

    resolution = [640, 480]

    cme = CameraModelEstimator(resolution, points2d, points3d)
    res = cme.estimate()

    intrinsic = np.asarray([[res.x[0], 0, res.x[2]],
                 [0, res.x[1], res.x[3]]])

    print("intrinsic:\n{}".format(intrinsic))
    print("thetax: {}, thetay: {}, thetaz: {}".format(res.x[4], res.x[5], res.x[6]))
    print("tx: {}, ty: {}, z: {}".format(res.x[7], res.x[8], res.x[9]))
    print("k: {}, p: {}".format(res.x[10:13], res.x[13:15]))

    cm = CameraModel(resolution)
    cm.set_c([res.x[2], res.x[3]])
    cm.set_f([res.x[0], res.x[1]])
    cm.create_extrinsic([res.x[4], res.x[5], res.x[6]],
                              [res.x[7], res.x[8], res.x[9]])
    points2d = cme._points2d_inliers
    cm.update_point_cloud(cme._points3d_inliers[:,0:3])
    points2d_diff = points2d - cm.points2d


    print("reprojection error without dist: {}".format(np.linalg.norm(points2d_diff)))

    cm.add_distortion([res.x[10], res.x[11], res.x[12]], [res.x[13], res.x[14]])
    cm.update_point_cloud(cme._points3d_inliers[:,0:3])
    points2d_diff = points2d - cm.points2d
    print("reprojection error: {}".format(np.linalg.norm(points2d_diff)))

if __name__ == '__main__':
    main()
