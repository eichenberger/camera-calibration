"""
File: poc.py
Author: Stefan Eichenberger
Email: eichest@gmail.com
Github: eichenberger
Description: This is a proof of concept for point cloud base camera calibration
"""
import math
import sys
import numpy as np
from matplotlib import pyplot as plt
import multiprocessing as mp
import cv2
import logging

#from pointmodel import PointModel
from poc_app import ProofOfConcept
from cameramodel import CameraModel

def array_string(array, space):
    output = ""
    for entry in array:
        entry = str(entry)
        output = output + entry
        output = output + " "*(space - len(entry))

    return output

def run_application(args):
    poc = ProofOfConcept(args)

    image = poc.get_image()
    plt.imshow(image)

    cameraparams = poc.cameraparams
    cameraparams_est, res, cme = poc.estimate()


    print("Cost (reduced data): " + str(res.cost))
    print("Optimality, gradient value (reduced data): " + str(res.optimality))
    inliers = np.array(cme._inliers)
    points = np.array(poc.mod.points)
    print("Inliers in percent: " + str(100.0*inliers.shape[0]/points.shape[0]))

    print(array_string([" ", "fx", "fy", "cx", "cy", "thetax", "thetay", "thetaz",
                        "tx", "ty", "tz", "k1", "k2", "k3", "p1", "p2"], 8))
    print("shuld:  " + cameraparams.get_string(8))
    print("is:     " + cameraparams_est.get_string(8))

    dist = np.array(cameraparams.get_as_array()) - np.array(cameraparams_est.get_as_array())

    print("diff:   " + array_string(np.round(dist, 2), 8))
    print("Tot. diff: {}".format(np.linalg.norm(dist)))

    cm_est = CameraModel(poc.resolution, cameraparams_est)
    cm_est.update_point_cloud(poc.mod.points)
    image_est = cm_est.get_image()

    image1 = np.zeros((poc.resolution[0], poc.resolution[1], 3))
    image2 = np.zeros((poc.resolution[0], poc.resolution[1], 3))
    image1[:,:,2] = image_est
    image2[:,:,1] = image
    kernel = np.ones((3,3),np.uint8)
    diff_img = image1 + image2
    diff_img = cv2.dilate(diff_img, kernel, iterations=1)

    cv2.imwrite("diff_img.png", diff_img*255)
    plt.figure()
    plt.imshow(diff_img)
    plt.show()

# Problem: tx and fx/fy depend on each other. We can change fx/fy or tx
# for zooming. Idea: If we take two pictures, we can probably fix tx to the
# same value
def main():
    import argparse
    parser = argparse.ArgumentParser(description='This is a proof of concept application which can be used to test several camera calibration algorithms')
    parser.add_argument('-c', '--cameramodel', help='JSON file which contains the camrea model', dest='cameramodel', type=str, default='cameramodel.json')
    parser.add_argument('-n', '--noise', help='Noise in the image data', dest='noise', type=float, default=0.0)
    parser.add_argument('-m', '--missclassified', help='Points missclassified (wrong location)', dest='missclassified', type=int, default=0)
    parser.add_argument('-p', '--pointmodel', help='Point model to use', dest='pointmodel', type=str, default='model.json')
    parser.add_argument('-v', help='Verbose output', dest='verbose', action='store_true')
    parser.add_argument('--npoints', help='Number of points to use for the image', dest='npoints', type=int, default=20)

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
    else:
        logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    run_application(args)

if __name__ == '__main__':
    main()
