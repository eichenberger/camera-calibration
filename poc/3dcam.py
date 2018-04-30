## License: Apache 2.0. See LICENSE file in root directory.
## Copyright(c) 2017 Intel Corporation. All Rights Reserved.

#####################################################
##              Align Depth to Color               ##
#####################################################

# First import the library
import pyrealsense2 as rs
# Import Numpy for easy array manipulation
import numpy as np
# Import OpenCV for easy image rendering
import cv2

import sys
import io
import json

import matplotlib.pyplot as plt

cv2.ocl.setUseOpenCL(False)
def main():
    outfile = sys.argv[1]
    outimage = sys.argv[2]
    # Create a pipeline
    pipeline = rs.pipeline()

    #Create a config and configure the pipeline to stream
    #  different resolutions of color and depth streams
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    profile = pipeline.start(config)

    # Getting the depth sensor's depth scale (see rs-align example for explanation)
    depth_sensor = profile.get_device().first_depth_sensor()
    laser_range = depth_sensor.get_option_range(rs.option.laser_power)
    depth_sensor.set_option(rs.option.laser_power, 4.0)
    depth_scale = depth_sensor.get_depth_scale()
    print ("Depth Scale is: ", depth_scale)

    # We will be removing the background of objects more than
    #  clipping_distance_in_meters meters away
    # clipping_distance_in_meters = 4 #1 meter
    # clipping_distance = clipping_distance_in_meters / depth_scale

    # Create an align object
    # rs.align allows us to perform alignment of depth frames to others frames
    # The "align_to" is the stream type to which we plan to align depth frames.
    align_to = rs.stream.color
    align = rs.align(align_to)

    # Not calibrated yet!!!!!
    camera_matrix = [[463.990, 0, 320],[0, 463.889, 240], [0, 0, 1]]
    camera_matrix_inv = np.linalg.inv(camera_matrix)

    # Streaming loop
    try:
        while True:
            # Get frameset of color and depth
            frames = pipeline.wait_for_frames()
            # frames.get_depth_frame() is a 640x360 depth image

            # Align the depth frame to color frame
            aligned_frames = align.process(frames)

            # Get aligned frames
            aligned_depth_frame = aligned_frames.get_depth_frame() # aligned_depth_frame is a 640x480 depth image
            color_frame = aligned_frames.get_color_frame()

            # Validate that both frames are valid
            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            gray_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2GRAY)
            orb = cv2.ORB_create(200, 1.4, 8, fastThreshold=20)
            kp = orb.detect(gray_image, None)
            kps, image_descriptors = orb.compute(gray_image, kp)

            keypoints = []
            keypoints_for_image = []
            for i, kp in enumerate(kps):
                x = int(kp.pt[0])
                y = int(kp.pt[1])
                if depth_image[y, x] > 0:
                    world = np.asarray(np.matmul(camera_matrix_inv, [[x],[y],[1]]))
                    world[2] = float(depth_scale*depth_image[y, x])
                    keypoints.append({
                        'position': np.transpose(world[:,0]).tolist(),
                        'descriptors': image_descriptors[i].tolist()})
                    keypoints_for_image.append(kp)


            test_image = color_image.copy()
            test_image = cv2.drawKeypoints(color_image, keypoints_for_image, test_image)

            # Render images
    #        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    #        images = np.hstack((color_image, depth_colormap))
            plt.imshow(test_image)
            plt.show()

            save = input("Save [y/n]:")
            if save == 'y':
                with io.open(outfile, 'w') as f:
                    f.write(json.JSONEncoder().encode(keypoints))
                cv2.imwrite(outimage, color_image)

    finally:
        pipeline.stop()


if __name__ == '__main__':
    main()
