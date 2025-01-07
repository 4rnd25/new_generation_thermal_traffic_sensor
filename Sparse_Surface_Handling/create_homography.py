"""
Created on Sep 10 08:38

@author: ISAC - pettirsch
"""

import argparse
import cv2
import numpy as np
import os

from Utils.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

import matplotlib
matplotlib.use('TkAgg')

parser = argparse.ArgumentParser()
parser.add_argument('--Faces', type=str,
                    default='/roadsurfaceTriangulationFaces.csv',
                    help='json config path')
parser.add_argument('--Points', type=str,
                    default='/roadsurfaceTriangulationPoints.csv',
                    help='json config path')
parser.add_argument('--calibrationPath', type=str,
                    default='/calibrationMatrix.xml',
                    help='json config path')
parser.add_argument('--outputpath',
                    default='',
                    help='save to project/name')
opt = parser.parse_args()

def createHomography(calibration_file, roadsurfaceTriangulationFaces_file, roadsurfaceTriangulationPoints_file,
                     remove_threshold=150, output_path=None):

    # Create Perspective Transform
    perspectiveTransform = PerspectiveTransform()

    # Get all files in the folder
    assert calibration_file is not None, "Calibration file not found"
    assert roadsurfaceTriangulationFaces_file is not None, "Roadsurface Triangulation Faces file not found"
    assert roadsurfaceTriangulationPoints_file is not None, "Roadsurface Triangulation Points file not found"

    # Update perspective transform
    perspectiveTransform.updateCalibration(calibrationPath=calibration_file,
                                           triangulationFacesPath=roadsurfaceTriangulationFaces_file,
                                           triangulationPointsPath=roadsurfaceTriangulationPoints_file,
                                           calibration_type="Delaunay triangulation")

    # Load points
    all_points =  np.loadtxt(roadsurfaceTriangulationPoints_file, delimiter=',')

    # Get the cameraposition
    cameraPosition = perspectiveTransform.getCameraPosition()

    # For all points calculate the distance to the camera without loop and save the indices with distance < remove_threshold
    distances = np.linalg.norm(all_points - cameraPosition, axis=1)
    indices = np.where(distances < remove_threshold)[0]

    # Get only the valid points
    valid_points = all_points[indices]

    # Project the valid points to the image
    projected_points = perspectiveTransform.worldToPixel(valid_points)

    valid_points_flags = np.ones(len(valid_points))

    # Remove the points that are not in the image
    valid_points_flags[projected_points[:, 0] < 0] = 0
    valid_points_flags[projected_points[:, 0] > 640] = 0
    valid_points_flags[projected_points[:, 1] < 0] = 0
    valid_points_flags[projected_points[:, 1] > 480] = 0

    # Get the valid points and their projections
    valid_points = valid_points[valid_points_flags == 1]
    projected_points = projected_points[valid_points_flags == 1]

    # Undistort the points
    projected_points_undist = cv2.undistortPoints(projected_points, perspectiveTransform.cameraMat, perspectiveTransform.distortionCoefficients, None,
                                                     perspectiveTransform.cameraMat)
    # projected_points_undist = projected_points_undist[:,0,:]

    valid_points_undist_flags = np.ones(len(valid_points))
    valid_points_undist_flags[projected_points_undist[:, 0, 0] < 0] = 0
    valid_points_undist_flags[projected_points_undist[:, 0, 0] > 640] = 0
    valid_points_undist_flags[projected_points_undist[:, 0, 1] < 0] = 0
    valid_points_undist_flags[projected_points_undist[:, 0, 1] > 480] = 0

    valid_points = valid_points[valid_points_undist_flags == 1]
    projected_points_undist = projected_points_undist[valid_points_undist_flags == 1]

    # Create the homography
    homography_matrix, mask = cv2.findHomography(projected_points_undist, valid_points[:,0:2], method=cv2.RANSAC)

    # Test the homography
    test_points = projected_points_undist[5,0,:]
    test_points = np.array([[[test_points[0], test_points[1]]]], dtype=np.float32)
    test_points_hom = cv2.perspectiveTransform(test_points, homography_matrix)

    diff = np.linalg.norm(test_points_hom - valid_points[5,0:2])
    print("Difference between homography and ground truth: ", diff)


    # Save the homography
    homography_file = os.path.join(output_path, "homography_sparse.csv")
    np.savetxt(homography_file, homography_matrix, delimiter=',')


if __name__ == '__main__':
    createHomography(calibration_file=opt.calibrationPath,
                     roadsurfaceTriangulationFaces_file=opt.Faces,
                     roadsurfaceTriangulationPoints_file=opt.Points,
                     output_path=opt.outputpath)
