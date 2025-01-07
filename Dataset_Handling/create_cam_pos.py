"""
Created on Nov 19 07:21

@author: ISAC - pettirsch
"""

import argparse
import os

import numpy as np

from Utils.File_Folder_Handling.dataset_folder_utils import getFolderList
from Utils.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

parser = argparse.ArgumentParser(description='Create a dataset in yolo format from a given dataset in the ISAC format')
parser.add_argument("--xmlDatasetPath", help="Path to current dataset",
                    default="")
parser.add_argument("--yoloDatasetPath", help="Path to new dataset",
                    default="")
args = parser.parse_args()


def createCamPos(xmlDatasetPath, yoloDatasetPath, surfaceModel="triangulation"):
    # Get XML folder list
    train_folder_list, val_folder_list, test_folder_list = getFolderList(xmlDatasetPath)
    split_folder_list = [train_folder_list, val_folder_list, test_folder_list]
    splits = ["train", "val", "test"]

    # Create Perspective Transform
    perspectiveTransform = PerspectiveTransform()

    # Iterate through all splits
    for i, split in enumerate(splits):
        # Create folder for the cam pos
        cam_pos_folder = os.path.join(yoloDatasetPath, "cam_pos", split)
        if not os.path.exists(cam_pos_folder):
            os.makedirs(cam_pos_folder)

        # Iterate through all folders in the split
        for folder in split_folder_list[i]:
            print(f"Creating cam pos for {folder}")

            # Change XML_Files to Calibration in the folder path and remove the last subfolder
            calib_folder = folder.replace("XML_Files", "Calibration")
            calib_folder = os.path.dirname(calib_folder)


            # Get all files in the folder
            for file in os.listdir(calib_folder):
                if "calibrationMatrix" in file:
                    calibration_file = os.path.join(calib_folder, file)
                elif "roadsurfaceTriangulationFaces" in file:
                    roadsurfaceTriangulationFaces_file = os.path.join(calib_folder, file)
                elif "roadsurfaceTriangulationPoints" in file:
                    roadsurfaceTriangulationPoints_file = os.path.join(calib_folder, file)
            assert calibration_file is not None, "Calibration file not found"
            assert roadsurfaceTriangulationFaces_file is not None, "Roadsurface Triangulation Faces file not found"
            assert roadsurfaceTriangulationPoints_file is not None, "Roadsurface Triangulation Points file not found"

            # Update perspective transform
            perspectiveTransform.updateCalibration(calibrationPath=calibration_file,
                                                   triangulationFacesPath=roadsurfaceTriangulationFaces_file,
                                                   triangulationPointsPath=roadsurfaceTriangulationPoints_file,
                                                   calibration_type="Delaunay triangulation")

            # Get camera position
            cam_pos = perspectiveTransform.getCameraPosition()

            # Get all XML files in the folder
            xml_files = [f for f in os.listdir(folder) if f.endswith(".xml")]

            # Iterate through all XML files
            for xml_file in xml_files:
                # Get the name of the cam pos file
                cam_pos_file_name = xml_file.replace(".xml", "_cam_pos.npy")

                # Create the cam pos file path
                cam_pos_file_path = os.path.join(cam_pos_folder, cam_pos_file_name)

                # Save the cam pos map
                np.save(cam_pos_file_path, cam_pos)

if __name__ == '__main__':
    createCamPos(args.xmlDatasetPath, args.yoloDatasetPath)
