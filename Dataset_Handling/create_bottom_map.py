"""
Created on Nov 18 13:31

@author: ISAC - pettirsch
"""

import argparse
import os

import numpy as np
import xml.etree.ElementTree as ET



from Utils.File_Folder_Handling.dataset_folder_utils import getFolderList
from Utils.PerspectiveTransform.perspectiveTransform import PerspectiveTransform

parser = argparse.ArgumentParser(description='Create a dataset in yolo format from a given dataset in the ISAC format')
parser.add_argument("--xmlDatasetPath", help="Path to current dataset",
                    default="")
parser.add_argument("--yoloDatasetPath", help="Path to new dataset",
                    default="")
parser.add_argument("--surfaceModel", help="SF Model to create bottom maps",
                    default="homography_sparse") #triangulation_all, triangulation_sparse, homography_all, homography_sparse
parser.add_argument("--newSurfaceFolder", help="Path to new surface folder",
                    default="")
args = parser.parse_args()


def createBottomMaps(xmlDatasetPath, yoloDatasetPath, surfaceModel="triangulation", newSurfaceFolder=None):
    # Get XML folder list
    train_folder_list, val_folder_list, test_folder_list = getFolderList(xmlDatasetPath)
    split_folder_list = [train_folder_list, val_folder_list, test_folder_list]
    splits = ["train", "val", "test"]

    # Create Perspective Transform
    perspectiveTransform = PerspectiveTransform()

    matching_objects_xml = 0
    not_matching_objects_xml = 0
    matching_objects_txt = 0
    not_matching_objects_txt = 0
    missing_txt = 0

    # Iterate through all splits
    for i, split in enumerate(splits):
        # Create folder for the bottom_maps
        if surfaceModel == "triangulation_sparse":
            bottom_map_folder = os.path.join(yoloDatasetPath, "bottom_maps_tria_sparse", split)
        elif surfaceModel == "triangulation_all":
            bottom_map_folder = os.path.join(yoloDatasetPath, "bottom_maps", split)
        elif surfaceModel == "homography_all":
            bottom_map_folder = os.path.join(yoloDatasetPath, "bottom_maps_homography_all", split)
        elif surfaceModel == "homography_sparse":
            bottom_map_folder = os.path.join(yoloDatasetPath, "bottom_maps_homography_sparse", split)
        if not os.path.exists(bottom_map_folder):
            os.makedirs(bottom_map_folder)

        # Iterate through all folders in the split
        for folder in split_folder_list[i]:
            print(f"Creating bottom maps for {folder}")

            # Change XML_Files to Calibration in the folder path and remove the last subfolder
            calib_folder = folder.replace("XML_Files", "Calibration")
            calib_folder = os.path.dirname(calib_folder)

            if surfaceModel == "triangulation_all":
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
            elif surfaceModel == "triangulation_sparse":
                for file in os.listdir(calib_folder):
                    if "calibrationMatrix" in file:
                        calibration_file = os.path.join(calib_folder, file)
                assert calibration_file is not None, "Calibration file not found"

                sensor_id = calib_folder.split("/")[-3]
                surface_folder = os.path.join(newSurfaceFolder, sensor_id)
                for file in os.listdir(surface_folder):
                    if "roadsurfaceTriangulationFaces" in file:
                        roadsurfaceTriangulationFaces_file = os.path.join(surface_folder, file)
                    elif "roadsurfaceTriangulationPoints" in file:
                        roadsurfaceTriangulationPoints_file = os.path.join(surface_folder, file)
                assert roadsurfaceTriangulationFaces_file is not None, "Roadsurface Triangulation Faces file not found"
                assert roadsurfaceTriangulationPoints_file is not None, "Roadsurface Triangulation Points file not found"

                # Update perspective transform
                perspectiveTransform.updateCalibration(calibrationPath=calibration_file,
                                                       triangulationFacesPath=roadsurfaceTriangulationFaces_file,
                                                       triangulationPointsPath=roadsurfaceTriangulationPoints_file,
                                                       calibration_type="Delaunay triangulation")


            elif surfaceModel == "homography_all":
                for file in os.listdir(calib_folder):
                    if "calibrationMatrix" in file:
                        calibration_file = os.path.join(calib_folder, file)
                assert calibration_file is not None, "Calibration file not found"
                sensor_id = calib_folder.split("/")[-3]
                surface_folder = os.path.join(newSurfaceFolder, sensor_id)
                for file in os.listdir(surface_folder):
                    if "homography" in file and not "sparse" in file:
                        homography_file = os.path.join(surface_folder, file)
                assert homography_file is not None, "Homography file not found"

                perspectiveTransform.updateCalibration(calibrationPath=calibration_file,
                                                       homographyPath=homography_file,
                                                       calibration_type="Homography")
            elif surfaceModel == "homography_sparse":
                for file in os.listdir(calib_folder):
                    if "calibrationMatrix" in file:
                        calibration_file = os.path.join(calib_folder, file)
                assert calibration_file is not None, "Calibration file not found"
                sensor_id = calib_folder.split("/")[-3]
                surface_folder = os.path.join(newSurfaceFolder, sensor_id)
                for file in os.listdir(surface_folder):
                    if "homography" in file and "sparse" in file:
                        homography_file = os.path.join(surface_folder, file)
                assert homography_file is not None, "Homography file not found"

                perspectiveTransform.updateCalibration(calibrationPath=calibration_file,
                                                       homographyPath=homography_file,
                                                       calibration_type="Homography")

            # Get Bottom Map
            bottom_map = perspectiveTransform.createBottomMap()

            # campos
            camera_pos = perspectiveTransform.getCameraPosition()

            # Get all XML files in the folder
            xml_files = [f for f in os.listdir(folder) if f.endswith(".xml")]

            # Iterate through all XML files

            for xml_file in xml_files:
                # Check XML file
                xml_file_path = os.path.join(folder, xml_file)
                tree = ET.parse(xml_file_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    # Bottom center image
                    bc_image = obj.find("bottom_center_image").text
                    bc_image = np.asarray(bc_image.split(" ")).astype(float)
                    bc_image[0] = min(max(bc_image[0], 0), 639)
                    bc_image[1] = min(max(bc_image[1], 0), 479)

                    # Bottom center world
                    bc_world = obj.find("bottom_center_world").text
                    bc_world = np.asarray(bc_world.split(" ")).astype(float)

                    # check bc world
                    bc_world_test = bottom_map[int(bc_image[0]), int(bc_image[1])] + camera_pos

                    diff = np.linalg.norm(bc_world_test[:2] - bc_world[:2])
                    if diff > 0.1:
                        print(f"World coordinates do not match: {diff}")
                        not_matching_objects_xml += 1
                    else:
                        matching_objects_xml += 1

                # Check txt file
                txt_file_name = xml_file.replace(".xml", ".txt")
                txt_file_path = os.path.join(yoloDatasetPath, "labels", split, txt_file_name)
                # Check if the txt file exists
                if not os.path.exists(txt_file_path):
                    print(f"Missing txt file: {txt_file_path}")
                    missing_txt += 1
                    continue

                with open(txt_file_path, "r") as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.split(" ")
                        bc_image = np.array([int(round(float(line[5])*640,0)), int(round(float(line[6])*480,0))])
                        # clip bc_image[0] and bc_image[1] to the image size
                        bc_image[0] = min(max(bc_image[0], 0), 639)
                        bc_image[1] = min(max(bc_image[1], 0), 479)

                        bc_world = np.array([float(line[7]), float(line[8]), float(line[9])])

                        bc_world_test = bottom_map[bc_image[0], bc_image[1]] + camera_pos
                        diff = np.linalg.norm(bc_world_test[:2] - bc_world[:2])
                        if diff > 0.1:
                            print(f"World coordinates do not match: {diff}")
                            not_matching_objects_txt += 1
                        else:
                            matching_objects_txt += 1

                # Get the name of the bottom map file
                bottom_map_file_name = xml_file.replace(".xml", ".npy")

                # Create the bottom map file path
                bottom_map_file_path = os.path.join(bottom_map_folder, bottom_map_file_name)

                # Save the bottom map
                np.save(bottom_map_file_path, bottom_map)

            print(f"Matching objects xml: {matching_objects_xml}")
            print(f"Not matching objects xml: {not_matching_objects_xml}")
            print(f"Quotient xml: {matching_objects_xml / (matching_objects_xml + not_matching_objects_xml+1e-6)}")

            print(f"Matching objects txt: {matching_objects_txt}")
            print(f"Not matching objects txt: {not_matching_objects_txt}")
            print(f"Quotient txt: {matching_objects_txt / (matching_objects_txt + not_matching_objects_txt+1e-6)}")

            print(f"Missing txt: {missing_txt}")

if __name__ == '__main__':
    createBottomMaps(args.xmlDatasetPath, args.yoloDatasetPath, args.surfaceModel, args.newSurfaceFolder)
