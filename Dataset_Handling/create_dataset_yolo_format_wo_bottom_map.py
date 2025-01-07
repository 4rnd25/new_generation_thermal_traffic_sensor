"""
Created on Apr 19 13:04

@author: ISAC - pettirsch
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np

from Utils.File_Folder_Handling.dataset_folder_utils import getFolderList


parser = argparse.ArgumentParser(description='Create a dataset in yolo format from a given dataset in the ISAC format')
parser.add_argument("--xmlDatasetPath", help="Path to current dataset",
                    default="")
parser.add_argument("--yoloDatasetPath", help="Path to new dataset",
                    default="")
args = parser.parse_args()

class_name_to_id_mapping = {"motorcycle": 0, "car": 1, "truck": 2, "bus": 3, "person": 4,
                            "bicycle": 5, "e-scooter": 6}


def parseXMLFile(xmlfile):

    # Initialize annotation dictionary
    annotation_dict = {}

    # Parse XML file
    tree = ET.parse(xmlfile)
    root = tree.getroot()

    # Get image path
    image_path = root.find("filename").text
    annotation_dict["image_path"] = image_path

    # Get image dimensions
    height = int(root.find("size").find("height").text)
    width = int(root.find("size").find("width").text)
    annotation_dict["img_height"] = height
    annotation_dict["img_width"] = width

    boxes = []
    labels = []
    bottom_centers_img = []
    bottom_centers_world = []
    lengths = []
    widths = []
    heights = []
    obs_angles = []
    yaw_angles = []

    for obj in root.findall("object"):

        # Label
        label = obj.find("name").text
        labels.append(label)

        # Bounding Box
        ymin = float(obj.find('bndbox').find('ymin').text)
        xmin = float(obj.find('bndbox').find('xmin').text)
        ymax = float(obj.find('bndbox').find('ymax').text)
        xmax = float(obj.find('bndbox').find('xmax').text)
        box = [xmin, ymin, xmax, ymax]
        boxes.append(box)

        # Bottom center image
        bc_image = obj.find("bottom_center_image").text
        bc_image = np.asarray(bc_image.split(" ")).astype(float)
        bottom_centers_img.append(bc_image)

        # Bottom center world
        bc_world = obj.find("bottom_center_world").text
        bc_world = np.asarray(bc_world.split(" ")).astype(float)
        bottom_centers_world.append(bc_world)

        # Length
        length = float(obj.find("length").text)
        lengths.append(length)

        # Width
        width = float(obj.find("width").text)
        widths.append(width)

        # Height
        height = float(obj.find("height").text)
        heights.append(height)

        # Observation angle
        obs_angle = float(obj.find("obs_angle_glob").text)
        obs_angles.append(obs_angle)

        # Yaw angle
        yaw_angle = float(obj.find("yaw_angle_glob").text)
        yaw_angles.append(yaw_angle)

    # Add all lists to the annotation dictionary
    annotation_dict["labels"] = labels
    annotation_dict["boxes"] = boxes
    annotation_dict["bottom_centers_img"] = bottom_centers_img
    annotation_dict["bottom_centers_world"] = bottom_centers_world
    annotation_dict["lengths"] = lengths
    annotation_dict["widths"] = widths
    annotation_dict["heights"] = heights
    annotation_dict["obs_angles"] = obs_angles
    annotation_dict["yaw_angles"] = yaw_angles

    return annotation_dict

def create_yolo_txt_file(annotation_dict,newfolderTXT, file):

    print_buffer = []
    for i, box in enumerate(annotation_dict["boxes"]):

        # Class ID
        class_id = class_name_to_id_mapping[annotation_dict["labels"][i]]

        # Bounding Box 2D
        xmin = box[0]
        ymin = box[1]
        xmax = box[2]
        ymax = box[3]
        xmin = max(0.0, min(round(xmin / annotation_dict["img_width"], 3), 1.0))
        ymin = max(0.0, min(round(ymin / annotation_dict["img_height"], 3), 1.0))
        xmax = max(0.0, min(round(xmax / annotation_dict["img_width"], 3), 1.0))
        ymax = max(0.0, min(round(ymax / annotation_dict["img_height"], 3), 1.0))
        x_center = round((xmin + xmax) / 2, 3)
        y_center = round((ymin + ymax) / 2, 3)
        box_width = round(xmax - xmin, 3)
        box_height = round(ymax - ymin, 3)

        # Bottom Center Image
        bc_img_x = max(0.0,
                       min(round(annotation_dict["bottom_centers_img"][i][0] / annotation_dict["img_width"], 3), 1.0))
        bc_img_y = max(0.0,
                       min(round(annotation_dict["bottom_centers_img"][i][1] / annotation_dict["img_height"], 3), 1.0))

        # Bottom Center World
        bc_world_x = round(annotation_dict["bottom_centers_world"][i][0], 3)
        bc_world_y = round(annotation_dict["bottom_centers_world"][i][1], 3)
        bc_world_z = round(annotation_dict["bottom_centers_world"][i][2], 3)

        # Length
        length = round(annotation_dict["lengths"][i], 3)

        # Width
        width = round(annotation_dict["widths"][i], 3)

        # Height
        height = round(annotation_dict["heights"][i], 3)

        # Observation Angle
        obs_angle = round(annotation_dict["obs_angles"][i], 3)
        if obs_angle < 0:
            obs_angle = obs_angle + 2 * np.pi
        if obs_angle > 2 * np.pi:
            obs_angle = obs_angle - 2 * np.pi

        # Yaw Angle
        yaw_angle = round(annotation_dict["yaw_angles"][i], 3)
        if yaw_angle < 0:
            yaw_angle = yaw_angle + 2 * np.pi
        if yaw_angle > 2 * np.pi:
            yaw_angle = yaw_angle - 2 * np.pi

        # Debug
        dims = np.array([length, width, height])
        half_dim = dims / 2
        corners_Loc_bc = np.array([
            [1, -1, 0],  # front-right-bottom
            [1, 1, 0],  # front-left-bottom
            [-1, 1, 0],  # back-left-bottom
            [-1, -1, 0],  # back-right-bottom
            [1, -1, 1],  # front-right-top
            [1, 1, 1],  # front-left-top
            [-1, 1, 1],  # back-left-top
            [-1, -1, 1]  # back-right-top
        ], dtype=np.float) # Shape: (1, 8, 3)

        # create array shape n,8,3 with n = 1
        corners_Loc_bc = corners_Loc_bc[np.newaxis, :, :]
        half_dim = half_dim[np.newaxis, :]

        # Scale the local corners by dimensions
        scaled_corners = corners_Loc_bc * half_dim  # Shape: (n, 8, 3)

        # Compute rotation matrices
        cos_yaw = np.cos(yaw_angle)
        sin_yaw = np.sin(yaw_angle)
        rotation_matrices = np.stack([
            np.stack([cos_yaw, -sin_yaw, np.zeros_like(yaw_angle)]),
            np.stack([sin_yaw, cos_yaw, np.zeros_like(yaw_angle)]),
            np.stack([np.zeros_like(yaw_angle), np.zeros_like(yaw_angle), np.ones_like(yaw_angle)])
        ])  # Shape: (n, 3, 3)

        # Rotate local corners
        rotated_corners = np.matmul(scaled_corners, rotation_matrices)  # Shape: (n, 8, 3)

        # Translate to world coordinates
        bc_3d = np.array([bc_world_x, bc_world_y, bc_world_z])
        corners_World_bc = rotated_corners + bc_3d  # Shape: (n, 8, 3)

        bottom_corners_world = np.mean(corners_World_bc[:, 0:4, :], axis=1)

        test_diff = np.linalg.norm(bottom_corners_world - bc_3d)

        assert test_diff < 1e-6, "Error in 3D corner calculation"

        # Append to print buffer
        print_buffer.append(
            "{} {:.3} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f} {:.3f}".format(
                class_id, x_center, y_center, box_width, box_height, bc_img_x, bc_img_y, bc_world_x, bc_world_y,
                bc_world_z, length, width, height, obs_angle, yaw_angle))

    # Write print buffer to txt file
    if len(print_buffer) > 0:
        print("\n".join(print_buffer), file=open(os.path.join(newfolderTXT, file.replace(".xml", ".txt")), "w"))

    return len(print_buffer)

def createYoloDataset(xmlDatasetPath, yoloDatasetPath):

    train_folder_list, val_folder_list, test_folder_list = getFolderList(xmlDatasetPath)
    split_folder_list = [train_folder_list, val_folder_list, test_folder_list]
    splits = ["train", "val", "test"]

    # Iterate through all folders in the dataset
    for i, split in enumerate(splits):
        # Create new folders for the new dataset
        newfolderTXT = os.path.join(yoloDatasetPath, "labels", split)
        newfolderIMG = os.path.join(yoloDatasetPath, "images", split)
        if not os.path.exists(newfolderTXT):
            os.makedirs(newfolderTXT)
        if not os.path.exists(newfolderIMG):
            os.makedirs(newfolderIMG)

        # Iterate through all folders in the split
        for folder in split_folder_list[i]:
            print("Processing folder: {}".format(folder))

            # Iterate through all files in the folder
            for file in os.listdir(folder):
                if file.endswith(".xml"):
                    annotation_dict = parseXMLFile(os.path.join(folder, file))

                    # Create yolo txt file
                    num_labels = create_yolo_txt_file(annotation_dict, newfolderTXT, file)

                    # Copy image file
                    if num_labels > 0:
                        image_folder = folder.replace("XML_Files", "Raw_Images")
                        image_file = os.path.join(image_folder, annotation_dict["image_path"])
                        cp_image_file = os.path.join(newfolderIMG, annotation_dict["image_path"])
                        os.system("cp {} {}".format(image_file, cp_image_file))

if __name__ == '__main__':
    createYoloDataset(args.xmlDatasetPath, args.yoloDatasetPath)
