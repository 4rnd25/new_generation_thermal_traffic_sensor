"""
Created on Sep 17 2024 09:22

@author: ISAC - pettirsch
"""

import argparse
import os
import xml.etree.ElementTree as ET
import numpy as np

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--datasetFolder', type=str,
                    default='',
                    help='Mother Folder Dataset')
parser.add_argument('--OutputPath', type=str,
                    default='',
                    help='Output Path to save the mean sizes')
parser.add_argument('--split', type=str, default='All', help='Split to evaluate') #Train, Val, Test, All
opt = parser.parse_args()


def evaluate_3d_dataset(dataset_folder):
    dataset_dist_dict = {"Motorcycle": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                        "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                        "Test": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}},
                         "Car": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                 "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                 "Test": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}},
                         "Truck": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                   "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                   "Test": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}},
                         "Bus": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                 "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                 "Test": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}},
                         "Person": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                    "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                    "Test": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}},
                         "Bicycle": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                     "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                     "Test": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}},
                         "E-Scooter": {"Train": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                       "Val": {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0},
                                       "Test": {"0m - 30m": 0, "30m - 50m": 0,
                                                "> 50m": 0}}}  # Dictionary to store the distribution of the classes in the dataset

    sizes_dict = {"Motorcycle": {"Length": [], "Width": [], "Height": []},
                  "Car": {"Length": [], "Width": [], "Height": []},
                  "Truck": {"Length": [], "Width": [], "Height": []},
                  "Bus": {"Length": [], "Width": [], "Height": []},
                  "Person": {"Length": [], "Width": [], "Height": []},
                  "Bicycle": {"Length": [], "Width": [], "Height": []},
                  "E-Scooter": {"Length": [], "Width": [],
                                "Height": []}}  # Dictionary to store the sizes of the objects in the dataset

    location_dict = {"71": {"Train": 0, "Val": 0, "Test": 0, "Boxes":0},
                     "88": {"Train": 0, "Val": 0, "Test": 0, "Boxes":0},
                     "12002": {"Train": 0, "Val": 0, "Test": 0, "Boxes":0},
                     "12004": {"Train": 0, "Val": 0, "Test": 0, "Boxes":0}}  # Dictionary to store the distribution of the classes in the dataset

    evaluation_distances = ["0m - 30m", "30m - 50m", "> 50m"]  # List of the evaluation distances

    dict_all_distances = {"0m - 30m": 0, "30m - 50m": 0, "> 50m": 0}

    # Iterate over all folders in the dataset folder
    for measurement_folder in os.listdir(dataset_folder):
        measurement_folder_path = os.path.join(dataset_folder, measurement_folder)
        # Check if the folder is a directory
        if os.path.isdir(measurement_folder_path):
            for sensor_folder in os.listdir(measurement_folder_path):
                sensor_folder_path = os.path.join(measurement_folder_path, sensor_folder)
                # Check if the folder is a directory
                if os.path.isdir(sensor_folder_path):
                    for record_folder in os.listdir(sensor_folder_path):
                        print(f"Processing measurement folder: {measurement_folder}, sensor folder: {sensor_folder}, "
                                f"record folder: {record_folder}")
                        record_folder_path = os.path.join(sensor_folder_path, record_folder)
                        # Check if the folder is a directory
                        if os.path.isdir(record_folder_path):
                            xml_files_path = os.path.join(record_folder_path, "XML_Files")
                            try:
                                assert os.path.exists(xml_files_path), f"Path does not exist: {xml_files_path}"
                            except:
                                continue

                            for split_folder in os.listdir(xml_files_path):
                                if opt.split != "All" and not opt.split in split_folder:
                                    continue

                                split_folder_path = os.path.join(xml_files_path, split_folder)
                                # Check if the folder is a directory
                                if os.path.isdir(split_folder_path):
                                    for xml_file in os.listdir(split_folder_path):
                                        xml_file_path = os.path.join(split_folder_path, xml_file)
                                        # Check if the file is a file
                                        if os.path.isfile(xml_file_path) and xml_file.endswith(".xml"):
                                            # Read the xml file
                                            classes, distances, lengths, widths, heights = evaluate_xml_file(
                                                xml_file_path)

                                            # Add one image to the location dict
                                            location_dict[str(sensor_folder)][str(split_folder)] += 1

                                            for i, object_class in enumerate(classes):
                                                if object_class in dataset_dist_dict.keys():
                                                    if distances[i] <= 30:
                                                        dataset_dist_dict[object_class][split_folder]["0m - 30m"] += 1
                                                        dict_all_distances["0m - 30m"] += 1
                                                    elif distances[i] <= 50:
                                                        dataset_dist_dict[object_class][split_folder]["30m - 50m"] += 1
                                                        dict_all_distances["30m - 50m"] += 1
                                                    else:
                                                        dataset_dist_dict[object_class][split_folder]["> 50m"] += 1
                                                        dict_all_distances["> 50m"] += 1
                                                else:
                                                    print(f"Class {object_class} not in dataset_dist_dict")
                                                    raise ValueError
                                                sizes_dict[object_class]["Length"].append(lengths[i])
                                                sizes_dict[object_class]["Width"].append(widths[i])
                                                sizes_dict[object_class]["Height"].append(heights[i])

                                                # Add one box to the location dict
                                                location_dict[str(sensor_folder)]["Boxes"] += 1

    # Calculate mean sizes per class
    mean_dict = {}
    for object_class in sizes_dict.keys():
        mean_length = np.mean(sizes_dict[object_class]["Length"])
        mean_width = np.mean(sizes_dict[object_class]["Width"])
        mean_height = np.mean(sizes_dict[object_class]["Height"])
        mean_dict[object_class] = {"Length": mean_length, "Width": mean_width, "Height": mean_height}

    # Check if the output path exists
    if not os.path.exists(opt.OutputPath):
        os.makedirs(opt.OutputPath)

    # Save the mean sizes to a file in opt.OutputPath
    with open(os.path.join(opt.OutputPath, "mean_sizes.txt"), "w") as file:
        for object_class in mean_dict.keys():
            file.write(f"{object_class}: {mean_dict[object_class]}\n")

    # Save the location of the boxes to a file in opt.OutputPath
    with open(os.path.join(opt.OutputPath, "location_boxes.txt"), "w") as file:
        for sensor in location_dict.keys():
            file.write(f"{sensor}:\n")
            for split in location_dict[sensor].keys():
                file.write(f"\t{split}: {location_dict[sensor][split]}\n")

    # Save the dict_all_distances to a file in opt.OutputPath
    with open(os.path.join(opt.OutputPath, "dict_all_distances.txt"), "w") as file:
        for distance in dict_all_distances.keys():
            file.write(f"{distance}: {dict_all_distances[distance]}\n")
            

    # Save the distribution of the classes to a file in opt.OutputPath
    with open(os.path.join(opt.OutputPath, "class_distribution.txt"), "w") as file:
        for object_class in dataset_dist_dict.keys():
            file.write(f"{object_class}:\n")
            for split in dataset_dist_dict[object_class].keys():
                file.write(f"\t{split}:\n")
                for distance in dataset_dist_dict[object_class][split].keys():
                    file.write(f"\t\t{distance}: {dataset_dist_dict[object_class][split][distance]}\n")

    # Save the overall number of boxes per class, split and in general to a file in opt.OutputPath
    numbers_per_class = {"Motorcycle": 0, "Car": 0, "Truck": 0, "Bus": 0, "Person": 0, "Bicycle": 0, "E-Scooter": 0,
                         "All": 0}
    numbers_per_split = {"Train": 0, "Val": 0, "Test": 0}
    for object_class in dataset_dist_dict.keys():
        for split in dataset_dist_dict[object_class].keys():
            total = sum(dataset_dist_dict[object_class][split].values())
            numbers_per_class[object_class] += total
            numbers_per_split[split] += total
            numbers_per_class["All"] += total
    with open(os.path.join(opt.OutputPath, "overall_number_of_boxes.txt"), "w") as file:
        for object_class in numbers_per_class.keys():
            file.write(f"{object_class}: {numbers_per_class[object_class]}\n")
        for split in numbers_per_split.keys():
            file.write(f"{split}: {numbers_per_split[split]}\n")

    # Plot the distribution of the classes
    alphas = {"0m - 30m": 1, "30m - 50m": 0.75, "> 50m": 0.5}
    plt.figure(figsize=(20, 16))  # Increase figure height
    # plt.subplots_adjust(top=0.85)
    plt.rcParams.update({'font.size': 24, 'font.family': 'Times New Roman'})

    # Extracting class names
    classes = list(dataset_dist_dict.keys())

    # Number of classes
    num_classes = len(classes)

    # Bar width
    cls_bar_width = 0.9
    bar_width = cls_bar_width / 3

    # X positions for the bars
    x = np.arange(num_classes)
    x = x + 1

    split_colors = [(218 / 255, 227 / 255, 243 / 255), (226 / 255, 240 / 255, 217 / 255),
                    (251 / 255, 229 / 255, 214 / 255), (237 / 255, 206 / 255, 204 / 255)]

    # Plotting the bars
    overall_max = 0
    for i, (key, val) in enumerate(dataset_dist_dict.items()):
        x_pos_cls = x[i]
        height_bar_save = []
        for j, split in enumerate(val.keys()):
            x_pos_split = x_pos_cls + (-1 + j) * bar_width
            for d, distance in enumerate(val[split].keys()):
                values = [val[split]["0m - 30m"], val[split]["30m - 50m"], val[split]["> 50m"]]
                total_value = sum(values)
                height_bar = total_value

                heights_diff = [val[split]["0m - 30m"], val[split]["30m - 50m"], val[split]["> 50m"]]

            for q, height in enumerate(heights_diff):
                if q == 0:
                    bottom = 0
                else:
                    bottom = np.sum(heights_diff[:q], axis=0)
                plt.bar(x_pos_split, height, bar_width, bottom=bottom,
                        label=((split + " " + evaluation_distances[q]) if i == 0 else None),
                        color=split_colors[j],
                        alpha=alphas[evaluation_distances[q]],
                        edgecolor='black', linewidth=1.5)
            height_bar_save.append(height_bar)

        max_height = max(height_bar_save)
        overall_max = max(overall_max, max_height)
        for w, height_bar in enumerate(height_bar_save):
            x_pos_split = x_pos_cls + (-1 + w) * bar_width
            vertical_offset = max_height * 0.05
            plt.text(x_pos_split, (round(max_height, 2)) + 500, str(round(height_bar, 2)),  # + "%",
                     ha='center', va='bottom', rotation=90)

    plt.xticks(x, classes)

    # Set y lim to be 0 to max value + 1000
    plt.ylim(0, overall_max + 3000)

    # Add legend and labels
    plt.xlabel("Classes")
    plt.ylabel("Number of 3D Boxes")
    plt.title("Distribution of Classes in the Dataset")
    plt.legend()

    plt.savefig(os.path.join(opt.OutputPath, "class_distribution.png"))


def evaluate_xml_file(xml_file_path):
    classes = []
    distances = []
    lengths = []
    widths = []
    heights = []

    # Read the xml file
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    for object in root.findall("object"):
        # Extrac the obj position
        obj_position = object.find("center_3d_camera").text
        obj_position = np.asarray(obj_position.split(" ")).astype(float)
        obj_position_x = obj_position[0]
        obj_position_y = obj_position[1]
        obj_position_z = obj_position[2]

        posiotion_array = np.array([obj_position_x, obj_position_y, obj_position_z])
        distance = np.linalg.norm(posiotion_array)

        # Extract the class
        obj_class = object.find("name").text

        # Set all letters to lowercase and set the first letter to uppercase
        obj_class = obj_class.lower()
        if "-" in obj_class:
            obj_class_split = obj_class.split("-")
            for i, part in enumerate(obj_class_split):
                obj_class_split[i] = part.capitalize()
            obj_class = obj_class_split[0] + "-" + obj_class_split[1]
        else:
            obj_class = obj_class.capitalize()

        # Extract the size
        obj_length = float(object.find("length").text)
        obj_width = float(object.find("width").text)
        obj_height = float(object.find("height").text)

        classes.append(obj_class)
        distances.append(distance)
        lengths.append(obj_length)
        widths.append(obj_width)
        heights.append(obj_height)

    return classes, distances, lengths, widths, heights


if __name__ == "__main__":
    evaluate_3d_dataset(dataset_folder=opt.datasetFolder)
