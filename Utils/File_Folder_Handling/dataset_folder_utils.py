"""
Created on Nov 18 13:31

@author: ISAC - pettirsch
"""

import os

def getFolderList(datasetpath, foldername="XML_Files"):
    train_folder_list = []
    val_folder_list = []
    test_folder_list = []

    # Iterate through all folder and subfolders in the dataset. If folder is named "Train", "Val" or "Test", add it to the respective list
    for root, dirs, files in os.walk(datasetpath):
        if not foldername in root:
            continue
        for dir in dirs:
            if dir == "Train":
                train_folder_list.append(os.path.join(root, dir))
            elif dir == "Val":
                val_folder_list.append(os.path.join(root, dir))
            elif dir == "Test":
                test_folder_list.append(os.path.join(root, dir))

    return train_folder_list, val_folder_list, test_folder_list