# New generation thermal traffic sensor

Repository of paper: New generation thermal traffic sensor: A novel dataset and monocular 3D thermal vision framework. <br>

Link to paper:

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Ploting](#evaluation)
7. [Acknowledgements](#acknowledgements)
8. [License](#license)

## Introduction 
This is the official repository for the paper: New generation thermal traffic sensor: A novel dataset and monocular 3D thermal vision framework. As reference to this code please cite the paper.

## Installation
This code was build using python 3.11. The requirements.txt file includes all necessary packages. Additionally this repo should be part of the PYTHONPATH. Moreover it is important to clone with --recurse-submodules to get also the modified yolov7 code.


## Data
The folder Dataset_Handling contains all necessary files: <br>

	- create_dataset_yolo_format_wo_bottom_map.py transfers the images and .csv files to yolov7 format 
	- dataset_analysis.py enables to plot the class distribution and to get the mean sizes  
	- create_bottom_map.py creates the bottom maps
	- create_cam_pos.py create the Camere Position files
        
## Training
- For the general domain adaption and training on xtA the original train.py file could be used.
- For training with the teacher network and the context module: train_with_teacher_and_context_module.py
- For training with the teacher network, the context module and the remember module: train_with_teacher_context_module_and_remember_module.py
- For training with the teacher network, the context module and the remember module with previous images in val: train_with_teacher_context_module_and_remember_module_mixed_val.py
- During all experiments with yolo-tiny:  yolov7/cfg/training/yolov7-tiny_cls_7_IDetect.yaml  and yolov7/data/hyp.scratch.tiny_therm_no_aug.yaml were used with batch size 16.
- During training of the teacher: yolov7/cfg/training/yolov7_cls_7_IDetect.yaml and yolov7/data/hyp.scratch.costum_therm_no_aug.yaml were used with batch size 16.

## Evaluation
- Models can be evaluated using Evaluation/model_evaluator.py
- Pseudo labels can be evaluated using Evaluation/pseudo_label_evaluator.py or for multiple pseudo-labels: Evaluation/pseudo_label_folder_evaluator.py
- The inference time could be mesured with yolov7/detect_xx.py and for the onnx models with Evaluation/inference_time.py

## Export
- tbd

## Acknowledgements
This repositroy includes the following third-party submodules: <br>

- yolov7 located at /yolov7 -
  Repository: https://github.com/WongKinYiu/yolov7 -
  License: GPL3.0 -
  License details are available at: yolov7/LICENSE.md -
  Changed files: 
  Added files: 

## License
This repositroy is released under the GPL License (refer to the LICENSE file for details).

