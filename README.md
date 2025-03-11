# New generation thermal traffic sensor

This repository contains the code related to the paper:

New Generation Thermal Traffic Sensor: A Novel Dataset and Monocular 3D Thermal Vision Framework

📄 Paper link: [DOI: 10.1016/j.knosys.2025.113334](https://doi.org/10.1016/j.knosys.2025.113334) 
📂 Dataset: 

## 📌 Citation
If you use this code, please cite the paper:

```bibtex
@Article{PETTIRSCH2025113334,
  author  = {Arnd Pettirsch and Alvaro Garcia-Hernandez},
  title   = {New generation thermal traffic sensor: A novel dataset and monocular 3D thermal vision framework},
  journal = {Knowledge-Based Systems},
  year    = {2025},
  issn    = {0950-7051},
  pages   = {113334},
  doi     = {https://doi.org/10.1016/j.knosys.2025.113334},
  url     = {https://www.sciencedirect.com/science/article/pii/S0950705125003818}
}
```

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Data](#data)
4. [Training](#training)
5. [Evaluation](#evaluation)
6. [Detect](#detection)
7. [Export](#export)
8. [Acknowledgements](#acknowledgements)
9. [License](#license)

## Introduction 
This is the official repository for the paper: New generation thermal traffic sensor: A novel dataset and monocular 3D thermal vision framework. As reference to this code please cite the paper.

## Installation
This code was build using python 3.11. The requirements.txt file includes all necessary packages. Additionally this repo should be part of the PYTHONPATH. Moreover it is important to clone with --recurse-submodules to get also the modified yolov7 code. The yolov7 folder should also be part of the Pythonpath.


## Data
The folder Dataset_Handling contains all necessary files: <br>

	- create_dataset_yolo_format_wo_bottom_map.py transfers the images and .csv files to yolov7 format 
	- dataset_analysis.py enables to plot the class distribution and to get the mean sizes
	- create_bottom_map.py creates the bottom maps
	- create_cam_pos.py create the Camere Position files
        
## Training
- Training the models can be done with the train_thermal_3d.py file.  
- During all experiments: yolov7/cfg/training/yolov7-tiny_cls_7_IDetectMon3D.yaml  and yolov7/data/hyp.scratch.tiny_therm_3d.yaml were used with batch size 16.


## Evaluation
- Models can be evaluated using yolov7/test_thermal_3d.py
- The inference time could be mesured with yolov7/detect_thermal_3d.py and for the onnx models with Evaluation/inference_time.py

## Detection
- Detection could be made with yolov7/detect_thermal_3d.py or yolov7/conflict_detect.py

## Export
- The model could be exported to onnx-format using: yolov7/export_to_onnx.py

## Acknowledgements
This repositroy includes the following third-party submodules: <br>

- yolov7 located at /yolov7 -
  Repository: https://github.com/WongKinYiu/yolov7 -
  License: GPL3.0 -
  License details are available at: yolov7/LICENSE.md -
  Changed files: models/yolo.py, utils/datasets.py, utils/loss.py, utils/plots.py utils/torch_utils.py
  Added files: export_to_onnx.py, detect_thermal_3d.py, conflict_detect.py, hyp.scratch.tiny_therm_3d.yaml, yolov7-tiny_cls_7_IDetectMon3D.yaml, train_thermal_3d.py, 

## License
This repositroy is released under the GPL License (refer to the LICENSE file for details).

