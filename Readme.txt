Project Overview
This project includes implementations of several object detection models, with all necessary dependencies and datasets already provided. Please ensure that the file paths are correct when running the notebooks. The results of various metrics are saved in a separate folder.
Directory Structure
The following is the directory structure of the project, with the trained models organized in their respective named folders:
Source_files/
│
├── Code_for_the_models/
│      ├── Detection_code_for_all_model.ipynb
│      ├── DETR.ipynb
│      ├── FasterRCNN.ipynb
│      └── yolov5.ipynb
│
├── Dataset_DETR/
│      ├── README.dataset.txt
│      ├── README.roboflow.txt
│      ├── test/
│      ├── train/
│      └── valid/
│
├── Dataset_FasterRcnn/
│      ├── test_ann/
│      ├── test_images/
│      ├── train_ann/
│      ├── train_images/
│      ├── val_ann/
│      └── val_images/
│
├── Dataset_YOLOv5/
│      ├── data.yaml
│      ├── README.dataset.txt
│      ├── README.roboflow.txt
│      ├── test/
│      ├── train/
│      └── valid/
│
├── Detection_results/
│      ├── DETR.png
│      ├── ground_truth.png
│      ├── ResNet-50.png
│      ├── ResNet-110.png
│      └── Yolov5.png
│
├── Detr_model_files/
│      ├── config.json
│      └── model.safetensors
│
├── FasterRCNN_model_files/
│      ├── faster_rcnn_resnet50.pth
│      └── faster_rcnn_resnet101.pth
│
├── YOLOv5_model_file/
│      └── best.pt
│
└── Metrics_images/
        ├── Detr_model_ROC_Curve.png
        ├── Detr_PR.png
        ├── Detr_train.png
        ├── F1_score.png
        ├── Faster_RCNN_Resnet_50_Precision-Recall_Curve.png
        ├── Faster_RCNN_Resnet_50_ROC_Curve.png
        ├── Faster_RCNN_Resnet_110_ROC_Curve.png
        ├── Faster_RCNNResnet_110_Precision-Recall_Curve.png
        ├── mAP_score_comparison.png
        ├── Yolov5_PR_curve.png
        ├── YOLOv5_ROC_Curve.png
        └── YOLOv5_train.png
Trained Models
    • Detr_model_files/: Contains the trained DETR model files.
    • FasterRCNN_model_files/: Contains the trained Faster R-CNN model files.
    • YOLOv5_model_file/: Contains the trained YOLOv5 model file.

Notebooks
    • Detection Transformer (DETR): Implemented in ‘DETR.ipynb’.
    • Faster R-CNN: Implemented in ‘FasterRCNN.ipynb’.
    • YOLOv5: Implemented in ‘yolov5.ipynb.

Detection Results’
To obtain the detection or prediction results from all models, please refer to the ‘Detection_code_for_all_model.ipynb’. Running this notebook will generate the detection results.
