****Evaluating the Impact of Data Augmentation in Object Detection****
**Project Overview**

This project evaluates the effect of different data augmentation techniques on object detection performance using the YOLOv5 model, trained on the COCO128 dataset. Various augmentation techniques, such as scaling, rotation, horizontal flipping, color jittering, Gaussian blurring, and perspective transformation, were applied. The model's performance was measured through mean Average Precision (mAP) scores.

**Requirements**

Required Packages and Versions
To run this project, the following packages are required. In Google Colab, most packages come pre-installed, but for specific versions, you can use !pip install to install or upgrade as needed.

Python >= 3.8
torch (PyTorch) >= 1.10
torchvision >= 0.11
numpy >= 1.19
opencv-python >= 4.5
matplotlib >= 3.4
Pillow >= 8.0
scipy >= 1.7
yolov5 (GitHub repository for YOLOv5)


**Setting Up and Running the Project:**

	Step 1: Open Google Colab and Connect to a GPU
		Open Google Colab.
		Select Runtime > Change runtime type.
		Choose GPU in the Hardware accelerator dropdown, then click Save.

	Step 2: Clone the YOLOv5 Repository
		Clone the YOLOv5 repository and navigate to its directory:

		!git clone https://github.com/ultralytics/yolov5
		%cd yolov5

	Step 3: Install Required Packages
		Install any additional or custom packages as specified in the requirements file:

		!pip install -r requirements.txt

	Step 4: Load the Dataset
		Download and unzip the COCO128 dataset:

		!curl -L "https://github.com/ultralytics/yolov5/releases/download/v1.0/coco128.zip" -o coco128.zip
		!unzip -q coco128.zip

	Step 5: Test the Model without Augmentation
		Evaluate the baseline performance of the model without augmentation:

		!python val.py --img 640 --batch 16 --data coco128.yaml --weights yolov5s.pt

	Step 6: Prepare Data Augmentation Code
		Implement or upload your code for each augmentation technique, including:
		Scaling (scale.py)
		Rotation (rotation.py)
		Horizontal flipping (horizontal_flip.py)
		Color jittering (color_jitter.py)
		Gaussian blurring (Gaussian.py)
		Perspective transformation (perspective_transform.py)
		You may also prepare the modified YAML file for the augmented dataset as needed.
		coco128_sc.yaml,coco128_ro.yaml,coco128_hf.yaml,coco128_cj.yaml,coco128_gb.yaml,coco128_pt.yaml

	Step 7: Train the Model with Augmentation
		Train the model with each augmentation technique applied individually:

		!python train.py --img 640 --batch 16 --epochs 100 --data /content/coco128_sc.yaml --hyp hyp.no-augmentation.yaml --weights yolov5s.pt --patience 5

	Step 8: Test the Fine-Tuned Model
		Evaluate the model's performance after training with augmentation:

		!python val.py --img 640 --batch 16 --data /content/coco128_sc.yaml --weights /content/yolov5/runs/train/exp/weights/best.pt

**Additional Notes**

Data Augmentation Configuration: Modify the augmentation settings in the YAML or configuration files as needed for each test.
Performance Metrics: Collect and compare the mAP scores for each augmentation technique to evaluate its impact on object detection performance.
