# Projet-E4

Luggage Detection and Tracking System
This repository contains the code for a luggage detection and tracking system, designed to identify and monitor abandoned luggage in public spaces or facilities. The system leverages advanced computer vision techniques and deep learning models to provide real-time alerts for security and surveillance purposes.

Overview
The project utilizes a combination of Python, MongoDB, OpenCV, and the Ultralytics YOLO model to analyze video footage, frame by frame, and detect abandoned luggage. It includes mechanisms for tracking objects across frames, evaluating their movement (or lack thereof), and storing tracking data in MongoDB for further analysis.

Features
Real-time Object Detection and Tracking: Uses YOLOv8 for efficient and accurate identification of luggage and people within the video frames.
MongoDB Integration: Stores tracking information and associations between detected objects in a MongoDB database for persistence and later analysis.
Abandoned Luggage Alerting: Implements logic to determine when luggage has been stationary for a configurable amount of time, triggering alerts for potential security concerns.


Installation

1.Install Required Packages: Ensure Python 3.6+ is installed on your system, then install the required Python packages: pip install pandas pymongo opencv-python-headless ultralytics

2.MongoDB Setup: Make sure MongoDB is installed and running on mongodb://localhost:27017/. Create the tracking_database and the collections humans, suitcases, and associations as needed.

3.Environment Configuration: Set the necessary environment variable to avoid OpenMP conflicts: export KMP_DUPLICATE_LIB_OK=TRUE  #For Unix systems  or  set KMP_DUPLICATE_LIB_OK=TRUE  #For Windows

4.Model Download: Download the YOLOv8 model weights (yolov8n.pt) from the Ultralytics website or use the model provided by the YOLO package.

Usage
To run the luggage detection and tracking system, execute the main Python script with the path to your video file: python track_and_associate.py --source Video1.mp4
The system will process the video, detecting and tracking luggage and people. When a piece of luggage remains stationary beyond the configured threshold, it triggers an alert and marks the luggage in the video output.

Output
The processed video is saved with visual indicators for abandoned luggage. Tracking information, detection details, and object associations are stored in the designated MongoDB collections.

Customization
You can adjust several parameters, including the MongoDB URI, the detection threshold, and the stationary frame count threshold, to tailor the system to your specific needs and environment
