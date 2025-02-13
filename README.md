# License Plate Detection with YOLO and PaddleOCR 🚗🔍

## 📌 Project Overview  
This project leverages **YOLO** for detecting license plates and **PaddleOCR** for performing Optical Character Recognition (OCR) on the plates. The aim is to detect and read license plates from video files, apply preprocessing steps to enhance OCR accuracy, and save the annotated video with text overlay. 🧑‍💻🎥

## 🛠️ Tools & Technologies  
- **YOLO** – Object detection model for license plate detection  
- **PaddleOCR** – OCR tool to read text from license plates  
- **OpenCV** – Used for video processing and image manipulations  
- **Python** – Programming language used for implementation  
- **NumPy** – Library for numerical operations  
- **cv2** – OpenCV module for handling video and image processing  

## 🎯 Key Features  
1. **License Plate Detection** – Using YOLO to detect and extract bounding boxes around license plates.  
2. **OCR Preprocessing** – Resize, crop, and enhance images before performing OCR.  
3. **Video Annotation** – Annotate video frames with bounding boxes and OCR text.  
4. **Output** – Generate a new video with text overlay and bounding boxes. 🎬

## 🔧 How it Works  
1. **Video Input** – The project takes a video file as input.  
2. **Object Detection** – YOLO model detects license plates in each frame.  
3. **OCR Processing** – PaddleOCR is applied to extracted license plates to read the text.  
4. **Preprocessing** – Includes resizing, cropping, and scaling for better OCR accuracy.  
5. **Output Video** – The video with annotated bounding boxes and OCR results is saved.  

## 📊 Workflow Overview  
- The video is loaded frame-by-frame.
- YOLO predicts license plate locations and extracts the regions of interest (ROI).
- The image undergoes preprocessing (cropping, resizing) to improve OCR performance.
- PaddleOCR performs OCR on the preprocessed image and extracts text.
- The resulting text is overlaid on the video frame, and the video is saved with annotations.


## 🚀 How to Use  
1. Clone the repository.  
2. Install the required libraries:  
   ```bash
   pip install opencv-python ultralytics paddlepaddle paddleocr numpy
```bash

<h2> Contact </h2>
[Linked-in](https://www.linkedin.com/in/mohamed-wasef-789743233/)

