import cv2
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR

# Initialize YOLO model
model = YOLO('/content/best.pt')

# Initialize PaddleOCR reader
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')

# Open video file
video_capture = cv2.VideoCapture('/content/demo.mp4')

# Create a VideoWriter object to save the annotated video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('/content/annotated_video_with_text.avi', fourcc, 30, (int(video_capture.get(3)), int(video_capture.get(4))))

# Define preprocessing parameters
initial_crop_percentage = 0.1  # 10% of the ROI dimensions
top_bottom_crop_percentage = 0.1  # 10% of the image height
scale_factor = 4  # Resize image to four times its original size

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model.predict(rgb_image, conf=0.3)

    # Access detection results
    for result in results:
        boxes = result.boxes  # Access bounding boxes

        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
            conf = box.conf[0]  # Get confidence
            cls = box.cls[0]  # Get class label

            # Extract license plate
            license_plate = rgb_image[y1:y2, x1:x2]

            # Image preprocessing
            # Resize image to enhance OCR accuracy
            resized_image = cv2.resize(license_plate, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

            # Additional cropping parameters
            h, w = resized_image.shape[:2]

            # First additional cropping 
            crop_x1 = int(w * initial_crop_percentage)
            crop_y1 = int(h * initial_crop_percentage)
            crop_x2 = w - crop_x1
            crop_y2 = h - crop_y1
            additional_cropped_image = resized_image[crop_y1:crop_y2, crop_x1:crop_x2]

            # Second additional cropping (top and bottom only)
            top_bottom_crop_y1 = int(h * top_bottom_crop_percentage)
            top_bottom_crop_y2 = h - top_bottom_crop_y1
            second_additional_cropped_image = additional_cropped_image[top_bottom_crop_y1:top_bottom_crop_y2, :]

            # Resize image to four times its original size before OCR
            final_resized_image = cv2.resize(second_additional_cropped_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

            # Perform OCR on the processed image
            ocr_result = ocr_reader.ocr(final_resized_image)
            if ocr_result[0] is None:
                continue
            txts = [line[1][0] for line in ocr_result[0]]

            # Draw bounding box with YOLO-style
            thickness = 2  # Thickness of the rectangle
            top_thickness = 5  # Thickness of the top side
            color = (0, 255, 0)  # Green color for the rectangle

            # Draw the top side of the rectangle
            cv2.line(frame, (x1, y1), (x2, y1), color, top_thickness)
            # Draw the other sides of the rectangle
            cv2.line(frame, (x2, y1), (x2, y2), color, thickness)
            cv2.line(frame, (x2, y2), (x1, y2), color, thickness)
            cv2.line(frame, (x1, y2), (x1, y1), color, thickness)

            # Draw OCR text on the frame
            text = txts[0]
            text_color = (0, 0, 0)  # Black color for text
            background_color = (255, 255, 255)  # White background for text

            # Calculate the position of the text
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            text_w, text_h = text_size

            # Background rectangle for the text
            cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1 - 10), background_color, -1)
            cv2.putText(frame, text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)

    # Write the frame to the output video
    out.write(frame)

# Release resources
video_capture.release()
out.release()
cv2.destroyAllWindows()
