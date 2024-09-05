import cv2
import numpy as np
from paddleocr import PaddleOCR
import matplotlib.pyplot as plt

# Initialize PaddleOCR reader 
ocr_reader = PaddleOCR(use_angle_cls=True, lang='en')  # Set use_angle_cls=True to enable angle classification (note important)

cropped_rois = []

# Define additional cropping parameters
initial_crop_percentage = 0.1  # Adjust as needed (10% of the ROI dimensions)
top_bottom_crop_percentage = 0.1  # Adjust as needed (10% of the image height)

# Define dilation and erosion parameters
kernel = np.ones((3, 3), np.uint8)  # 3x3 kernel; adjust as needed
dilation_iterations = 1  # Number of dilation iterations
erosion_iterations = 1  # Number of erosion iterations

# Define resizing scale factor
scale_factor = 4  # Resize image to 4 times its original size

# Process detection results
for result in results:
    # Extract boxes, confidence scores, and class IDs
    boxes = result.boxes.xyxy  # Bounding box coordinates
    confidences = result.boxes.conf  # Confidence scores
    class_ids = result.boxes.cls  # Class IDs

    for box, confidence, class_id in zip(boxes, confidences, class_ids):
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to int

        # Crop the ROI from the image
        roi = img[y1:y2, x1:x2]

        # Image preprocessing for better OCR results

        # 1. Convert to grayscale
        gray_image = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # 2. Resize image to enhance OCR accuracy
        resized_image = cv2.resize(gray_image, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

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

        # Apply binarization
        # 1. Otsu's Thresholding
        _, binarized_image = cv2.threshold(second_additional_cropped_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # 2. Apply dilation
        dilated_image = cv2.dilate(binarized_image, kernel, iterations=dilation_iterations)

        # 3. Apply erosion
        eroded_image = cv2.erode(dilated_image, kernel, iterations=erosion_iterations)

        # 4. Resize image to 4times its original size before OCR
        final_resized_image = cv2.resize(eroded_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

        # Perform OCR using PaddleOCR on the resized image
        ocr_result = ocr_reader.ocr(final_resized_image)

        # Check if OCR result is valid and non-empty
        if ocr_result[0] is None:
            continue
        txts = [line[1][0] for line in ocr_result[0]]

        # Save the cropped and processed ROI to a file
        roi_filename = f"roi_{class_id}_{x1}_{y1}_{x2}_{y2}.jpg"
        cv2.imwrite(roi_filename, final_resized_image)
        cropped_rois.append(roi_filename)

        # Display the cropped and processed ROI
        plt.imshow(final_resized_image)
        plt.title(f"OCR Result: {txts} ")
        plt.axis('off')
        plt.show()

        # print the OCR results
        print(f"OCR Result: {txts}")
