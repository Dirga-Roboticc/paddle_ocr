from paddleocr import PaddleOCR
import cv2
import numpy as np

# Path to the exported model
model_dir = './output/det_db_inference/Student2'

# Initialize PaddleOCR with the exported model
ocr = PaddleOCR(det_model_dir=model_dir,
                det_model_type='ch_PP-OCRv4',
                use_angle_cls=False)

# Path to the image for inference
img_path = './tes.png'

# Perform inference
result = ocr.ocr(img_path)
print(result)

# Load the original image
img = cv2.imread(img_path)

# Loop through the detection results and draw bounding boxes
for line in result[0]:
    points = line[0]
    
    # Check if 'points' is a list or array
    if isinstance(points, list) and all(isinstance(p, list) or isinstance(p, tuple) for p in points):
        # Draw bounding box (rectangle around the text)
        points = [(int(point[0]), int(point[1])) for point in points]
        cv2.polylines(img, [np.array(points)], isClosed=True, color=(0, 255, 0), thickness=2)
        
        # Get the text and its position
        text = line[1][0]
        text_position = (points[0][0], points[0][1] - 10)  # Adjust the y-coordinate to position the label
        
        # Draw the label on the image
        cv2.putText(img, text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    else:
        print(f"Unrecognized coordinate format: {points}")

# Save the image with bounding boxes
output_path = './output.png'
cv2.imwrite(output_path, img)

print(f"Result saved to {output_path}")
