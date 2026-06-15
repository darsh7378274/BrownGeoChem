import cv2
import numpy as np

image = cv2.imread("test image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_area = 100
fossils = [c for c in contours if cv2.contourArea(c) > min_area]

print(f"Number of fossils detected: {len(fossils)}")

result = image.copy()
for i, contour in enumerate(fossils):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        cv2.circle(result, (cx, cy), 20, (0, 255, 0), 2)
        cv2.putText(result, str(i+1), (cx-10, cy+10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        # Crop each fossil
        x, y, w, h = cv2.boundingRect(contour)
        padding = 10
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])
        
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"fossil_{i+1}.jpg", crop)
        
        crop = image[y1:y2, x1:x2]
        cv2.imwrite(f"fossil_{i+1}.jpg", crop)
        print(f"Fossil {i+1}: x={cx}, y={cy}")

cv2.imwrite("output.jpg", result)
print("Saved output.jpg")