import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread("test image.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
normalized = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray, cmap='gray')
axes[0].set_title("Original Grayscale")
axes[1].imshow(normalized, cmap='gray')
axes[1].set_title("Normalized")
plt.show()
_, thresh = cv2.threshold(normalized, 80, 255, cv2.THRESH_BINARY)
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
        cv2.putText(result, str(i+1), (cx - 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        print(f"Fossil {i+1}: x={cx}, y={cy}")

        #mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        #draw contours
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)



        masked_fossil = cv2.bitwise_and(image, image, mask=mask)

        # Crop
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

    
        x, y, w, h = cv2.boundingRect(contour)
        padding = 5
        x1 = max(x - padding, 0)
        y1 = max(y - padding, 0)
        x2 = min(x + w + padding, image.shape[1])
        y2 = min(y + h + padding, image.shape[0])

        cropped_image = image[y1:y2, x1:x2]
        cropped_mask = mask[y1:y2, x1:x2]

        #background
        masked_black = cv2.bitwise_and(cropped_image, cropped_image, mask=cropped_mask)
        cv2.imwrite(f"fossil_{i+1}_black.jpg", masked_black)

        #no background
        bgra = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2BGRA)
        bgra[:, :, 3] = cropped_mask
        cv2.imwrite(f"fossil_{i+1}_transparent.png", bgra)

cv2.imwrite("output.jpg", result)
print("Saved output.jpg")