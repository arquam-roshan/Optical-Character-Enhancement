import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Correct the image angle

# Load the image
image_path = '/Users/arquamroshan/Desktop/DIP-CW/images/download.png'
image = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply edge detection
edges = cv2.Canny(gray, 50, 150, apertureSize=3)

# Detect lines using Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Calculate the angle of each line
angles = []
for line in lines:
    x1, y1, x2, y2 = line[0]
    angle = np.arctan2(y2 - y1, x2 - x1)
    angles.append(angle)

# Compute the median angle
median_angle = np.median(angles)

# Rotate the image to correct the skew
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Convert angle from radians to degrees and rotate the image
angle_degrees = median_angle * (180 / np.pi)
corrected_image = rotate_image(image, angle_degrees)


# Display the corrected image and extracted text
# print("Corrected Image:")
# plt.figure(figsize=(10, 10))
# plt.subplot(1, 2, 1)
# plt.title("Corrected Image")
# plt.imshow(cv2.cvtColor(corrected_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')

# plt.show()
# print("Successfully Corrected the image")


# Step 2: Upscale the image
scale_factor = 15  # You can adjust the scale factor as needed
upscaled_image = cv2.resize(corrected_image, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

# Display images using matplotlib
# plt.figure(figsize=(20, 20))

# Display the upscaled image
# plt.subplot(1, 2, 1)
# plt.title("Upscaled Image")
# plt.imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
# # print("Sucessfully Upscaled")

# Step 3: Crop the image

# Convert to grayscale
gray = cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2GRAY)

# Apply adaptive thresholding to highlight text
threshold_image = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 11, 2)

# Find contours in the thresholded image
contours, _ = cv2.findContours(threshold_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to keep only the largest contour area (text region)
mask = np.zeros_like(gray)
for contour in contours:
    # Draw the contour on the mask
    cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)

# Find bounding box of the largest contour
x, y, w, h = cv2.boundingRect(mask)
cropped_image = upscaled_image[y:y+h, x:x+w]

# Display the original and cropped images
# plt.figure(figsize=(20, 20))
# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(upscaled_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.subplot(1, 3, 2)
# plt.title("Cropped Image")
# plt.imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
# plt.axis('off')
# plt.show()
print("Successfully Cropped")

# Step 4: Gray scale the cropped image
gray_cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)

# Display Gray scale image
# plt.figure(figsize=(20, 20))
# plt.subplot(1, 2, 1)
# plt.title("Gray Cropped Image")
# plt.imshow(gray_cropped_image, cmap='gray')
# plt.axis('off')
# plt.show()
# print("Successfully Grayscaled")

# Step 5: Denoising the image

# Remove noise using Gaussian Blurring
denoised_image = cv2.GaussianBlur(gray_cropped_image, (5, 5), 0)

# Display denoised image
# plt.figure(figsize=(20, 20))
# plt.subplot(1, 2, 1)
# plt.title("Denoised Image")
# plt.imshow(denoised_image, cmap='gray')
# plt.axis('off')
# plt.show()
# print("Successfully Denoised")


# Step 6: Apply adaptive thresholding to highlight text
binary_image = cv2.adaptiveThreshold(denoised_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY_INV, 11, 2)

# Define a horizontal kernel
horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))

# Detect horizontal lines using morphological operations
detected_lines = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, horizontal_kernel, iterations=2)

# Subtract the detected lines from the binary image to remove them
letters_only = cv2.subtract(binary_image, detected_lines)

# Apply contour detection to isolate letters
contours, _ = cv2.findContours(letters_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Create a mask to keep only the letters
letter_mask = np.zeros_like(letters_only)
cv2.drawContours(letter_mask, contours, -1, (255), thickness=cv2.FILLED)

# Apply the mask to the denoised image to isolate the letters
final_image = cv2.bitwise_and(denoised_image, denoised_image, mask=letter_mask)

# display the images
plt.figure(figsize=(20, 20))

# plt.subplot(1, 3, 1)
# plt.title("Original Image")
# plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
# plt.axis('off')

plt.subplot(1, 2, 1)
plt.title("Binary Image")
plt.imshow(binary_image, cmap='gray')
plt.axis('off')


plt.show()