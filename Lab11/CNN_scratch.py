## Implement convolution operations from scratch. Assume a 3x3 kernel and apply it on an input image of 32x32.

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input image
# ============================================================

image = np.zeros((32, 32))

image[10:22, 10:22] = 1


# ============================================================
# 3x3 Kernel
# ============================================================

kernel = np.array([
    [1,  0, -1],
    [1,  0, -1],
    [1,  0, -1]
])


# ============================================================
# Convolution function
# ============================================================

def convolution(image, kernel):

    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    # Output dimensions
    output_height = image_height - kernel_height + 1
    output_width = image_width - kernel_width + 1

    # Create empty output
    output = np.zeros((output_height, output_width))

    # Slide the kernel over the image
    for i in range(output_height):

        for j in range(output_width):

            # Extract 3x3 region
            region = image[
                i:i + kernel_height,
                j:j + kernel_width
            ]

            # Element-wise multiplication
            multiplication = region * kernel

            # Sum all values
            output[i, j] = np.sum(multiplication)

    return output


# ============================================================
# Apply convolution
# ============================================================

output = convolution(image, kernel)


# ============================================================
# Print shapes
# ============================================================

print("Input image shape :", image.shape)
print("Kernel shape      :", kernel.shape)
print("Output shape      :", output.shape)


# ============================================================
# Display
# ============================================================

plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Input Image")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(output, cmap="gray")
plt.title("Convolution Output")
plt.axis("off")

plt.show()