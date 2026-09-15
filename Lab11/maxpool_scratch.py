import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Input image
# ============================================================

image = np.random.rand(32, 32)


# ============================================================
# Max Pooling function
# ============================================================

def max_pooling(image, pool_size=2, stride=2):

    image_height, image_width = image.shape

    # Calculate output dimensions
    output_height = (image_height - pool_size) // stride + 1
    output_width = (image_width - pool_size) // stride + 1

    # Create output
    output = np.zeros((output_height, output_width))

    # Slide pooling window over image
    for i in range(0, output_height):

        for j in range(0, output_width):

            # Starting position of pooling window
            row_start = i * stride
            col_start = j * stride

            # Extract pooling region
            region = image[
                row_start:row_start + pool_size,
                col_start:col_start + pool_size
            ]

            # Take maximum value
            output[i, j] = np.max(region)

    return output


# ============================================================
# Apply Max Pooling
# ============================================================

output = max_pooling(
    image,
    pool_size=2,
    stride=2
)


# ============================================================
# Print shapes
# ============================================================

print("Input image shape :", image.shape)
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
plt.title("Max Pooling Output")
plt.axis("off")

plt.show()