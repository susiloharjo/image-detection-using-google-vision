import io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from google.cloud import vision
from google.cloud.vision_v1 import types

# Create a Vision client
client = vision.ImageAnnotatorClient()

# Read the image file
image_path = "image2.jpg"
with io.open(image_path, "rb") as image_file:  # Use io.open for binary data
    content = image_file.read()

# Create an instance of vision.Image
image = types.Image(content=content)

# Perform object detection on the image (not label detection)
response = client.object_localization(image=image)  # Use object_localization

# Load the image using PIL
img = Image.open(image_path)
draw = ImageDraw.Draw(img)

# Print detected objects and draw rectangles
# Print detected objects, draw rectangles, and add labels
print("Objects:")
for object_annotation in response.localized_object_annotations:
    print(object_annotation.name)

    # Extract bounding box information
    bounding_poly = object_annotation.bounding_poly.normalized_vertices

    if len(bounding_poly) >= 3:
        vertices = [(vertex.x * img.width, vertex.y * img.height) for vertex in bounding_poly]
        draw.polygon(vertices, outline='red')

        # Calculate label position
        x_min, y_min, x_max, y_max = min(vertices[0][0], vertices[2][0]), min(vertices[0][1], vertices[1][1]), max(vertices[1][0], vertices[3][0]), max(vertices[2][1], vertices[3][1])
        text_x = x_min  # Center label horizontally within the rectangle
        text_y = y_max + 15  # Place label 15 pixels below the rectangle

        # Add label text
        font = ImageFont.truetype("arial.ttf", 14)  # Adjust font and size as needed
        draw.text((text_x, text_y), object_annotation.name, font=font, fill='red')

# Save the image with rectangles
img_with_rectangles_path = "image_with_rectangles.jpg"
img.save(img_with_rectangles_path)

# Provide feedback about the saved image path
print(f"Image with rectangles saved at: {img_with_rectangles_path}")

# Show the image with rectangles
plt.imshow(img)
plt.axis('off')
plt.show()
