from PIL import Image
import numpy as np

def load_image(image_path):
    """ Load an image and convert it to grayscale for comparison. """
    with Image.open(image_path) as img:
        img = img.convert('L')  # Convert to grayscale
        return np.array(img)    # Convert to a NumPy array

def compare_images(image1_path, image2_path):
    """ Compare two images and return whether they are identical. """
    img1 = load_image(image1_path)
    img2 = load_image(image2_path)
    
    # Check if the images are of the same size
    if img1.shape != img2.shape:
        return False
    
    # Compare pixel values
    return np.array_equal(img1, img2)

# Paths to your images
image1_path = "D:\DataScience_For_mySelf\Projects myself\RagMLOPS\TriModalRAG_System\data\image\weather_images_47.png"
image2_path = "D:\DataScience_For_mySelf\Projects myself\RagMLOPS\TriModalRAG_System\data\image\weather_images_49.png"

# Compare the images
are_identical = compare_images(image1_path, image2_path)
print(f"The images are {'identical' if are_identical else 'different'}.")