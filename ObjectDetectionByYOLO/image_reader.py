"""Read_image"""
import os
def read_images_add():
    """Get Address of Test Images"""
    base_add = "./pics"
    image_files = []
    for root, dirs, files in os.walk(base_add):
        for file in dirs:
            print(os.path.join(root,file))
            image_files.append(os.path.join(root,file))
        for file in files:
            print(os.path.join(root,file))
            image_files.append(os.path.join(root,file))
    return image_files
