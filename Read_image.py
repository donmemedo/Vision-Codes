import os
def read_images_add():
    BASE_ADD = "/home/adminstrator/Final-code/test_images"
    image_files = []
    for root, dirs, files in os.walk(BASE_ADD):
        for file in files:
            (os.path.join(root,file))
            image_files.append(os.path.join(root,file))
    return  image_files,

