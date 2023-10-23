from PIL import Image, ImageOps
import os, sys
import numpy as np

# Resizes all images, maintaning aspect ratio
# Maximum height is 2560 and saves resized images in output_folder
def resize(dirs, path, output_folder):
    for item in dirs:
        if os.path.isfile(path + item):
            im = Image.open(path + item)
            f, e = os.path.splitext(path + item)
            im.thumbnail((2560, 2560))
            output_path = os.path.join(output_folder, os.path.basename(f) + '.png')
            im.save(output_path, 'PNG', quality=90)

# Padding of the image so that each images has dimensions 640x2560
def padding(expected_size,path,dirs):
    for item in dirs:
        if os.path.isfile(path+item):
            desired_size = expected_size
            
            img = Image.open(path+item)
            f, e = os.path.splitext(path+item)
            
            delta_width = desired_size[0] - img.size[0]
            delta_height = desired_size[1] - img.size[1]
            pad_width = delta_width // 2
            pad_height = delta_height // 2
            padding = (pad_width, pad_height, delta_width - pad_width, delta_height - pad_height)
            
            im_expand = ImageOps.expand(img, padding)
            im_expand.save(f + ' expand.png', 'png', quality=90) 


def slicing(dirs, input_img_path, input_mask_path, output_img_path, output_mask_path, number_of_slices):
    for img_file in dirs:
        img_name, img_ext = os.path.splitext(img_file)
        mask_name = 'mask' + img_name[5:] + img_ext  # Construct the corresponding mask file name
        
        img_path = os.path.join(input_img_path, img_file)
        mask_path = os.path.join(input_mask_path, mask_name)  # Construct the full mask file path
        
        if os.path.isfile(img_path) and os.path.isfile(mask_path):
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            width, height = img.size
            slice_height = height // number_of_slices
            
            for i in range(number_of_slices):
                top = i * slice_height
                bottom = (i + 1) * slice_height
                
                img_crop = img.crop((0, top, width, bottom))
                mask_crop = mask.crop((0, top, width, bottom))
                
                output_img_file = os.path.join(output_img_path, img_name + '_' + str(i) + img_ext)
                output_mask_file = os.path.join(output_mask_path, img_name + '_' + str(i) + img_ext)
                
                img_crop.save(output_img_file, 'PNG', quality=95)
                mask_crop.save(output_mask_file, 'PNG', quality=95)


# Function to check if the sum of pixel intensities in the mask is below the threshold
def has_sum_below_threshold(mask, threshold):
    if np.sum(mask) < threshold:
        print(np.sum(mask))
    return np.sum(mask) < threshold
