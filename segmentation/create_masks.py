import cv2
import numpy as np
import os

input_dir = 'C:/Users/hk/Desktop/unet_segmentation/Drawings'
output_dir = 'C:/Users/hk/Desktop/Masks'
os.makedirs(output_dir, exist_ok=True)

blue = np.array([255, 0, 0]) 

all_files = os.listdir(input_dir)

for fname in all_files:
    fpath = os.path.join(input_dir, fname)
    
    img = cv2.imread(fpath)
    
    mask = cv2.inRange(img, blue, blue)

    out_name = 'mask_' + fname
    output_path = os.path.join(output_dir, out_name)
    cv2.imwrite(output_path, mask)
    print(f"Processed: {out_name}")