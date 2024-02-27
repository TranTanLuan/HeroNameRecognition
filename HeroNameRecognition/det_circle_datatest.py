import cv2 
import numpy as np 
import os
from tqdm import tqdm
import sys

def crop_circle(in_dir, out_dir):
    list_names = os.listdir(in_dir)
    for i in tqdm(range(len(list_names))):
        # Read image. 
        img_path = os.path.join(in_dir, list_names[i])
        img = cv2.imread(img_path, cv2.IMREAD_COLOR) 
        haft_w = int(img.shape[1] / 4)
        img = img[:, :haft_w, :]
        
        # Convert to grayscale. 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        
        # Blur using 3 * 3 kernel. 
        gray_blurred = cv2.blur(gray, (3, 3)) 
        
        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(gray_blurred,  
                        cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                    param2 = 30, minRadius = 1, maxRadius = 40) 
        
        # Draw circles that are detected. 
        if detected_circles is not None: 
        
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
        
            max_a = 0
            max_b = 0
            max_r = 0
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                if max_r < r:
                    max_r = r
                    max_a = a
                    max_b = b
        
            haft_max_r = int(max_r)
            x_min, y_min = max_a - haft_max_r, max_b - haft_max_r
            x_max, y_max = max_a + haft_max_r, max_b + haft_max_r
            out_img = img[y_min:y_max, x_min:x_max]
            cv2.imwrite(os.path.join(out_dir, list_names[i]), out_img)

if __name__ == "__main__":
    in_dir = sys.argv[1]
    out_dir = sys.argv[2]
    os.makedirs(out_dir, exist_ok=True)
    crop_circle(in_dir, out_dir)