import os
import pickle
from _util.util_v1 import * ; import _util.util_v1 as uutil
import multiprocessing
import cv2
from tqdm.contrib.concurrent import thread_map
def process_image(bn, ed):
    image = cv2.imread(bn,cv2.IMREAD_UNCHANGED)
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    for y in range(height):
        for x in range(width):
            alpha = image[y,x,3]
            if alpha > 200:
                mask[y,x] = 255
    output_folder = os.path.dirname(ed)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    cv2.imwrite(ed, mask)
    
img_num = 16
bns = [f'./_data/lustrous/renders/human_rutileE/xyza/{bn[-1]}/{bn}/{i:04d}.png' for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_all.csv')for i in range(img_num)]
ens = [f'./_data/lustrous/renders/human_rutileE/rgb_mask/{bn[-1]}/{bn}/{i:04d}.png' for bn in uutil.read_bns('./_data/lustrous/subsets/human_rutileEB_all.csv')for i in range(img_num)]
pool = multiprocessing.Pool(processes=16)
pool.starmap(process_image, zip(bns, ens))
pool.close()
pool.join()
print("down!")





    
