import os
import pickle
from _util.util_v1 import * ; import _util.util_v1 as uutil
import cv2


front_bns = [
    f'./_data/lustrous/renders/rutileE/ortho/{bn[-1]}/{bn}/front.png'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
back_bns = [
    f'./_data/lustrous/renders/rutileE/ortho/{bn[-1]}/{bn}/back.png'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
left_bns = [
    f'./_data/lustrous/renders/rutileE/ortho/{bn[-1]}/{bn}/left.png'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
right_bns = [
    f'./_data/lustrous/renders/rutileE/ortho/{bn[-1]}/{bn}/right.png'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
top_bns = [
    f'./_data/lustrous/renders/rutileE/ortho/{bn[-1]}/{bn}/top.png'
    for bn in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]

front_eds = [
    f'./_data/lustrous/renders/rutileE/ortho_mask/{ed[-1]}/{ed}/front.png'
    for ed in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
back_eds = [
    f'./_data/lustrous/renders/rutileE/ortho_mask/{ed[-1]}/{ed}/back.png'
    for ed in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
left_eds = [
    f'./_data/lustrous/renders/rutileE/ortho_mask/{ed[-1]}/{ed}/left.png'
    for ed in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
right_eds = [
    f'./_data/lustrous/renders/rutileE/ortho_mask/{ed[-1]}/{ed}/right.png'
    for ed in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]
top_eds = [
    f'./_data/lustrous/renders/rutileE/ortho_mask/{ed[-1]}/{ed}/top.png'
    for ed in uutil.read_bns('./_data/lustrous/subsets/rutileEB_all.csv')
]

bns_eds_list = [zip(front_bns,front_eds),zip(back_bns,back_eds),zip(left_bns,left_eds),zip(right_bns,right_eds)]
for cur in bns_eds_list:
    for bn,ed in tqdm(cur):
        # preprocess
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
    
