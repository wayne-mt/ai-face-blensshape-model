original_file = "/home/ubuntu/data/ExpW/data/label/label.lst"
lines = open(original_file, "r").readlines()
new_training_label = "/home/ubuntu/data/ExpW/data/label/expw_arcface_pipeline_label.lst"
fw = open(new_training_label, "w")
import os
img_realfolder = "/home/ubuntu/data/ExpW/data/image_face_aligned/"
real_list = os.listdir(img_realfolder)
real_list_set = set(real_list)


for ll in lines:
    segs = ll.split(" ")
    to_write_line = " ".join([segs[0], segs[-1]])
    if segs[0] in real_list_set:
        fw.write(to_write_line)
fw.close()

import random
random.shuffle(lines)
new_validation_label = "/home/ubuntu/data/ExpW/data/label/expw_arcface_pipeline_val_label.lst"
fw = open(new_validation_label, "w")
for ll in lines[:2500]:
    segs = ll.split(" ")
    to_write_line = " ".join([segs[0], segs[-1]])
    fw.write(to_write_line)
fw.close()