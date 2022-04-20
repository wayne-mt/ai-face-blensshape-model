train_lst1 = "/home/ubuntu/data/WeiIphoneCapData/MySlate_4_iPhoneLiveCtr.csv"
train_lst2 = "/home/ubuntu/data/WeiIphoneCapData/MySlate_5_iPhoneLiveCtr.csv"
test_lst = "/home/ubuntu/data/WeiIphoneCapData/MySlate_6_iPhoneLiveCtr.csv"



lines1 = open(train_lst1, "r").readlines()[1:]
for i, ll in enumerate(lines1):
    prefolder = "blendshape_wei_04_face_aligned/"
    segs = ll.split(",")
    segs[0] = prefolder+"img"+str(i+1).zfill(6)+".jpg"
    lines1[i] = ",".join(segs)

lines2 =open(train_lst2, "r").readlines()[1:]
for i, ll in enumerate(lines2):
    prefolder = "blendshape_wei_05_face_aligned/"
    segs = ll.split(",")
    segs[0] = prefolder+"img"+str(i+1).zfill(6)+".jpg"
    lines2[i] = ",".join(segs)

lines3 =open(test_lst, "r").readlines()[1:]
for i, ll in enumerate(lines3):
    prefolder = "blendshape_wei_06_face_aligned/"
    segs = ll.split(",")
    segs[0] = prefolder+"img"+str(i+1).zfill(6)+".jpg"
    lines3[i] = ",".join(segs)



lines_training = lines1 + lines2
lines_testing = lines3

##
img_folder_1 = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_04_face_aligned"
img_folder_2 = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_05_face_aligned"
img_folder_3 = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_06_face_aligned"

img_set=set()
import os
img_list_1 = os.listdir(img_folder_1)
fd = img_folder_1.split("/")[-1]
img_list_1 = [fd+"/"+t for t in img_list_1]

img_list_2 = os.listdir(img_folder_2)
fd = img_folder_2.split("/")[-1]
img_list_2 = [fd+"/"+t for t in img_list_2]

img_list_3 = os.listdir(img_folder_3)
fd = img_folder_3.split("/")[-1]
img_list_3 = [fd+"/"+t for t in img_list_3]

img_list = img_list_1 + img_list_2 + img_list_3

for pp in img_list:
    img_set.add(pp)
print(img_list[:5])
print("total img files number is {}".format(len(img_list)))

tr_lst = "/home/ubuntu/data/WeiIphoneCapData/tr_blendshape.lst"
ts_lst = "/home/ubuntu/data/WeiIphoneCapData/ts_blendshape.lst"

fw = open(tr_lst, "w")
for ll in lines_training:
    segs = ll.split(",")
    new_ll = " ".join(segs)
    if segs[0] in img_set:
        fw.write(new_ll)
fw.close()

fw = open(ts_lst, "w")
for ll in lines_testing:
    segs = ll.split(",")
    new_ll = " ".join(segs)
    if segs[0] in img_set:
        fw.write(new_ll)
fw.close()
