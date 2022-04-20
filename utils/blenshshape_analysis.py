import numpy as np

tr_lst = "/home/ubuntu/data/WeiIphoneCapData/ts_blendshape.lst"

lines = open(tr_lst).readlines()

pos_bin = np.zeros((61,))
neg_bin = np.zeros((61,))


aug_lines = []
#
for ll in lines:
    segs = ll.split(" ")
    for dt in segs[2:]:
        if float(dt) > 0.33:
            aug_lines.append(ll)
            break

#
# newg_lines=[]
# for j,ll in enumerate(aug_lines):
#     segs = ll.split(" ")
#     for i,dt in enumerate(segs[2:]):
#         if float(dt) < 0.:
#             segs[i+2]= str(1)
#         else:
#             segs[i+2] = str(0)
#     newg_lines.append(" ".join(segs)+"\n")


for ll in aug_lines:
    # print(ll)
    segs = ll.split(" ")
    for i,dt in enumerate(segs[2:]):
        dt_v = float(dt)
        if dt_v > 0.25:
            pos_bin[i] +=1
        else:
            neg_bin[i] +=1





for i in range(61):
    print("shape {} p:n {}:{}".format(i, int(pos_bin[i]), int(neg_bin[i])))
print(len(aug_lines))
new_bce_fp=open("/home/ubuntu/data/WeiIphoneCapData/ts_blendshape_aug.lst", "w")
for ll in aug_lines:
    new_bce_fp.write(ll)
new_bce_fp.close()

