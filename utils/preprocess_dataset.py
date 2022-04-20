import numpy as np
import cv2
import os
import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from skimage import transform as trans

def get_aligned(rimg, landmark):
    assert landmark.shape[0] == 106 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 106:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36] + landmark[39]) / 2
        landmark5[1] = (landmark[81] + landmark[93]) / 2
        landmark5[2] = landmark[86]
        landmark5[3] = landmark[52]
        landmark5[4] = landmark[61]
        # print("landmark selected")
        # print(landmark5)
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()

    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img_warp = cv2.warpAffine(rimg,
                         M, (image_size[1], image_size[0]),
                         borderValue=0.0)
    if alignment_flag:
        cv2.imwrite(os.path.join(debug_folder, imgname.replace(".","-aligned.")), img_warp)
    # img_warp = cv2.cvtColor(img_warp, cv2.COLOR_BGR2RGB)
    return img_warp
    # img_flip = np.fliplr(img_warp)
    # img = np.transpose(img_warp, (2, 0, 1))  # 3*112*112, RGB
    # img_flip = np.transpose(img_flip, (2, 0, 1))
    # input_blob = np.zeros((2, 3, image_size[1], image_size[0]), dtype=np.uint8)
    # input_blob[0] = img
    # input_blob[1] = img_flip
    # return input_blob

# input_folder = "/home/ubuntu/data/ExpW/data/image"
# debug_folder = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_debug"
# output_folder = "/home/ubuntu/data/ExpW/data/image_face_aligned"

input_folder = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_04"
debug_folder = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_debug"
output_folder = "/home/ubuntu/data/WeiIphoneCapData/blendshape_wei_04_face_aligned"


if not os.path.exists(debug_folder):
    os.makedirs(debug_folder)
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

img_list = os.listdir(input_folder)
image_size = (128, 128)
print("number of img folder list is {}".format(len(img_list)))

visualize_flag = False
alignment_flag = False

app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'], allowed_modules=['detection', 'landmark_2d_106'])
app.prepare(ctx_id=0, det_size=(640, 640))
src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
src[:, 0] += 8.0
src*=128./112.

existing_saving_files_list = os.listdir(output_folder)
existing_saving_files_set = set(existing_saving_files_list)

for i,imgname in enumerate(img_list):
    if i%100==0:
        print("{}/{}".format(i, len(img_list)))
    if imgname in existing_saving_files_set:
        continue
    img_full_path = os.path.join(input_folder, imgname)
    img = cv2.imread(img_full_path)
    if img is not None:
        faces = app.get(img)
    else:
        faces = []
    if visualize_flag:
        im_cpy = img.copy()

    for face in faces:
        lmk = face.landmark_2d_106
        ret =get_aligned(img, lmk)
        cv2.imwrite(os.path.join(output_folder, imgname), ret)
        # print(lmk.shape)
        if visualize_flag:
            color = (0, 0, 255)
            lmk = np.round(lmk).astype(np.int)
            for i in range(lmk.shape[0]):
                p = tuple(lmk[i])
                if i in [66, 70, 75, 79, 54, 84, 90]:
                    color = [0, 255, 0]
                else:
                    color = (0, 0, 255)
                cv2.circle(im_cpy, p, 2, color, 2, cv2.LINE_AA)
                cv2.putText(im_cpy, str(i-1), p, cv2.FONT_HERSHEY_SIMPLEX, .5,
                            (0,255,0), 1)
            cv2.imwrite(os.path.join(debug_folder, imgname), im_cpy)


