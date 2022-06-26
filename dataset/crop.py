# encoding:utf-8

from xml.dom import NotFoundErr
import dlib
import numpy as np
import cv2
import os

def rect_to_bb(rect): # 获得人脸矩形的坐标信息
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    return (x, y, w, h)

def shape_to_np(shape, dtype="int"): # 将包含68个特征的的shape转换为numpy array格式
    coords = np.zeros((68, 2), dtype=dtype)
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def resize(image, width=1200):  # 将待检测的image进行resize
    r = width * 1.0 / image.shape[1]
    dim = (width, int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    return resized

def feature(image_file):
    # image_file = "test.jpg"
    # print(image_file)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    image = cv2.imread(image_file)
    h,w,c = image.shape
    # print(h,w,c)
    # image = resize(image, width=1200)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)
    shapes = []

    rect = dlib.rectangle(max(rects[0].left(), 0), max(0, rects[0].top()), min(w, rects[0].right()), min(h, rects[0].bottom()))
    # rect = dlib.rectangle(0, 0, w, h)
    rects = dlib.rectangles()
    rects.append(rect)

    for (i, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        shape = shape_to_np(shape)
        # print(shape[19],shape[24],shape[27])
        shapes.append(shape)

        left = max(0, (shape[0][0]+shape[17][0])//2)
        right = min(w-1, (shape[26][0]+shape[16][0])//2)
        up = int(max(0, min(shape[19][1], shape[24][1])-(shape[27][1] - max(shape[19][1], shape[24][1]))*0.7))
        down = min(h-1, shape[8][1])
    
    # print(left, right, up, down)

    return left, right, up, down


if __name__ == "__main__":
    for fold_name in os.listdir("CASMEII/CASME2_RAW_selected"):
        for clip_name in os.listdir(os.path.join("CASMEII/CASME2_RAW_selected",fold_name)):
            clip_rel = os.path.join("CASMEII/CASME2_RAW_selected", fold_name, clip_name)
            print(clip_rel)
            names = os.listdir(clip_rel)
            f = lambda x:x.split('.')[-1].lower() in ['jpg','bmp','png']
            names = list(filter(f, names))
            names.sort()
            left, right, up, down = feature(os.path.join(clip_rel, names[0]))

            os.makedirs(os.path.join('CASMEII/CASME2_RAW_selected_cropped', fold_name, clip_name), exist_ok=True)
            for name in names:
                image = cv2.imread(os.path.join(clip_rel, name))
                image = image[up:down+1,left:right+1,:]
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join('CASMEII/CASME2_RAW_selected_cropped', fold_name, clip_name, name), image)
    
    for fold_name in os.listdir("SAMM/SAMM"):
        for clip_name in os.listdir(os.path.join("SAMM/SAMM",fold_name)):
            clip_rel = os.path.join("SAMM/SAMM", fold_name, clip_name)
            print(clip_rel)
            names = os.listdir(clip_rel)
            f = lambda x:x.split('.')[-1].lower() in ['jpg','bmp','png']
            names = list(filter(f, names))
            names.sort()
            left, right, up, down = feature(os.path.join(clip_rel, names[0]))

            os.makedirs(os.path.join('SAMM/SAMM_cropped', fold_name, clip_name), exist_ok=True)
            for name in names:
                image = cv2.imread(os.path.join(clip_rel, name))
                image = image[up:down+1,left:right+1,:]
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join('SAMM/SAMM_cropped', fold_name, clip_name, name), image)

    for fold_name in os.listdir("SMIC/SMIC_all_raw/HS"):
        for attr_name in os.listdir(os.path.join("SMIC/SMIC_all_raw/HS",fold_name, 'micro')):
            for clip_name in os.listdir(os.path.join("SMIC/SMIC_all_raw/HS", fold_name, 'micro', attr_name)):
                clip_rel = os.path.join("SMIC/SMIC_all_raw/HS", fold_name, 'micro', attr_name, clip_name)
                print(clip_rel)
                names = os.listdir(clip_rel)
                f = lambda x:x.split('.')[-1].lower() in ['jpg','bmp','png']
                names = list(filter(f, names))
                names.sort()
                left, right, up, down = feature(os.path.join(clip_rel, names[0]))

                os.makedirs(os.path.join('SMIC/SMIC_all_raw/HS_cropped', fold_name, clip_name), exist_ok=True)
                for name in names:
                    image = cv2.imread(os.path.join(clip_rel, name))
                    image = image[up:down+1,left:right+1,:]
                    # print(image.shape)
                    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join('SMIC/SMIC_all_raw/HS_cropped', fold_name, clip_name, name), image)

    # for fold_name in os.listdir("megc2022-synthesis/source_samples"):
    for fold_name in ['SAMM_challenge','casme2_challenge',]:
        for clip_name in os.listdir(os.path.join("megc2022-synthesis/source_samples",fold_name)):
            clip_rel = os.path.join("megc2022-synthesis/source_samples", fold_name, clip_name)
            print(clip_rel)
            names = os.listdir(clip_rel)
            f = lambda x:x.split('.')[-1].lower() in ['jpg','bmp','png']
            names = list(filter(f, names))
            names.sort()
            left, right, up, down = feature(os.path.join(clip_rel, names[0]))

            os.makedirs(os.path.join('megc2022-synthesis/source_samples_cropped', fold_name, clip_name), exist_ok=True)
            for name in names:
                image = cv2.imread(os.path.join(clip_rel, name))
                image = image[up:down+1,left:right+1,:]
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

                cv2.imwrite(os.path.join('megc2022-synthesis/source_samples_cropped', fold_name, clip_name, name), image)


    for fold_name in ['Smic_challenge']:
        for clip_name in os.listdir(os.path.join("megc2022-synthesis/source_samples",fold_name)):
            clip_rel = os.path.join("megc2022-synthesis/source_samples", fold_name, clip_name)
            print(clip_rel)
            names = os.listdir(clip_rel)
            f = lambda x:x.split('.')[-1].lower() in ['jpg','bmp','png']
            names = list(filter(f, names))
            names.sort()

            os.makedirs(os.path.join('megc2022-synthesis/source_samples_cropped', fold_name, clip_name), exist_ok=True)
            for name in names:
                image = cv2.imread(os.path.join(clip_rel, name))
                image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
                cv2.imwrite(os.path.join('megc2022-synthesis/source_samples_cropped', fold_name, clip_name, name), image)


    clip_rel = "megc2022-synthesis/target_template_face"
    names = os.listdir(clip_rel)
    f = lambda x:x.split('.')[-1].lower() in ['jpg','bmp','png']
    names = list(filter(f, names))
    
    os.makedirs(os.path.join('megc2022-synthesis/target_template_face_cropped'), exist_ok=True)
    for name in names:
        left, right, up, down = feature(os.path.join(clip_rel, name))
        image = cv2.imread(os.path.join(clip_rel, name))
        image = image[up:down+1,left:right+1,:]
        image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)

        cv2.imwrite(os.path.join('megc2022-synthesis/target_template_face_cropped', name), image)

