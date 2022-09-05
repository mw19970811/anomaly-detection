import os, glob, tqdm, sys
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
import numpy as np
import cv2
import matplotlib.pyplot as plt
from create_marked_img import create_masked_img
import warnings
warnings.filterwarnings("ignore")

label_names_path = 'D:/dataset/FMD/OriCroppedMarked/label_names.txt'
root_path = 'D:/dataset/FMD/OriCroppedMarked'
with open(label_names_path,'r',encoding='utf8') as f:
    label_dict = dict(map(lambda x:(int(x.split()[1]),x.split()[0]),f.readlines()))

UNCLASSIFIED = -1
NOISE = -2


def indices2logic(m:np.array,size:int):
    n = np.zeros(size,dtype='bool')
    n[m] = True
    return n


def eliminate_edge(img):
    if len(img.shape) == 3:
        H, W, C = img.shape
        for h in range(0, H+10, 1024):
            img[(h-2):(h+2)] = 0
        for w in range(0, W+10, 1024):
            img[:, (w-2):(w+2)] = 0
        # print('--ok--',h,w)
    return img


def count_meaning_bboxes(src_bboxes, predict_points, predict_class, nb_classes):
    mem = np.zeros(len(src_bboxes))
    # print(src_bboxes.shape)
    xmin, ymin, xmax, ymax = np.split(src_bboxes, 4,axis=-1)
    xmin, ymin, xmax, ymax = xmin.reshape(-1,1), ymin.reshape(-1,1), xmax.reshape(-1,1), ymax.reshape(-1,1)
    for indices in range(nb_classes):
        # print(len(predict_class))
        # print(len(predict_points))
        cluster_points = predict_points[predict_class==indices][:,::-1]
        xp, yp = np.split(cluster_points, 2,axis=-1)
        xp, yp = xp.reshape(1,-1), yp.reshape(1,-1)
        cluster_length = len(cluster_points)
        nb_predicted = np.all((xmin<xp,xp<xmax,ymin<yp,yp<ymax),axis=0)
        nb_predicted = (np.sum(nb_predicted,axis=-1) > (cluster_length*0.7)).astype('int')
        mem = np.any((mem,nb_predicted),axis=0)
        # print('nb predicted: ',nb_predicted)
        # print('nb target: ',cluster_length)
    print('mem mean: ',mem.mean(),mem.sum(),len(mem))
    return mem.sum()


def create_predicted_images(label_file, ori_img_file, img_file, point_cloud_img, predicted_img):
    # 在原始残差图的基础上进行二值化得到点云图，并对点云图进行聚类来预测异物的位置，
    # 最后得到二值化图和带有真实标签框和预测标签框的标注图
    global label_dict
    with open(label_file, 'r', encoding='utf8') as f:
        bboxes = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split())), f.readlines())))
        
    out_img = cv2.imread(ori_img_file)
    for bbox in bboxes:
        ptLeftTop = bbox[1:3]
        ptRightBottom = bbox[3:]
        cv2.rectangle(out_img, ptLeftTop, ptRightBottom, color=(
            0, 255, 0), thickness=2, lineType=4)

        label  = label_dict[bbox[0]]
        t_size = cv2.getTextSize(label, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        textlbottom = ptLeftTop + np.array(list(t_size))
        cv2.rectangle(out_img, ptLeftTop, textlbottom,  (0,255,0), -1)
        ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
        cv2.putText(out_img, label, ptLeftTop, cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)
    
    img = cv2.imread(img_file)
    H, W, C = out_img.shape
    img = img[:H,:W]
    img = eliminate_edge(img)
    img = cv2.boxFilter(img, -1, (9, 9))
    his_mean = np.mean(img)
    his_std = np.std(img)
    img_kernel = np.any(img > (his_mean + 8 * his_std),
                     axis=-1).astype('uint8') * 255
    img_total = np.any(img > (his_mean + 6 * his_std),
                     axis=-1).astype('uint8') * 255
    kernel_points = np.argwhere(img_kernel).reshape((-1,2))
    total_points = np.argwhere(img_total).reshape((-1,2))
    radius = 9
    eps = 4
    n_kernel_points = len(kernel_points)
    n_total_points = len(total_points)
    classifications = np.array([UNCLASSIFIED] * n_total_points)
    cluster_id = 0
    for point_id in tqdm.tqdm(kernel_points):
        point_ind = np.argwhere(np.all(total_points == point_id,axis=-1)).item()
        if classifications[point_ind] == UNCLASSIFIED: 
            local_set_indices = (np.linalg.norm(total_points-point_id,axis=-1) < radius)
            if np.sum(local_set_indices) > eps:
                classifications[point_ind] = cluster_id
                seeds = np.argwhere(local_set_indices & (classifications == UNCLASSIFIED)).flatten()
                seeds = seeds.tolist()
                while True:
                    if len(seeds) == 0: break
                    seed = seeds.pop(0)
                    if seed in seeds: sys.exit()
                    sub_set_indices = (np.linalg.norm(total_points-total_points[seed],axis=-1) < radius)
                    if np.sum(sub_set_indices) > eps:
                        classifications[seed] = cluster_id
                        sub_seeds = np.argwhere(sub_set_indices & (classifications == UNCLASSIFIED) & ~indices2logic(seeds,n_total_points))
                        sub_seeds = sub_seeds.flatten().tolist()
                        seeds.extend(sub_seeds)
                    else:
                        classifications[seed] = NOISE
                cluster_id += 1
            else:
                classifications[point_ind] = NOISE
        
    # print(cluster_id)

    m = 0
    total = 0
    for clu_id in range(cluster_id):
        cluster_points = total_points[classifications==clu_id]
        # print(len(cluster_points))
        min_points = np.min(cluster_points,axis=0)
        max_points = np.max(cluster_points,axis=0)
        # print(max_points-min_points)
        cv2.rectangle(out_img,min_points[::-1],max_points[::-1],(0,0,255),thickness=2)
    with open(label_file, 'r', encoding='utf8') as f:
        bboxes = np.array(
            list(map(lambda x: list(map(lambda y: int(y), x.split())), f.readlines())))
    print(bboxes)
    m += count_meaning_bboxes(bboxes[:,1:],total_points,classifications,cluster_id)
    total += len(bboxes)
    
    cv2.imwrite(predicted_img, out_img)
    cv2.imwrite(point_cloud_img, img_total)

    return m, total


if __name__ == '__main__':
    ms, totals = 0, 0
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            if dir == 'Annotations':
                os.makedirs(os.path.join(root,'Output'),exist_ok=True)
                for label_file in glob.glob(os.path.join(root, 'Annotations', '*.txt')):
                    name = os.path.split(label_file)[-1][:-4]
                    ori_img_file = os.path.join(root,'JPEGImages',f'{name}.jpg')
                    img_file = os.path.join(root,'Output',f'{name}_res.jpg')
                    if os.path.exists(ori_img_file) and os.path.exists(img_file):
                        point_cloud_img = os.path.join(root,'Output',f'{name}_cloud6.jpg')
                        predicted_img = os.path.join(root,'Output',f'{name}_predicted_radius9.jpg')
                        print(name)
                        m, total = create_predicted_images(label_file,ori_img_file,img_file,point_cloud_img,predicted_img)
                        with open(os.path.join(root,'Output',f'{name}_annotation.txt'),'w') as f:
                            f.write(f'The number of marked images {total}\n')
                            f.write(f'The number of predicted images {m}\n')
                        ms += m
                        totals += total
    
    print('accuracy: ',ms/totals)
    