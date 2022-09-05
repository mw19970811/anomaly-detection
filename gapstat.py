import os, glob
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
with open(label_names_path,'r',encoding='utf8') as f:
    label_dict = dict(map(lambda x:(int(x.split()[1]),x.split()[0]),f.readlines()))

def eliminate_edge(img):
    if len(img.shape) == 3:
        H, W, C = img.shape
        for h in range(0, H+10, 1024):
            img[(h-2):(h+2)] = 0
        for w in range(0, W+10, 1024):
            img[:, (w-2):(w+2)] = 0
        # print('--ok--',h,w)
    return img


def gapStat(data, refs=None, nrefs=10):
    # MC
    data = data.astype('float')
    shape = data.shape
    if refs == None:
        x_max = data.max(axis=0)
        x_min = data.min(axis=0)
        dists = np.matrix(np.diag(x_max-x_min))
        rands = np.random.random_sample(size=(shape[0], shape[1], nrefs))
        for i in range(nrefs):
            rands[:, :, i] = rands[:, :, i]*dists+x_min
    else:
        rands = refs
    gaps = []
    gapDiff = []
    sdk = []
    cluster_means = []
    cluster_ress = []
    k = 50
    while True:
        (cluster_mean, cluster_res) = scipy.cluster.vq.kmeans2(data, k)
        Wk = np.linalg.norm(data-cluster_mean[cluster_res], axis=-1).sum()
        WkRef = np.zeros((rands.shape[2],))
        for j in range(rands.shape[2]):
            (kmc, kml) = scipy.cluster.vq.kmeans2(rands[:, :, j], k)
            WkRef[j] = np.linalg.norm(rands[:, :, j]-kmc[kml], axis=-1).sum()
        cluster_means.append(cluster_mean)
        cluster_ress.append(cluster_res)
        gaps.append(np.log(np.mean(WkRef))-np.log(Wk))
        sdk.append(np.sqrt((1.0+nrefs)/nrefs)*np.std(np.log(WkRef)))

        if len(gaps) > 1:
            gapDiff.append(gaps[-1] - gaps[-2] - sdk[-1])
            if gapDiff[-1] < 0:
                break
        k += 1
    return gaps, gapDiff, cluster_means[-2], cluster_ress[-2], k - 1


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
    # his = cv2.calcHist([img],channels=[0],mask=None,histSize=[256],ranges=(0,255))
    his_mean = np.mean(img)
    his_std = np.std(img)
    img = np.any(img > (his_mean + 6 * his_std),
                     axis=-1).astype('uint8') * 255
    anomaly_points = np.argwhere(img).reshape((-1,2))
    if len(anomaly_points):
        print(anomaly_points.shape)
        _, _, _, cluster_ress, k = gapStat(anomaly_points, refs=None, nrefs=10)

        ob_clusters = []
        local_mins = []
        local_maxs = []
        nb_clusters = []
        for i in range(k):
            ob_cluster = anomaly_points[cluster_ress == i]
            ob_cluster = ob_cluster.reshape((-1, 2))
            if len(ob_cluster):
                ob_clusters.append(ob_cluster)
                nb_clusters.append(len(ob_cluster))
                local_mins.append(np.min(ob_cluster, axis=0))
                local_maxs.append(np.max(ob_cluster, axis=0))

        cluster_indices = np.argsort(nb_clusters)[::-1]
        cluster_indices = cluster_indices[:20]
        ob_clusters = np.array(ob_clusters,dtype=object)[cluster_indices]
        local_mins = np.array(local_mins)[cluster_indices]
        local_maxs = np.array(local_maxs)[cluster_indices]

        for ob_cluster, local_min, local_max in zip(ob_clusters, local_mins, local_maxs):
            cv2.rectangle(
                out_img, local_min[::-1], local_max[::-1], color=(0, 0, 255), thickness=2)

    cv2.imwrite(predicted_img, out_img)
    cv2.imwrite(point_cloud_img, img)


if __name__ == '__main__':
    # label_file = 'D:/dataset/FMD/OriCroppedMarked/C_top/605_top/Annotations/3_143000.824_7_level2.txt'
    # ori_img_file = 'D:/dataset/FMD/OriCroppedMarked/C_top/605_top/JPEGImages/3_143000.824_7_level2.jpg'
    # img_file = 'D:/dataset/FMD/OriCroppedMarked/C_top/605_top/Output/3_143000.824_7_level2_res.jpg'
    # point_cloud_img = 'D:/dataset/FMD/OriCroppedMarked/C_top/605_top/Output/3_143000.824_7_level2_cloud.jpg'
    # predicted_img = 'D:/dataset/FMD/OriCroppedMarked/C_top/605_top/Output/3_143000.824_7_level2_predicted.jpg'
    # create_predicted_images(label_file,ori_img_file,img_file,point_cloud_img,predicted_img)
    root_path = 'D:/dataset/FMD/OriCroppedMarked'
    for root, dirs, files in os.walk(root_path):
        for dir in dirs:
            if dir == 'Annotations':
                os.makedirs(os.path.join(root,'Output'),exist_ok=True)
                for label_file in glob.glob(os.path.join(root, 'Annotations', '*.txt')):
                    name = os.path.split(label_file)[-1][:-4]
                    ori_img_file = os.path.join(root,'JPEGImages',f'{name}.jpg')
                    img_file = os.path.join(root,'Output',f'{name}_res.jpg')
                    if os.path.exists(ori_img_file) and os.path.exists(img_file):
                        point_cloud_img = os.path.join(root,'Output',f'{name}_cloud.jpg')
                        predicted_img = os.path.join(root,'Output',f'{name}_predicted.jpg')
                        create_predicted_images(label_file,ori_img_file,img_file,point_cloud_img,predicted_img)
    