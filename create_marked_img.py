import os, glob
import cv2
import numpy as np

def create_masked_img(img_path,out_path,bboxes,label_dict):
    point_color = (0, 255, 0)
    thickness = 2
    lineType = 4
    src = cv2.imread(img_path)
    for bbox in bboxes:
        label  = label_dict[bbox[0]]
        ptLeftTop = bbox[1:3]
        ptRightBottom = bbox[3:]

        cv2.rectangle(src, ptLeftTop, ptRightBottom, point_color, thickness, lineType)

        t_size = cv2.getTextSize(label, 1, cv2.FONT_HERSHEY_PLAIN, 1)[0]
        textlbottom = ptLeftTop + np.array(list(t_size))
        cv2.rectangle(src, tuple(ptLeftTop), tuple(textlbottom),  point_color, -1)
        ptLeftTop[1] = ptLeftTop[1] + (t_size[1]/2 + 4)
        cv2.putText(src, label, tuple(ptLeftTop), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 0, 255), 1)

    cv2.imwrite(out_path, src)

if __name__ == '__main__':
    xml_path = 'D:/dataset/FMD/OriCroppedMarked'
    label_names_path = 'D:/dataset/FMD/OriCroppedMarked/label_names.txt'
    with open(label_names_path,'r',encoding='utf8') as f:
        label_dict = dict(map(lambda x:(int(x.split()[1]),x.split()[0]),f.readlines()))
    for root, dirs, files in os.walk(xml_path):
        for dir in dirs:
            if dir == 'Annotations':
                os.makedirs(os.path.join(root,'BoxesMarked'),exist_ok=True)
                for label_file in glob.glob(os.path.join(root, 'Annotations', '*.txt')):
                    name = os.path.split(label_file)[-1][:-4]
                    img_path = os.path.join(root,'JPEGImages',f'{name}.jpg')
                    if os.path.exists(img_path):
                        out_path = os.path.join(root,'BoxesMarked',f'{name}.jpg')
                        with open(label_file,'r',encoding='utf8') as f:
                            bboxes = np.array(list(map(lambda x:list(map(lambda y:int(y),x.split())),f.readlines())))
                        create_masked_img(img_path,out_path,bboxes,label_dict)
    