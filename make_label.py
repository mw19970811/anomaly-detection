import os, glob
import tqdm
import pprint
from lxml import etree
import json
import shutil

save_txt_path = 'D:/dataset/FMD/OriCroppedMarked/label_names.txt'
class_dict = dict(map(lambda x: (x.split()[0],int(x.split()[1])),open(save_txt_path,'r').readlines()))
total_img = 0
total_obj = 0

def parse_xml_to_dict(xml):
    """
    将xml文件解析成字典形式，参考tensorflow的recursive_parse_xml_to_dict
    Args：
        xml: xml tree obtained by parsing XML file contents using lxml.etree

    Returns:
        Python dictionary holding XML contents.
    """

    if len(xml) == 0:  # 遍历到底层，直接返回tag对应的信息
        return {xml.tag: xml.text}

    result = {}
    for child in xml:
        child_result = parse_xml_to_dict(child)  # 递归遍历标签信息
        if child.tag != 'object':
            result[child.tag] = child_result[child.tag]
        else:
            if child.tag not in result:  # 因为object可能有多个，所以需要放入列表里
                result[child.tag] = []
            result[child.tag].append(child_result[child.tag])
    return {xml.tag: result}

def xml_to_txt(xml_path,txt_path):
    global total_img
    global total_obj
    # read xml
    with open(xml_path,encoding='utf8') as fid:
        xml_str = fid.read()
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]

    # write object info into txt
    assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
    if len(data["object"]) == 0:
        # 如果xml文件中没有目标就直接忽略该样本
        print("Warning: in '{}' xml, there are no objects.".format(xml_path))

    with open(txt_path,'w') as f:
        for index, obj in enumerate(data["object"]):
            # 获取每个object的box信息
            xmin = int(obj["bndbox"]["xmin"])
            xmax = int(obj["bndbox"]["xmax"])
            ymin = int(obj["bndbox"]["ymin"])
            ymax = int(obj["bndbox"]["ymax"])
            class_name = obj["name"]
            class_index = class_dict[class_name]  # 目标id从0开始

            # 进一步检查数据，有的标注信息中可能有w或h为0的情况，这样的数据会导致计算回归loss为nan
            if xmax <= xmin or ymax <= ymin:
                print("Warning: in '{}' xml, there are some bbox w/h <=0".format(xml_path))
                continue
            
            line = f'{class_index} {xmin} {ymin} {xmax} {ymax}\n'
            # print(line,end='')
            f.write(line)
            total_obj += 1
    total_img += 1

if __name__ == '__main__':
    xml_path = 'D:/dataset/FMD/OriCroppedMarked'
    for root, dirs, files in os.walk(xml_path):
        for dir in dirs:
            if dir == 'Annotations':
                for xml_file in glob.glob(os.path.join(root, 'Annotations', '*.xml')):
                    name = os.path.split(xml_file)[-1][:-4]
                    if os.path.exists(os.path.join(root,'JPEGImages',f'{name}.jpg')):
                        xml_to_txt(xml_file,os.path.join(root,dir,f'{name}.txt'))
    print('total img: ',total_img)
    print('total obj: ',total_obj)