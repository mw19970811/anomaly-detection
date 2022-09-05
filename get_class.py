import os, glob
import tqdm
import pprint
from lxml import etree
import json
import shutil

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

def parse_dict_class_name(xml_path):
    class_list = []
    try:
        with open(xml_path,encoding='utf8') as fid:
            xml_str = fid.read()
    except Exception as er:
        print(er)
        print(f"{xml_path} can't be opened!")
        return []
    xml = etree.fromstring(xml_str)
    data = parse_xml_to_dict(xml)["annotation"]

    # write object info into txt
    assert "object" in data.keys(), "file: '{}' lack of object key.".format(xml_path)
    if len(data["object"]) == 0:
        # 如果xml文件中没有目标就直接忽略该样本
        print("Warning: in '{}' xml, there are no objects.".format(xml_path))

    for obj in data["object"]:
        # 获取每个object的class信息
        class_list.append(obj['name'])
    return class_list

if __name__ == '__main__':
    xml_path = 'D:/dataset/FMD/OriCroppedMarked/'
    save_txt_path = 'D:/dataset/FMD/OriCroppedMarked/label_names.txt'
    class_list = []
    for root, dirs, files in os.walk(xml_path):
        for dir in dirs:
            if dir == 'Annotations':
                for xml_file in glob.glob(os.path.join(root, 'Annotations', '*.xml')):
                    name = os.path.split(xml_file)[-1][:-4]
                    if os.path.exists(os.path.join(root,'JPEGImages',f'{name}.jpg')):
                        class_list.extend(parse_dict_class_name(xml_file))
    class_list = list(set(class_list))
    with open(save_txt_path,'w') as f:
        f.writelines(map(lambda x:f'{x[1]} {x[0]}\n',enumerate(class_list)))