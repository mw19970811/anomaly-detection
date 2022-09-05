import os, tqdm, shutil
import cv2

root_dir = 'D:/dataset/FMD/full/ori'
target_dir = 'D:/dataset/FMD/full/fine'
txt_file = 'D:/dataset/FMD/Ori/Fine_list.txt'

os.makedirs(target_dir,exist_ok=True)

with open(txt_file,'r') as f:
    file_list = list(map(lambda x:x.split()[0],f.readlines()))

for name in tqdm.tqdm(file_list):
    shutil.copyfile(os.path.join(root_dir,f'{name}.jpg'),os.path.join(target_dir,f'{name}.jpg'))