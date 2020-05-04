import numpy as np
import os    #遍历文件夹
import nibabel as nib #nii格式一般都会用到这个包
import imageio   #转换成图像
import shutil
import random
import pandas as pd

def nii_to_image(niifile):
    for root, dirs, files in os.walk(filepath):
        for name in files:
            # print(os.path.join(root, name))
            img_path = os.path.join(root, name)
            if name == ".DS_Store":
                continue
            try:
                #开始读取nii文件
                img = nib.load(img_path)    #读取nii
                img_fdata = img.get_fdata()
                fname = name.replace('.nii','')   #去掉nii的后缀名
                img_f_path = os.path.join(imgfile, fname)
                #创建nii对应的图像的文件夹
                if not os.path.exists(img_f_path):
                    os.mkdir(img_f_path)    #新建文件夹

                #开始转换为图像
                (x,y,z) = img.shape
                for i in range(z):      #z是图像的序列
                    silce = img_fdata[i, :, :]   #选择哪个方向的切片都可以
                    imageio.imwrite(os.path.join(img_f_path,'{}.png'.format(i)), silce)
                                #保存图像
            except:
                print(img_path)

def count_img(path):
    min = 999999
    max = 0
    for root, dirs, files in os.walk(path):
        # print(len(files))
        if files == ['.DS_Store']:
            continue
        if len(files) < min:
            min = len(files)
            file_path = files
        if len(files) > max:
            max = len(files)
    print(min, max)
    print(file_path)

def find_target():
    df=pd.read_csv('ADNI1_Complete_1Yr_1.5T_1_05_2020.csv')
    data_dict = {}      
    def map_dict(item):
        data_dict[str(item["Image Data ID"])] = [item["Group"]]
    df.apply(map_dict,axis=1)
    
    return data_dict

def split_dirs(open_path, save_root, start, view):
    if not os.path.exists(save_root):
        os.mkdir(save_root)
    for root, dirs, files in os.walk(open_path):
        if files == ['.DS_Store']:
            continue
        new_root = root[root.rfind('_I') + 2:]
        data_dict = find_target()
        if new_root not in data_dict:
            print(new_root)
            continue
        save_path = os.path.join(save_root, new_root)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cou_num = 0
        for i in range(start, len(files), int((len(files) - start) / view)):
            if cou_num < view :
                file_path = os.path.join(root,'{}.png'.format(i))
                new_file_path = os.path.join(save_path, '{}.png'.format(i))
                shutil.copyfile(file_path, new_file_path)
                cou_num = cou_num + 1

def random_split(path, new_path):
    if not os.path.exists(new_path):
        os.mkdir(new_path)
    cou = 0
    for root, dirs, files in os.walk(path):
        if files == ['.DS_Store']:
            continue
        # print(root)
        nn = root.rfind('/')
        if nn < 0:
            continue
        new_root = os.path.join(new_path, 'train', root[nn + 1:])
        # print(new_root)
        if random.randint(1, 10) == 1 and cou < 200:
            new_root = os.path.join(new_path, 'test')
            if not os.path.exists(new_root):
                os.mkdir(new_root)
            shutil.move(root, new_root)
            cou = cou + 1
        else:
            new_root = os.path.join(new_path, 'train')
            if not os.path.exists(new_root):
                os.mkdir(new_root)
            shutil.move(root, new_root)

def cou(path):
    cn = 0
    mci = 0
    ad = 0
    c = 0
    for files in os.listdir(path):
        if files == '.DS_Store':
            continue
        # print(files)
        c = c + 1
        df=pd.read_csv('ADNI.csv')
        data_dict = {}      
        def map_dict(item):
            data_dict[str(item["ID"])] = item["target"]
        df.apply(map_dict,axis=1)
        # print(data_dict[files])
        if(data_dict[files]==0):
            cn = cn + 1
        elif(data_dict[files]==1):
            mci = mci + 1
        elif(data_dict[files]==2):
            ad = ad + 1
    print(c)
    print(cn,mci,ad) 

if __name__ == '__main__':
    filepath = 'nii_raw_data'
    imgfile = 'nii_image'
    save_root = 'view_image'
    data = 'data'
    # nii_to_image(filepath)
    # split_dirs(imgfile, save_root, 30, 20)
    # print(find_target())
    # random_split(save_root, data)
    cou('data/train')
    