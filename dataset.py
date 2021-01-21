import os
import sys
import re
import six
import math
import lmdb
import torch
import random

from natsort import natsorted
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, ConcatDataset, Subset
from torch._utils import _accumulate
import torchvision.transforms as transforms
from scipy.io import loadmat

sys.path.insert(0,"/home/ldl/桌面/论文/文本识别/TextRecognitionDataGenerator")
from trdg.generators import GeneratorFromDict

class Batch_Balanced_Dataset(object):

    def __init__(self, opt):
        """
        Modulate the data ratio in the batch.
        For example, when select_data is "MJ-ST" and batch_ratio is "0.5-0.5",
        the 50% of the batch is filled with MJ and the other 50% of the batch is filled with ST.
        """
        log = open(f'./saved_models/{opt.exp_name}/log_dataset.txt', 'a')
        dashed_line = '-' * 80
        print(dashed_line)
        log.write(dashed_line + '\n')
        print(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}')
        log.write(f'dataset_root: {opt.train_data}\nopt.select_data: {opt.select_data}\nopt.batch_ratio: {opt.batch_ratio}\n')
        assert len(opt.select_data) == len(opt.batch_ratio)

        _AlignCollate = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
        self.data_loader_list = []
        self.dataloader_iter_list = []
        batch_size_list = []
        Total_batch_size = 0
        for selected_d, batch_ratio_d in zip(opt.select_data, opt.batch_ratio):
            _batch_size = max(round(opt.batch_size * float(batch_ratio_d)), 1)
            print(dashed_line)
            log.write(dashed_line + '\n')
            _dataset, _dataset_log = hierarchical_dataset(root=opt.train_data, opt=opt, select_data=[selected_d])
            total_number_dataset = len(_dataset)
            log.write(_dataset_log)

            """
            The total number of data can be modified with opt.total_data_usage_ratio.
            ex) opt.total_data_usage_ratio = 1 indicates 100% usage, and 0.2 indicates 20% usage.
            See 4.2 section in our paper.
            """
            number_dataset = int(total_number_dataset * float(opt.total_data_usage_ratio))
            dataset_split = [number_dataset, total_number_dataset - number_dataset]
            indices = range(total_number_dataset)
            _dataset, _ = [Subset(_dataset, indices[offset - length:offset])
                           for offset, length in zip(_accumulate(dataset_split), dataset_split)]
            selected_d_log = f'num total samples of {selected_d}: {total_number_dataset} x {opt.total_data_usage_ratio} (total_data_usage_ratio) = {len(_dataset)}\n'
            selected_d_log += f'num samples of {selected_d} per batch: {opt.batch_size} x {float(batch_ratio_d)} (batch_ratio) = {_batch_size}'
            print(selected_d_log)
            log.write(selected_d_log + '\n')
            batch_size_list.append(str(_batch_size))
            Total_batch_size += _batch_size

            _data_loader = torch.utils.data.DataLoader(
                _dataset, batch_size=_batch_size,
                shuffle=True,
                num_workers=int(opt.workers),
                collate_fn=_AlignCollate, pin_memory=True)
            self.data_loader_list.append(_data_loader)
            self.dataloader_iter_list.append(iter(_data_loader))

        Total_batch_size_log = f'{dashed_line}\n'
        batch_size_sum = '+'.join(batch_size_list)
        Total_batch_size_log += f'Total_batch_size: {batch_size_sum} = {Total_batch_size}\n'
        Total_batch_size_log += f'{dashed_line}'
        opt.batch_size = Total_batch_size

        print(Total_batch_size_log)
        log.write(Total_batch_size_log + '\n')
        log.close()

    def get_batch(self):
        balanced_batch_images = []
        balanced_batch_texts = []

        for i, data_loader_iter in enumerate(self.dataloader_iter_list):
            try:
                image, text = data_loader_iter.next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except StopIteration:
                self.dataloader_iter_list[i] = iter(self.data_loader_list[i])
                image, text = self.dataloader_iter_list[i].next()
                balanced_batch_images.append(image)
                balanced_batch_texts += text
            except ValueError:
                pass

        balanced_batch_images = torch.cat(balanced_batch_images, 0)

        return balanced_batch_images, balanced_batch_texts


def hierarchical_dataset(root, opt, select_data='/'):
    """ select_data='/' contains all sub-directory of root directory """
    dataset_list = []
    dataset_log = f'dataset_root:    {root}\t dataset: {select_data[0]}'
    print(dataset_log)
    dataset_log += '\n'
    for dirpath, dirnames, filenames in os.walk(root+'/'):
        if not dirnames:
            select_flag = False
            for selected_d in select_data:
                if selected_d in dirpath:
                    select_flag = True
                    break

            if select_flag:
                dataset = LmdbDataset(dirpath, opt)
                sub_dataset_log = f'sub-directory:\t/{os.path.relpath(dirpath, root)}\t num samples: {len(dataset)}'
                print(sub_dataset_log)
                dataset_log += f'{sub_dataset_log}\n'
                dataset_list.append(dataset)

    concatenated_dataset = ConcatDataset(dataset_list)

    return concatenated_dataset, dataset_log

def iiit5k_mat_extractor(label_path):
    '''
    This code is to extract mat labels from IIIT5k dataset
    Input:
    label_path: mat label path file
    Output:
    dict_img: [image_name, labels, small_lexicon, medium_lexicon]
    '''
    # create empty list for news items
    dict_img = []

    mat_contents = loadmat(label_path)

    if 'traindata' in mat_contents:
        key = 'traindata'
    else:
        key = 'testdata'
    for i in range(len(mat_contents[key][0])):
        name = mat_contents[key][0][i][0][0]
        label = mat_contents[key][0][i][1][0]
        #small_lexi = [item[0] for item in mat_contents[key][0][i][2][0]]
        #medium_lexi = [item[0] for item in mat_contents[key][0][i][3][0]]
        dict_img.append([name, label])

    return dict_img
class base_cutimg_dataset(Dataset):
    def __init__(self,total_img_path,annotation_path,rgb_mode=False,sensitive=False,max_char_length=90):
        self.total_img_path = total_img_path
        self.annotation_path = annotation_path
        self.rgb_mode = rgb_mode
        self.sensitive = sensitive
        self.max_char_length = max_char_length

        self.dataset = self.loaddataset(total_img_path,annotation_path)

    def loaddataset(self, total_img_path, annotation_path):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        img_name, bdb, label = self.dataset[index]
        if len(label) > self.max_char_length:
            return None
        # print(img_name)
        
        if self.rgb_mode:
            img = Image.open(os.path.join(self.total_img_path,img_name)).convert('RGB')  # for color image
            print(img.size)
            (W,H,_) = img.size
            
        else:
            img = Image.open(os.path.join(self.total_img_path,img_name)).convert('L')
            W,H = img.size
        bdb = self.ctrl_xt(bdb,H,W)
        # print(H,W,bdb,label)
        img = img.crop(bdb)
        # if self.rgb_mode:
        #     raise NotImplementedError
        # else:
        #     W,H = img.size
        #     if H > W or W <= 0 or H<=0:
        #         print(W,H,'------------',img_name)
        #         return None

        return img, label
    
    def ctrl_xt(self,bdb,imgH,imgW):
        tleft_x,tleft_y,bright_x,bright_y = bdb
        tleft_x = max(0,tleft_x)
        tleft_y = max(0,tleft_y)
        bright_x = min(imgW-1,bright_x)
        bright_y = min(imgH-1,bright_y)

        return tleft_x,tleft_y,bright_x,bright_y

class mytrdg_cutimg_dataset(base_cutimg_dataset):
    def __init__(self,**argsdict):
        super().__init__(**argsdict)

    def loaddataset(self,total_img_path, annotation_path):
        dataset = []
        for labelname in os.listdir(annotation_path):
            imgname = labelname[:-3]+'jpg'
            for line in open(os.path.join(annotation_path,labelname)):
                # print(line)
                tleft_x,tleft_y,x1,y1,bright_x,bright_y,x2,y2,*label = line.split(',')
                x = [int(i) for i in [tleft_x,x1,bright_x,x2]]
                y = [int(i) for i in [tleft_y,y1,bright_y,y2]]

                tleft_x = min(x)
                tleft_y = min(y)
                bright_x = max(x)
                bright_y = max(y)
                h = bright_y - tleft_y
                w = bright_x - tleft_x
                if h > w or h <=1 or w <= 1:
                    continue

                dataset.append([imgname,[tleft_x,tleft_y,bright_x,bright_y],','.join(label).strip()])
        return dataset


class iiit5k_dataset_builder(Dataset):
    def __init__(self,total_img_path, annotation_path,opt):
        '''
        total_img_path: path with all images
        annotation_path: mat labeling file
        '''
        self.opt = opt
        self.total_img_path = total_img_path
        self.dictionary = iiit5k_mat_extractor(annotation_path)
        self.total_img_name = os.listdir(total_img_path)
        self.dataset = []

        for items in self.dictionary:
            if items[0].split('/')[-1] in self.total_img_name:
                self.dataset.append([items[0].split('/')[-1],items[1]])

    def __getitem__(self, index):
        img_name, label = self.dataset[index]
        if self.opt.rgb:
            img = Image.open(os.path.join(self.total_img_path,img_name)).convert('RGB')  # for color image
        else:
            img = Image.open(os.path.join(self.total_img_path,img_name)).convert('L')
        if not self.opt.sensitive:
            label = label.lower()

        # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
        out_of_char = f'[^{self.opt.character}]'
        label = re.sub(out_of_char, '', label)

        return (img, label)

    def __len__(self):
        return len(self.dataset)

class TextRecognition(Dataset):
    def __init__(self,count,textlength,dictpath):
        self.count = count
        fonts_dir = "/home/ldl/桌面/python-notebook/My_trdg/trdg/fonts/cn"
        fonts = [os.path.join(fonts_dir,i) for i in os.listdir(fonts_dir)]
        # dictpath = "/home/ldl/桌面/论文/文本识别/TextRecognitionDataGenerator/trdg/mydicts/all_4068.txt"
        img_dir = "/home/ldl/桌面/论文/文本识别/TextRecognitionDataGenerator/trdg/images"
        self.args = dict(
            count=self.count,
            length=textlength,
            allow_variable=True,
            fonts=fonts,
            language=dictpath,
            size=64,
            blur=2,
            random_blur=True,
            image_dir=img_dir,
            background_type=[0,1,2,3],
            distorsion_type=[0,1,2],
            text_color="#000000,#FF8F8F",
            image_mode="L",
            char_cat="",
            space_width=[1,2,3,4],
            character_spacing=[0,1,2,3,4,5])
        self.generator = GeneratorFromDict(**self.args)

    def __getitem__(self,index):
        try:
            img,label = self.generator.next()
        except StopIteration:
            self.generator = GeneratorFromDict(**self.args)
            img,label = self.generator.next()
        
        return img,label

    def __len__(self):
        return self.count

class TalOcrChnDataset(Dataset):

    def __init__(self,root):
        self.root = root
        self.count = len(os.listdir(root)) // 2

    def __getitem__(self,index):
        labelpath = os.path.join(self.root,f"{index}.txt")
        with open(labelpath,'r') as f:
            label = f.read().strip()
        imgpath = os.path.join(self.root,f"{index}.jpg")
        return Image.open(imgpath).convert('L'), label

    def __len__(self):
        return self.count

class TalOcrEngDataset(Dataset):

    def __init__(self,imgdir,labeltxtfile,rgb_mode=False):
        self.imgdir = imgdir
        self.dataset = self.loaddataset(labeltxtfile)
        self.rgb_mode = rgb_mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        imgname,label = self.dataset[index]

        if self.rgb_mode:
            return Image.open(os.path.join(self.imgdir,imgname)).convert('RGB'), label
        else:
            return Image.open(os.path.join(self.imgdir,imgname)).convert('L'), label

    def loaddataset(self,labeltxtfile):
        dataset = []
        for line in open(labeltxtfile):
            imgname,label = line.split("jpg ")
            dataset.append([imgname+'jpg',label.strip()])
        return dataset

class PpocrDataset(Dataset):

    def __init__(self,imgdir,labelfilepath,length,rgb_mode=False,split='\t'):
        self.imgdir = imgdir
        self.length = length
        self.rgb_mode = rgb_mode
        self.labeliter = open(labelfilepath)
        self.split = split
        # self.number = 0
        # self.test = test
        self.dataset = [self.get_imgname_label(i) for i in open(labelfilepath)]

    def __len__(self):
        return self.length
        # return len(self.dataset)
 
    def __getitem__(self,index):
        # if self.test and self.number<4000:
        #     imgname,label = random.choice(self.dataset)
        #     self.number += 1
        # elif self.number >= 4000:
        #     self.number = 0
        #     raise StopIteration
        # else:
        #     imgname,label = self.dataset[index]
        
        

        try:
            imgname,label =  random.choice(self.dataset)
            if self.rgb_mode:
                
                return Image.open(os.path.join(self.imgdir,imgname)).convert('RGB'), label
            else:
                return Image.open(os.path.join(self.imgdir,imgname)).convert('L'), label
        except:
            return None
    
    def get_imgname_label(self,line):

        # print(line)
        try:
            imgname,label = line.split(self.split)
            
            if imgname.startswith("Chinese_dataset") or imgname.startswith("Synthetic_Chinese_String_Dataset"):
                imgdir ,imgname = imgname.split('/')
                imgname = os.path.join(imgdir,"images",imgname)
            label = label.strip()
        except:
            print(line)
            return None
        # 过滤掉比九十还大的样本
        if len(label) > 90:
            return None
        return ((imgname+self.split).strip(),label)

class PpocrDataset_formtwi(PpocrDataset):

    def __init__(self,*args):
        super().__init__(*args)
    
    def get_imgname_label(self,line):
        try:
            s = line.split(".jpg")
            imgname = '.jpg'.join(s[:-1])+'.jpg'
            label = s[-1].strip()
            return (imgname,label)
        except :
            return None
class LmdbDataset(Dataset):

    def __init__(self, root, opt):

        self.root = root
        self.opt = opt
        self.env = lmdb.open(root, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
        if not self.env:
            print('cannot create lmdb from %s' % (root))
            sys.exit(0)

        with self.env.begin(write=False) as txn:
            nSamples = int(txn.get('num-samples'.encode()))
            self.nSamples = nSamples

            if self.opt.data_filtering_off:
                # for fast check or benchmark evaluation with no filtering
                self.filtered_index_list = [index + 1 for index in range(self.nSamples)]
            else:
                """ Filtering part
                If you want to evaluate IC15-2077 & CUTE datasets which have special character labels,
                use --data_filtering_off and only evaluate on alphabets and digits.
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/6593928855fb7abb999a99f428b3e4477d4ae356/dataset.py#L190-L192

                And if you want to evaluate them with the model trained with --sensitive option,
                use --sensitive and --data_filtering_off,
                see https://github.com/clovaai/deep-text-recognition-benchmark/blob/dff844874dbe9e0ec8c5a52a7bd08c7f20afe704/test.py#L137-L144
                """
                self.filtered_index_list = []
                for index in range(self.nSamples):
                    index += 1  # lmdb starts with 1
                    label_key = 'label-%09d'.encode() % index
                    label = txn.get(label_key).decode('utf-8')

                    if len(label) > self.opt.batch_max_length:
                        # print(f'The length of the label is longer than max_length: length
                        # {len(label)}, {label} in dataset {self.root}')
                        continue

                    # By default, images containing characters which are not in opt.character are filtered.
                    # You can add [UNK] token to `opt.character` in utils.py instead of this filtering.
                    out_of_char = f'[^{self.opt.character}]'
                    if re.search(out_of_char, label.lower()):
                        continue

                    self.filtered_index_list.append(index)

                self.nSamples = len(self.filtered_index_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'
        index = self.filtered_index_list[index]

        with self.env.begin(write=False) as txn:
            label_key = 'label-%09d'.encode() % index
            label = txn.get(label_key).decode('utf-8')
            img_key = 'image-%09d'.encode() % index
            imgbuf = txn.get(img_key)

            buf = six.BytesIO()
            buf.write(imgbuf)
            buf.seek(0)
            try:
                if self.opt.rgb:
                    img = Image.open(buf).convert('RGB')  # for color image
                else:
                    img = Image.open(buf).convert('L')

            except IOError:
                print(f'Corrupted image for {index}')
                # make dummy image and dummy label for corrupted image.
                if self.opt.rgb:
                    img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
                else:
                    img = Image.new('L', (self.opt.imgW, self.opt.imgH))
                label = '[dummy_label]'

            if not self.opt.sensitive:
                label = label.lower()

            # We only train and evaluate on alphanumerics (or pre-defined character set in train.py)
            out_of_char = f'[^{self.opt.character}]'
            label = re.sub(out_of_char, '', label)

        return (img, label)


class RawDataset(Dataset):

    def __init__(self, root, opt):
        self.opt = opt
        self.image_path_list = []
        for dirpath, dirnames, filenames in os.walk(root):
            for name in filenames:
                _, ext = os.path.splitext(name)
                ext = ext.lower()
                if ext == '.jpg' or ext == '.jpeg' or ext == '.png':
                    self.image_path_list.append(os.path.join(dirpath, name))

        self.image_path_list = natsorted(self.image_path_list)
        self.nSamples = len(self.image_path_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):

        try:
            if self.opt.rgb:
                img = Image.open(self.image_path_list[index]).convert('RGB')  # for color image
            else:
                img = Image.open(self.image_path_list[index]).convert('L')

        except IOError:
            print(f'Corrupted image for {index}')
            # make dummy image and dummy label for corrupted image.
            if self.opt.rgb:
                img = Image.new('RGB', (self.opt.imgW, self.opt.imgH))
            else:
                img = Image.new('L', (self.opt.imgW, self.opt.imgH))

        return (img, self.image_path_list[index])


class ResizeNormalize(object):

    def __init__(self, size, interpolation=Image.BICUBIC):
        self.size = size
        self.interpolation = interpolation
        self.toTensor = transforms.ToTensor()

    def __call__(self, img):
        img = img.resize(self.size, self.interpolation)
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        return img


class NormalizePAD(object):

    def __init__(self, max_size, PAD_type='right'):
        self.toTensor = transforms.ToTensor()
        self.max_size = max_size
        self.max_width_half = math.floor(max_size[2] / 2)
        self.PAD_type = PAD_type

    def __call__(self, img):
        img = self.toTensor(img)
        img.sub_(0.5).div_(0.5)
        c, h, w = img.size()
        Pad_img = torch.FloatTensor(*self.max_size).fill_(0)
        Pad_img[:, :, :w] = img  # right pad
        if self.max_size[2] != w:  # add border Pad
            Pad_img[:, :, w:] = img[:, :, w - 1].unsqueeze(2).expand(c, h, self.max_size[2] - w)
        return Pad_img


class AlignCollate(object):

    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, batch):
        batch = filter(lambda x: x is not None, batch)
        try:
            images, labels = zip(*batch)
        except:
            print(len(list(batch)))
            raise ''

        wlist = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            wlist.append(math.ceil(self.imgH * ratio))
        
        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = max(wlist)
            resized_max_w = resized_max_w if resized_max_w <= self.imgW else self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1

            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for i,image in enumerate(images):
                
                if wlist[i] > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = wlist[i]

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors, labels

class AlignCollateForInfer:
    def __init__(self, imgH=32, imgW=100, keep_ratio_with_pad=False):
        self.imgH = imgH
        self.imgW = imgW
        self.keep_ratio_with_pad = keep_ratio_with_pad

    def __call__(self, images):
    
        wlist = []
        for image in images:
            w, h = image.size
            ratio = w / float(h)
            wlist.append(math.ceil(self.imgH * ratio))
        
        if self.keep_ratio_with_pad:  # same concept with 'Rosetta' paper
            resized_max_w = max(wlist)
            resized_max_w = resized_max_w if resized_max_w <= self.imgW else self.imgW
            input_channel = 3 if images[0].mode == 'RGB' else 1

            transform = NormalizePAD((input_channel, self.imgH, resized_max_w))

            resized_images = []
            for i,image in enumerate(images):
                
                if wlist[i] > self.imgW:
                    resized_w = self.imgW
                else:
                    resized_w = wlist[i]

                resized_image = image.resize((resized_w, self.imgH), Image.BICUBIC)
                resized_images.append(transform(resized_image))
                # resized_image.save('./image_test/%d_test.jpg' % w)

            image_tensors = torch.cat([t.unsqueeze(0) for t in resized_images], 0)

        else:
            transform = ResizeNormalize((self.imgW, self.imgH))
            image_tensors = [transform(image) for image in images]
            image_tensors = torch.cat([t.unsqueeze(0) for t in image_tensors], 0)

        return image_tensors


def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor.cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(imtype)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


if __name__ == "__main__":

    # train_dataset = TextRecognition(4068*100)
    # AlignCollate_valid = AlignCollate(imgH=48, imgW=1024, keep_ratio_with_pad=True)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=32,
    #     shuffle=True,  # 'True' to check training progress with validation function.
    #     num_workers=int(4),
    #     collate_fn=AlignCollate_valid)
    # for i,(img,lables) in enumerate(train_loader):
    #     for j,label in  enumerate( lables):
    #         if " " in label or len(label)> 89:
    #             # print(lables)
    #             print(label,len(label),j)
    #     print("{:5}".format(i))
    #     if i > 1000:
    #         break
    ##################################################
    # train_dataset = TalOcrChnDataset("/home/ldl/桌面/论文/文本识别/data/TAL_OCR_CHN手写中文数据集/test_64")
    # train_dataset = TalOcrEngDataset("/home/ldl/桌面/论文/文本识别/data/TAL_OCR_ENG手写英文数据集/data_composition",
    #     "/home/ldl/桌面/论文/文本识别/data/TAL_OCR_ENG手写英文数据集/label_test.txt")
    # print(len(train_dataset))
    # for i,(img,label) in enumerate( train_dataset):
    #     print(label)
    #     img.save(f"/home/ldl/桌面/out/{label}.jpg")
    #     if i >= 100:
    #         break 
    ############################
    train_dataset = mytrdg_cutimg_dataset(total_img_path='/home/ldl/桌面/论文/文本识别/data/finish_data/eng_image/img',
        annotation_path='/home/ldl/桌面/论文/文本识别/data/finish_data/eng_image/gt')
    print(len(train_dataset))
    for i, (img,label) in enumerate(train_dataset):
        print(label)
        # print(f'{i:07}',end='/r')
        img.save(f'/home/ldl/桌面/out/{i}.jpg')
        if i >= 1000:
            break
    ######################
    # dataset_ICDAR2019_ArT = PpocrDataset("/home/ldl/桌面/论文/文本识别/data/paddleocr/ICDAR2019-ArT",
    #     "/home/ldl/桌面/论文/文本识别/data/paddleocr/ICDAR2019-ArT.txt",50029)
    # dataset_rctw = PpocrDataset("/home/ldl/桌面/论文/文本识别/data/paddleocr/icdar2017rctw_train_v1.2",
    #     "/home/ldl/桌面/论文/文本识别/data/paddleocr/icdar2017rctw_train_v1.2.txt",46739)
    # dataset_ICDAR2019 = PpocrDataset("/home/ldl/桌面/论文/文本识别/data/paddleocr/ICDAR2019-LSVT",
    #     "/home/ldl/桌面/论文/文本识别/data/paddleocr/ICDAR2019-LSVT.txt",240047)
    # dataset_mtwi = PpocrDataset("/home/ldl/桌面/论文/文本识别/data/paddleocr/mtwi_2018",
    #     "/home/ldl/桌面/论文/文本识别/data/paddleocr/mtwi_2018.txt",144202)
    # dataset_chn = PpocrDataset("/home/ldl/桌面/论文/文本识别/data/paddleocr/中文街景文字识别",
    #     "/home/ldl/桌面/论文/文本识别/data/paddleocr/中文街景文字识别.txt",212023)
    # dataset_Synthetic = PpocrDataset("/home/ldl/桌面/论文/文本识别/data/paddleocr/Synthetic_Chinese_String_Dataset/images",
    # "/home/ldl/桌面/论文/文本识别/data/paddleocr/Synthetic_Chinese_String_Dataset/train.txt",3279606)
    # dataset = ConcatDataset([dataset_ICDAR2019_ArT,dataset_rctw,dataset_ICDAR2019,dataset_Synthetic,
    #     dataset_chn,dataset_mtwi])
    # AlignCollate_valid = AlignCollate(imgH=32, imgW=1024, keep_ratio_with_pad=True)
    # train_loader = torch.utils.data.DataLoader(
    #     dataset_ICDAR2019_ArT, batch_size=4,
    #     shuffle=True,  # 'True' to check training progress with validation function.
    #     num_workers=int(1),
    #     collate_fn=AlignCollate_valid)
    # for image_tensors, labels in train_loader:
    #     print(image_tensors.shape)
    #     print(labels)
    #     break
    ######################################
    # print(len(dataset))
    # labelmaxlength = 0
    # maxwidth = 0
    # s = ''
    # for i, batch in enumerate(dataset_mtwi):
    #     if batch :
    #         image,lalel = batch
    #     else:
    #         continue
    #     W,H = image.size
    #     resizeW = W / H *32
    #     if resizeW > maxwidth:
    #         maxwidth = resizeW
    #     if len(lalel) > labelmaxlength:
    #         labelmaxlength = len(lalel)
    #         s = lalel
    #     print(f'{i:09}',end="\r")

    # print(labelmaxlength,maxwidth,s)
    ################################