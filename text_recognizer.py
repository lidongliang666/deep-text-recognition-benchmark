import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))

print(str(__dir__.parent))
sys.path.insert(3,str(__dir__.parent))

import torch
import cv2
from PIL import Image
import numpy as np

from  model import Model
from recognizer_utils import CTCLabelConverter
from dataset import AlignCollateForInfer 



class OcrTextRecognizer:
    

    def __init__(self,character,weigthpath,device="cpu"):
        '''
        character : 预测字符集字符串("abcde.....xyz")或者文件
        '''

        self.converter = CTCLabelConverter(character)
        config = model_config()
        config.num_class = len(self.converter.character)
        self.text_recognizer = Model(config)
        self.text_recognizer.to(device)
        # print(next(self.text_recognizer.parameters()).device)
        self.batch_size = 8 
        self.aligncollate = AlignCollateForInfer(imgH=32,imgW=1024,keep_ratio_with_pad=True)
        self.device = device
        self.setweigth(weigthpath)

    def setweigth(self,weigthpath='None'):
        if os.path.exists(weigthpath):
            self.text_recognizer.load_state_dict(
                torch.load(weigthpath,map_location=self.device)
            )
            self.text_recognizer.eval()
            print(f"load weigth file:{weigthpath}")
        else:
            print(f"not exists file{weigthpath}")
    def __call__(self,imgcut_list):
        '''
        imgcut_list 是PIL Image list类型
        '''
        img_num = len(imgcut_list)
        #数据处理，组装为batch，推理，解码
        result_list = []
        for startId in range(0,img_num,self.batch_size):
            endId = startId + self.batch_size
            img_tensor = self.aligncollate(imgcut_list[startId:endId])
            img_tensor = img_tensor.to(self.device)
            # print(img_tensor.shape)
            # print(img_tensor.device)
            with torch.no_grad():
                preds = self.text_recognizer(img_tensor,text=None)
                _, preds_index = preds.max(2)
                preds_str = self.converter.decode_for_predict(preds_index.data)
                result_list += preds_str


        return result_list

    def predict(self,imgpath,boxes_list):
        img = cv2.imread(imgpath,flags=cv2.IMREAD_GRAYSCALE)
        pilimglist = self.get_pilimglist(img,boxes_list)
        txtlist = self.__call__(pilimglist)
        return txtlist

    def get_pilimglist(self,imgcopy,bboxlist):
        #过滤那些 竖着的框（高比宽大）
        pilimglist = []
        for _,bbox in enumerate(bboxlist):
            imgcut = self.get_rotate_crop_image(imgcopy,bbox)
            if imgcut is None:
                continue
            #图片裁剪物可视化
            # cv2.imwrite(f"{i}.jpg",imgcut)
            pilimglist.append(imgcut)
        return pilimglist

    def get_rotate_crop_image(self, img, points):
        '''
        img_height, img_width = img.shape[0:2]
        left = int(np.min(points[:, 0]))
        right = int(np.max(points[:, 0]))
        top = int(np.min(points[:, 1]))
        bottom = int(np.max(points[:, 1]))
        img_crop = img[top:bottom, left:right, :].copy()
        points[:, 0] = points[:, 0] - left
        points[:, 1] = points[:, 1] - top
        '''
        points = np.array(points,dtype="float32")
        img_crop_width = int(
            max(
                np.linalg.norm(points[0] - points[1]),
                np.linalg.norm(points[2] - points[3])))
        img_crop_height = int(
            max(
                np.linalg.norm(points[0] - points[3]),
                np.linalg.norm(points[1] - points[2])))
        if img_crop_height > img_crop_width:
            return None
        pts_std = np.float32([[0, 0], [img_crop_width, 0],
                              [img_crop_width, img_crop_height],
                              [0, img_crop_height]],dtype="float32")
        M = cv2.getPerspectiveTransform(points, pts_std)
        dst_img = cv2.warpPerspective(
            img,
            M, (img_crop_width, img_crop_height),
            borderMode=cv2.BORDER_REPLICATE,
            flags=cv2.INTER_CUBIC)
        # dst_img_height, dst_img_width = dst_img.shape[0:2]
        # if dst_img_height * 1.0 / dst_img_width >= 1.5:
        #     dst_img = np.rot90(dst_img)
        pilimg = Image.fromarray(dst_img).convert("L")
        return pilimg





class model_config:
    Transformation ="None"
    FeatureExtraction="RCNN"
    SequenceModeling="BiLSTM"
    Prediction="CTC"
    input_channel=1
    output_channel=512
    hidden_size=96

if __name__ == "__main__":
    # a =OcrTextRecognizer("../TextRecognitionDataGenerator/trdg/mydicts/ppocr_keys_v1.txt")
    # print(a.text_recognizer)
    ################
    # from PIL import Image
    # imgdir = "/home/ldl/桌面/论文/文本识别/data/IIIT5K/test"
    # imglist = []
    # n = 0
    # for imgname in os.listdir(imgdir):
    #     imgpath = os.path.join(imgdir,imgname)
    #     print(imgpath)
    #     imglist.append(Image.open(imgpath).convert('L'))
    #     n += 1
    #     if n == 10:
    #         break
    m = OcrTextRecognizer("../TextRecognitionDataGenerator/trdg/mydicts/ppocr_keys_v1.txt",
        weigthpath="/home/ldl/桌面/论文/文本识别/deep_text_recognition_benchmark/saved_models/None-RCNN-BiLSTM-CTC-Seed1111/best_accuracy_recognizer.pth"
        ,device="cuda")
    # model_cp = "/home/ldl/桌面/论文/文本识别/deep_text_recognition_benchmark/saved_models/None-RCNN-BiLSTM-CTC-Seed1111/best_accuracy_recognizer.pth"
    # torch.save(m.text_recognizer.state_dict(), model_cp,_use_new_zipfile_serialization=False)
