import os

import torch

from  model import Model
from utils import CTCLabelConverter
from dataset import AlignCollateForInfer 

class OcrTextRecognizer:
    

    def __init__(self,character,device="cpu"):
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

    def setweigth(self,weigthpath='None'):
        if os.path.exists(weigthpath):
            self.text_recognizer.load_state_dict(
                torch.load(weigthpath,map_location=self.device)
            )
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
    from PIL import Image
    imgdir = "/home/ldl/桌面/论文/文本识别/data/IIIT5K/test"
    imglist = []
    n = 0
    for imgname in os.listdir(imgdir):
        imgpath = os.path.join(imgdir,imgname)
        print(imgpath)
        imglist.append(Image.open(imgpath).convert('L'))
        n += 1
        if n == 10:
            break
    m = OcrTextRecognizer("../TextRecognitionDataGenerator/trdg/mydicts/ppocr_keys_v1.txt",device="cuda")
    m.setweigth("/home/ldl/桌面/论文/文本识别/deep_text_recognition_benchmark/saved_models/None-RCNN-BiLSTM-CTC-Seed1111/best_accuracy.pth")
    print(len(imglist))
    print(m(imglist))
