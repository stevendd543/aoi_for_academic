import os
import math
import time 

import cv2
import torch
import torch.nn as nn
import numpy as np

import argparse
import pathlib


from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
from torchvision import transforms
from tqdm import tqdm

from models.resunet import create_resunet
import image_process

sample_size = 200
pi = 3.14159265359
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.41],[0.31])
])

class Detector():
    def __init__(self, args, seg_model):
        self.args = args
        self.resunet = seg_model
        self.thr = args.thr
        self.stride = args.stride
        mid_index = 90//self.stride +1
        decreasing_part = np.linspace(1, 0, mid_index + 1)
        self.decrese_w = np.concatenate([decreasing_part[:-1],decreasing_part[::-1]])

        if(self.args.gpu):
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.imp = image_process.ImageProcessor(None)

    def __load_weight(self, ckp = None):
        checkpoint = torch.load(ckp, map_location = self.device)
        
        if(isinstance(checkpoint, dict)):
            print("Loader warnning, your model is dictionary not weight")
        else:
            print("Weights loading ... ")
        checkpoint.eval()
        print("Evaluation mode is on")
        return checkpoint

    def create_system(self):
        weight_path = self.args.detector
        self.model = self.__load_weight(ckp = weight_path).to(self.device)
        self.model.eval()
    @staticmethod    
    def get_aux_point(radius,xc,yc,arc):
        ax = xc + radius * math.cos(arc)
        ay = yc + radius * math.sin(arc)
        return int(ax),int(ay)
        
    @staticmethod
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)
        
    def sampler(self,path):
        origin = self.imp.open_img(path, mode=image_process.ImageMode.BGR)
    
        origin_= self.imp.open_img(path)
        seg_image_ = self.resunet.pre_processing(origin)
        seg_image = self.resunet.test(images=seg_image_, originImg=origin_)
        seg_image = self.imp.image_process(np.uint8(seg_image))
        hc = self.imp.houghCircle(img=seg_image)
        if 'A' in os.path.basename(path):
            hc = self.imp.houghCircle(img=seg_image)
        elif 'B' in os.path.basename(path):
            hc = self.imp.houghCircle(img=seg_image,min_r=1420,max_r=1500)
        else:
            print("Failed to sample")
            return None,None,None
        samples = [] # collection of sample from CRS
        positions = [] # coordinate of sample
        text_pos = [] # coodinate of probability

        if hc is not None:
            # print(f"Note: generate {len(hc)} available Hough circles")
            p  = 0
            for idx,angle in enumerate(range(0,360,3)):
                xc, yc, r = hc[0][0]  # Fetch the first circle info
                arc = pi * angle / 180

                if 'B' in os.path.basename(path):  # regulate right side of B-image
                    offset = 90 / self.stride # split into 4 parts : 360/self.stride/4
                    if idx <= offset*1: # 1/4
                        r = r - 60* self.decrese_w[p]
                        p += 1
                    if idx >= offset*3: # 3/4
                        r = r - 60* self.decrese_w[p]
                        p += 1
                # coordinates of sample and proability   
                sx = int(xc + r * math.cos(arc)) # center of sample
                sy = int(yc + r * math.sin(arc)) 
                x_min,y_min,x_max,y_max = int(sx - sample_size/2),int(sy-sample_size/2),int(sx + sample_size/2),int(sy+sample_size/2) # corner of sample
                x_outside, y_outside = self.get_aux_point(radius = 1.2*r ,xc=xc ,yc=yc, arc=arc)
                if x_max>origin_.shape[1] or y_max>origin_.shape[0]:
                    origin_ = np.pad(origin_, pad_width=200,mode='constant',constant_values=0)
                if x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0:
                    print("sample out of range")
                    return None,None,None
                if x_max <= origin_.shape[1] and y_max <= origin_.shape[0]:
                    sample = origin_[y_min:y_max,x_min:x_max]
                else:
                    return None,None,None
                sample = transform(sample)
                if(sample.size(1)==sample_size and sample.size(2)==sample_size):
                    samples.append(sample)
                positions.append([[x_min,x_max,y_min,y_max]])
                text_pos.append([x_outside, y_outside])
                data = torch.stack(samples, dim=0)
            return data, positions, text_pos
        else:
            print("Failed to sample")
            return None,None,None
    
    @staticmethod
    def sigmoid(x):
        s = 1/(1+np.exp(-1*x))
        return s

    def detect_and_save(self,src_path,filename,destination):
        strat = time.time()
        image_path = os.path.join(src_path,filename)
        samples,sample_positions,text_position = self.sampler(image_path)
        if samples is not None:
            tensor_rot90 = torch.rot90(samples,k=1,dims=(2,3))
            tensor_rot180 = torch.rot90(samples,k=2,dims=(2,3))            
            tensor_rot270 = torch.rot90(samples,k=3,dims=(2,3))
            samples = torch.cat((samples,tensor_rot90,tensor_rot180,tensor_rot270),0)
            with torch.no_grad():
                outputs = self.model(samples.to(self.device))
            outputs = outputs.detach().cpu().numpy()
            sub_outputs = np.split(outputs,4)
            outputs = np.sum(sub_outputs,axis=0)/4
            output = self.sigmoid(outputs)
            output = output[:,1]
            # filter
            # left = np.roll(output,-1)
            # right = np.roll(output,1)
            # big_neighbor = np.where(left > right, left, right)
            # small_neighbor = np.where(left > right, right, left)
            # output = np.multiply(small_neighbor,output)   
            # output = np.sqrt(output)           
            text_info = []
            defect_index = np.where(output>self.thr)
        
            selected_position=[sample_positions[index] for index in defect_index[0]]
            for index in defect_index[0]:
                t = text_position[index]
                t.append(output[index])
                text_info.append(t)
            origin = self.imp.open_img(image_path, mode=image_process.ImageMode.BGR)
            for p in selected_position:
                xmin,xmax,ymin,ymax = p[0]
                origin = cv2.rectangle(
                    origin, (xmin, ymin), (xmax, ymax), (0,255,0), thickness=5
                )
            find = False
            for idx,(x,y,p) in enumerate(text_info):
                cv2.putText(origin,str(int(100*p)),(x,y),color=(255,255,0),fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,thickness=2)
                if int(100*p)>=self.thr:
                    find = True
            if find:
                if not os.path.exists(os.path.join(destination,"coating")):
                    os.mkdir(os.path.join(destination,"coating"))
                self.imp.write_img(origin,os.path.join(destination,"coating",filename))
            else:
                self.imp.write_img(origin,os.path.join(destination,filename))
            end = time.time()
            print(f"Execution time : {round(end-strat,2)}")
