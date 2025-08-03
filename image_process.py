# import glob
# import math
# import os
# import re
# import shutil
# import subprocess
# import sys
# import time

# import cv2
# import matplotlib.pyplot as plt
# import numpy as np
# import PIL.Image as Image
# import torch
# import torch.nn as nn
# import torchvision.models as models
# from torchvision import transforms
# from torchvision.io import read_image
# from torchvision.transforms import functional as TF
# from enum import Enum
# from models.resunet import create_resunet
import os
import math
import logging
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
from models.resunet import create_resunet
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class ImageMode(Enum):
    RGB = 0
    BGR = 1
    GRAY = 2

class ImageProcessor():

    def __init__(self, args):
        self.logger = logging.getLogger(__name__)

        if(args is not None):
            self.op = args.op
            self.origin_data_path = args.oip
            self.segmentation_path = args.sip
            self.stride = args.stride
            self.sample_size = 200
            self.des = args.srcdes

        self.HOUGH_PARAMS = {
            'dp': 1,
            'min_dist': 200,
            'param1': 20,
            'param2': 20,
            'min_radius': 1350,
            'max_radius': 1430
        }

        self.ADAPTIVE_THRESH_PARAMS = {
            'maxValue': 255,
            'adaptiveMethod': cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            'thresholdType': cv2.THRESH_BINARY,
            'blockSize': 5,
            'C': 2
        }

    @staticmethod
    def open_img(path: Union[str, Path], mode: ImageMode = ImageMode.GRAY) -> np.ndarray:
        """ open image
        Args:
            path: image path
            mode: mode of opening image
            
        Returns:
            numpy array
            
        Raises:
            FileNotFoundError
            ValueError
        """      
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Image file not found: {path}")
            
        if path.is_dir():
            raise ValueError(f"Path is a directory, not a file: {path}")
            
        try:
            img = cv2.imread(str(path))
            if img is None:
                raise ValueError(f"Cannot read image file: {path}")
                
            if mode == ImageMode.RGB:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif mode == ImageMode.BGR:
                return img
            else:  # GRAY
                return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
        except Exception as e:
            raise ValueError(f"Error reading image {path}: {e}")
        
    @staticmethod        
    def write_img(img: np.ndarray, destination: Union[str, Path]) -> bool:
        """save image array
        
        Args:
            img: image numpy array
            destination: saving path
            
        Returns:
            Status of svaing image
        """
        try:
            destination = Path(destination)
            destination.parent.mkdir(parents=True, exist_ok=True) 
            return cv2.imwrite(str(destination), img)
        except Exception as e:
            logging.error(f"Error writing image to {destination}: {e}")
            return False
    @staticmethod
    def list_folder(path):
        if(os.path.isdir(path)):
            return os.listdir(path)
        else:
            raise NotADirectoryError(f"Error: The path '{path}' is not a directory")
        
    def hough_circle(self, img: np.ndarray, **kwargs) -> Optional[np.ndarray]:
        """hough circle
        
        Args:
            img: image(gray)
            **kwargs: hough parameters
            
        Returns:
            np array or None if not find any hough circle
        """
        if len(img.shape) != 2:
            raise ValueError("Input image must be grayscale")
            
    
        params = {**self.HOUGH_PARAMS, **kwargs}
        
        circles = cv2.HoughCircles(
            img, 
            cv2.HOUGH_GRADIENT,
            dp=params['dp'],
            minDist=params['min_dist'],
            param1=params['param1'],
            param2=params['param2'],
            minRadius=params['min_radius'],
            maxRadius=params['max_radius']
        )
        
        return circles
   
    def image_process(self, seg_image: np.ndarray) -> np.ndarray:
        """image post-process
        
        Args:
            seg_image: image after segmentation
            
        Returns:
            binary image after processing
        """
        if seg_image is None or not isinstance(seg_image, np.ndarray):
            raise ValueError("seg_image must be a valid numpy array")
        
        binary_image = cv2.adaptiveThreshold(
            seg_image,
            **self.ADAPTIVE_THRESH_PARAMS
        )
        
        blurred = cv2.medianBlur(binary_image, 1)
        
        return blurred
    
    @staticmethod
    def parse_circle(x, y, r, angle):
        arc = math.pi * angle / 180
        r = r*1
        return x + r * math.cos(arc),y + r * math.sin(arc)
    
    @staticmethod
    def xywh2xyxy(x, y, w, h):
        return (int(x - w/2),
                int(y-h/2),
                int(x + w/2),
                int(y+h/2))
    
    def label(self, path):
        assert self.stride % 2 == 0
        
        label_dir = "label"
        label_file_path = os.path.join(label_dir, path)
        if os.path.exists(label_file_path):
            raise FileExistsError(
                f"Label file '{label_file_path}' already exists. Please delete it or choose a different label file name."
            )
        origin_img_paths = self.list_folder(self.origin_data_path)
        segmentation_paths = self.list_folder(self.segmentation_path) if self.segmentation_path != ' ' else []
        with open(label_file_path, "w") as f:
            for _o_path, _s_path in zip(origin_img_paths, segmentation_paths):
                o_path = os.path.join(self.origin_data_path, _o_path)
                s_path = os.path.join(self.segmentation_path, _s_path)
                origin_img = self.open_img(path = o_path, mode = ImageMode.GRAY)
                copy_origin = self.open_img(os.path.join(self.origin_data_path, _s_path), mode = ImageMode.GRAY)
                if s_path:
                    segmentation = self.open_img(path = s_path, mode = ImageMode.GRAY)
                    hc = self.houghCircle(img=segmentation)
                else:
                    raise AssertionError("No resUNet, you can input segmentation")

                if hc is not None:
                    print(f"Note: generate {len(hc)} available Hough circles")
                    xc, yc, r = hc[0][0]  # Fetch the first circle info
                    w = h = self.sample_size

                    for idx, angle in enumerate(range(0, 360, self.stride)):
                        c = self.parse_circle(xc, yc, r, angle)
                        xmin, ymin, xmax, ymax = self.xywh2xyxy(c[0], c[1], w=w, h=h)
                        assert (xmax - xmin == w and ymax - ymin == h), "Wrong sample size"
                        origin_img_ = cv2.rectangle(
                            origin_img, (xmin, ymin), (xmax, ymax), 255, thickness=5
                        )
                        cv2.namedWindow('Image', 0)
                        cv2.resizeWindow('Image', int(5472 / 3), int(3642 / 3))
                        cv2.imshow('Image', origin_img_)
                        key = cv2.waitKey(0)
                        origin_img = copy_origin.copy()
                        na = f"{_o_path.split('.')[0]}_{idx}"

                        if key == ord('1'):
                            label = 1
                        elif key == ord('0'):
                            label = 0
                        else:
                            label = -1

                        f.write(f"{na}.jpg {label}\n")
                        f.flush()
                        cv2.imwrite(f"{self.des}/{na}.jpg", copy_origin[ymin:ymax, xmin:xmax])
                    else:
                        print("Warning: Not find Hough circle")
        cv2.destroyAllWindows()
    def segmentation(self):
        network = create_resunet()
    
        for root, dirs, files in os.walk(self.origin_data_path):
            for file in files:
                origin = self.open_img(os.path.join(root, file), mode=ImageMode.BGR)
                origin_= self.open_img(os.path.join(root, file))
                seg_image_ = network.pre_processing(origin)
                seg_image = network.test(images=seg_image_, originImg=origin_)
                print(f"Generating '{file}' segmentation...")
                self.write_img(seg_image,os.path.join(self.segmentation_path,file))
            seg_image = self.image_process(np.uint8(seg_image))
