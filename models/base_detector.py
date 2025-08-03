import os
import sys
import abc
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import torch
import torch.nn as nn
from dataclasses import dataclass
import image_process 
import math
import cv2 
from models.resunet import create_resunet
from torchvision import transforms
from models.classifier import create_network

@dataclass
class BBox:
    center_x: float
    center_y: float
    width: float
    height: float
    angle: float = 0  # 度
    confidence: float = 1.0
    class_id: int = 0
    class_name: str = "defect"

@dataclass
class DetectionResult:
    bboxes: List[BBox]
    inference_time: float
    image_path: str
    model_name: str

class BaseDetector(abc.ABC):    
    def __init__(self, config: Dict):
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model_info = None

        self.resunet = create_resunet()
        self.logger.info("ResUNet model loaded successfully")

        self.sample_size = config.get('sample_size', 200)
        self.stride = config.get('stride', 2)
        self.thr = config.get('threshold', 0.78)
        self.pi = 3.14159265359
        
        mid_index = 90 // self.stride + 1
        decreasing_part = np.linspace(1, 0, mid_index + 1)
        self.decrese_w = np.concatenate([decreasing_part[:-1], decreasing_part[::-1]])
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.41], [0.31])
        ])

        self.imp = image_process.ImageProcessor(None)

    def load_model(self, weight_path: str) -> None:
        """Load the model weights from the specified path."""
        try:
            if os.path.exists(weight_path):
                checkpoint = torch.load(weight_path, map_location=self.device, weights_only=False)
                if isinstance(checkpoint, dict):
                    self.logger.warning("Model checkpoint is dictionary format")
                    model_type = self.config.get('model_type', 'mobilenet_v2')
                    print(f"Loading model type: {model_type}")
                    self.classifier, _ , self.model_info = create_network(model_type=model_type, pretrained=False)
                    self.classifier.load_state_dict(checkpoint)
                    self.classifier.eval()
                    self.classifier.to(self.device)
                    self.logger.info(f"Classifier weights loaded from {weight_path}")
                else:
                    self.classifier = checkpoint
                    self.classifier.eval()
                    self.classifier.to(self.device)
                    self.logger.info(f"Classifier model loaded from {weight_path}")
            else:
                raise FileNotFoundError(f"Weight file not found: {weight_path}")
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            raise
        
        @abc.abstractmethod
        def preprocess(self, image: np.ndarray) -> torch.Tensor:
            pass
    
    @abc.abstractmethod
    def inference(self, image: torch.Tensor) -> List[BBox]:
        pass
    
    @abc.abstractmethod
    def postprocess(self, raw_output) -> List[BBox]:
        pass

    def _detect_defects_on_image(self, image_path: str) -> List[BBox]:
            try:
                samples, sample_positions, text_positions = self._sampler(image_path)
            except Exception as e:
                self.logger.error(f"Failed to sample from image {image_path}: {e}")
                return []


            
            tensor_rot90 = torch.rot90(samples, k=1, dims=(2, 3))
            tensor_rot180 = torch.rot90(samples, k=2, dims=(2, 3))
            tensor_rot270 = torch.rot90(samples, k=3, dims=(2, 3))
            augmented_samples = torch.cat((samples, tensor_rot90, tensor_rot180, tensor_rot270), 0)
            
            with torch.no_grad():
                try:
                    outputs = self.classifier(augmented_samples.to(self.device))
                except Exception as e:
                    self.logger.error(f"Model inference failed: {e}")
                    return []
            
            outputs = outputs.detach().cpu().numpy()
            sub_outputs = np.split(outputs, 4)
            outputs = np.sum(sub_outputs, axis=0) / 4
            output_probs = self._sigmoid(outputs)
            
            if output_probs.shape[1] > 1:
                output_probs = output_probs[:, 1] 
            else:
                output_probs = output_probs.flatten()
            
            defect_indices = np.where(output_probs > self.thr)[0]
            
            bboxes = []
            for idx in defect_indices:
                if idx < len(sample_positions):
                    bbox = self._create_bbox(
                        sample_positions[idx],
                        text_positions[idx] if idx < len(text_positions) else None,
                        output_probs[idx],
                        idx
                    )
                    if bbox:
                        bboxes.append(bbox)
            
            self.logger.info(f"Detected {len(bboxes)} defects in {image_path}")
            return bboxes

    def _sampler(self, image_path: str) -> Tuple[Optional[torch.Tensor], List, List]:

        for_seg = self.imp.open_img(image_path, mode=image_process.ImageMode.BGR)
        origin = self.imp.open_img(image_path)  # gray mode for original image
 
        seg_image_ = self.resunet.pre_processing(for_seg)
        seg_image = self.resunet.test(images=seg_image_, originImg=origin)
        seg_image = self.imp.image_process(seg_image = np.uint8(seg_image))
    
        hc = self._get_hough_circles(seg_image, image_path)
        if hc is None:
            return None, None, None
        
        samples = []
        positions = []
        text_pos = []
        
     
        xc, yc, r = hc[0][0]  
        p = 0
        
        for idx, angle in enumerate(range(0, 360, self.stride)):  
            arc = self.pi * angle / 180
            
            if 'B' in os.path.basename(image_path):
                r_adjusted = self._adjust_radius_for_b_type(r, idx, p)
                if r_adjusted != r:
                    r = r_adjusted
                    p += 1
            
            sx = int(xc + r * math.cos(arc))
            sy = int(yc + r * math.sin(arc))
            
            x_min = int(sx - self.sample_size / 2)
            y_min = int(sy - self.sample_size / 2)
            x_max = int(sx + self.sample_size / 2)
            y_max = int(sy + self.sample_size / 2)
            
            x_outside, y_outside = self._get_aux_point(1.2 * r, xc, yc, arc)
            
            if x_max > origin.shape[1] or y_max > origin.shape[0]:
                origin = np.pad(origin, pad_width=200, mode='constant', constant_values=0)
            
            if (x_min < 0 or y_min < 0 or x_max < 0 or y_max < 0 or 
                x_max > origin.shape[1] or y_max > origin.shape[0]):
                self.logger.warning("Sample out of range, skipping")
                continue
            
            sample = origin[y_min:y_max, x_min:x_max]
            
            if sample.shape[0] == self.sample_size and sample.shape[1] == self.sample_size:
                sample_tensor = self.transform(sample)
                samples.append(sample_tensor)
                positions.append([[x_min, x_max, y_min, y_max]])
                text_pos.append([x_outside, y_outside])
        
        if samples:
            data = torch.stack(samples, dim=0)
            return data, positions, text_pos
        else:
            return None, None, None
    @abc.abstractmethod    
    def detect(self, image_path: str) -> DetectionResult:
        pass

    def visualize_detection_result(self, image_path: str, result: DetectionResult, 
                                output_path: str) -> None:
        try:
            image = self.imp.open_img(image_path, mode=image_process.ImageMode.BGR)
            
            for bbox in result.bboxes:
                x_min = int(bbox.center_x - bbox.width / 2)
                y_min = int(bbox.center_y - bbox.height / 2)
                x_max = int(bbox.center_x + bbox.width / 2)
                y_max = int(bbox.center_y + bbox.height / 2)
                
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
                
                confidence_text = f"{int(bbox.confidence * 100)}"
                text_x = int(bbox.center_x + bbox.width / 2 * 1.2)
                text_y = int(bbox.center_y)
                
                cv2.putText(image, confidence_text, (text_x, text_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            
            output_dir = Path(output_path).parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.imp.write_img(image, output_path)
            self.logger.info(f"Visualization saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to visualize result: {e}")

    def save_detection_results(self, image_path: str, result: DetectionResult, 
                             destination: str, output_name: str) -> None:
        try:
            filename = os.path.basename(image_path)
            output_name = output_name + filename if output_name else filename
            

            if not output_name:
                logging.warning("Output name is empty, using default filename")
                output_name = filename
            
            has_defects = any(bbox.confidence >= self.thr for bbox in result.bboxes)
            
            if has_defects:
                coating_dir = os.path.join(destination, "coating")
                os.makedirs(coating_dir, exist_ok=True)
                output_path = os.path.join(coating_dir, output_name)

            self.visualize_detection_result(image_path, result, filename)
            self.logger.info(f"Results saved to {'coating/' if has_defects else ''}{filename}")
            
            txt_output = output_path + ".txt"

            with open(txt_output, "w", encoding="utf-8") as f:
                if self.model_info:
                    f.write(self.model_info)
                    f.write("\n")
                f.write(f"Image: {result.image_path}\n")
                f.write(f"Inference time: {result.inference_time:.3f}s\n")
                f.write(f"Found {len(result.bboxes)} defects\n")

            self.logger.info(f"Detection info saved to {txt_output}")
        except Exception as e:
            self.logger.error(f"Failed to save detection results: {e}")

    def _get_hough_circles(self, seg_image: np.ndarray, image_path: str) -> Optional[np.ndarray]:
        if 'A' in os.path.basename(image_path):
            return self.imp.hough_circle(img=seg_image)
        elif 'B' in os.path.basename(image_path):
            return self.imp.hough_circle(img=seg_image, min_r=1420, max_r=1500)
        else:
            self.logger.warning("Unknown image type, using default parameters")
            return self.imp.hough_circle(img=seg_image)
    
    def _adjust_radius_for_b_type(self, r: float, idx: int, p: int) -> float:
        offset = 90 // self.stride
        if idx <= offset * 1 or idx >= offset * 3:  # 1/4 或 3/4 位置
            return r - 60 * self.decrese_w[p] if p < len(self.decrese_w) else r
        return r
    
    @staticmethod
    def _get_aux_point(radius: float, xc: float, yc: float, arc: float) -> Tuple[int, int]:
        ax = xc + radius * math.cos(arc)
        ay = yc + radius * math.sin(arc)
        return int(ax), int(ay)
    
    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def _create_bbox(self, position: List, text_pos: Optional[List], 
                           confidence: float, angle_idx: int) -> Optional[BBox]:
        try:
            x_min, x_max, y_min, y_max = position[0]
            
            center_x = (x_min + x_max) / 2
            center_y = (y_min + y_max) / 2
            
            width = x_max - x_min
            height = y_max - y_min
            
            angle = (angle_idx * self.stride) % 360  
            
            return BBox(
                center_x=center_x,
                center_y=center_y,
                width=width,
                height=height,
                angle=angle,
                confidence=confidence,
                class_id=0,
                class_name="defect"
            )
            
        except Exception as e:
            self.logger.error(f"Failed to create bbox: {e}")
            return None