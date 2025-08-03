import os
import sys
import math
import time
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score


from models.base_detector import BaseDetector, BBox, DetectionResult
import image_process

class Detector(BaseDetector):
    def __init__(self, config: Dict):
        super().__init__(config)
        self.model_name = config.get('model_type', 'Unknown')
        self.classifier = None
        self.logger = logging.getLogger(self.__class__.__name__)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image

    def inference(self, image_path: str) -> List[BBox]:
        return self._detect_defects_on_image(image_path)

    def postprocess(self, raw_output) -> List[BBox]:
        return raw_output

    def detect(self, image_path: str) -> DetectionResult:
        start_time = time.time()
        try:
            bboxes = self.inference(image_path)
            inference_time = time.time() - start_time
            return DetectionResult(
                bboxes=bboxes,
                inference_time=inference_time,
                image_path=image_path,
                model_name=self.model_name
            )
        except Exception as e:
            self.logger.error(f"Detection failed for {image_path}: {e}")
            return DetectionResult(
                bboxes=[],
                inference_time=time.time() - start_time,
                image_path=image_path,
                model_name=self.model_name
            )

if __name__ == "__main__":
    config = {
        'sample_size': 200,
        'stride': 3,
        'threshold': 0.78,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_type': None
    }

    weights_name = [
        'resnet34',
        'efficientnet_b0',
        'mobilenet_v2',
        'mobilenet_v3_small',
        'mobilenet_v3_large',
        'shufflenet_v2_x0_5',
        'shufflenet_v2_x1_0',
    ]

    for model in weights_name:
        config['model_type'] = model
        detector = Detector(config)
        print(f"Loading model weights: {model}")
        detector.load_model(f'./weights/{model}.pth')

        result = detector.detect('test_image.jpg')
        print(f"found {len(result.bboxes)} defects")
        print(f"inference_time: {result.inference_time:.3f}s")

        detector.save_detection_results(
            'test_image.jpg',
            result,
            'results/',
            output_name=f"{model}_"
        )