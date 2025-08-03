from .base_detector import BaseRotatedDetector,RotatedBBox
from typing import List, Dict, Tuple, Optional, Union
import torch
import torch.nn as nn
import numpy as np

class OrientedRCNNDetector(BaseRotatedDetector):    
    def __init__(self, config: Dict):
        super().__init__(config) #?????????????????????????????
        self.model_name = "Oriented-RCNN"
        
    def load_model(self, weight_path: str) -> None:
        """載入 Oriented R-CNN 模型"""
        try:
            # 這裡需要實際的 Oriented R-CNN 實現
            # 示例代碼框架
            from mmrotate.apis import init_detector
            
            config_file = self.config.get('config_file', 'oriented_rcnn_r50_fpn.py')
            self.model = init_detector(config_file, weight_path, device=self.device)
            self.logger.info(f"Loaded Oriented R-CNN model from {weight_path}")
            
        except ImportError:
            self.logger.warning("MMRotate not installed. Using dummy model.")
            self.model = self._create_dummy_model()
    
    def _create_dummy_model(self):
        """創建虛擬模型用於測試"""
        class DummyOrientedRCNN(nn.Module):
            def forward(self, x):
                batch_size = x.shape[0]
                # 返回虛擬檢測結果
                return torch.rand(batch_size, 100, 6)  # [x, y, w, h, angle, score]
        
        return DummyOrientedRCNN()
    
    def preprocess(self, image: np.ndarray) -> torch.Tensor:
        import cv2
        from torchvision import transforms
        
        # 調整圖像大小
        target_size = self.config.get('input_size', (1024, 1024))
        image = cv2.resize(image, target_size)
        
        # 轉換為RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 標準化
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        return transform(image).unsqueeze(0).to(self.device)
    
    def inference(self, image: torch.Tensor) -> List[RotatedBBox]:
        """Oriented R-CNN 推理"""
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(image)
        
        return self.postprocess(outputs)
    
    def postprocess(self, raw_output) -> List[RotatedBBox]:
        """Oriented R-CNN 後處理"""
        bboxes = []
        threshold = self.config.get('confidence_threshold', 0.5)
        
        # 解析模型輸出
        if isinstance(raw_output, torch.Tensor):
            detections = raw_output[0].cpu().numpy()  # 第一張圖片的結果
            
            for detection in detections:
                if len(detection) >= 6 and detection[5] > threshold:
                    bbox = RotatedBBox(
                        center_x=float(detection[0]),
                        center_y=float(detection[1]),
                        width=float(detection[2]),
                        height=float(detection[3]),
                        angle=float(detection[4]),
                        confidence=float(detection[5]),
                        class_id=0,
                        class_name="defect"
                    )
                    bboxes.append(bbox)
        
        return bboxes