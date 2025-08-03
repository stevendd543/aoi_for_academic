import os
import torch
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.classifier_hough_detector import Detector

def main():
    config = {
        'sample_size': 200,
        'stride': 3,
        'threshold': 0.78,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'model_type': 'mobilenet_v2'  
    }

    weight_path = './weights/mobilenet_v2.pth'
    image_path = 'test_image.jpg'
    result_dir = 'results/'

    detector = Detector(config)
    print(f"Loading model weights: {config['model_type']}")
    detector.load_model(weight_path)

    result = detector.detect(image_path)
    print(f"Found {len(result.bboxes)} defects")
    print(f"Inference time: {result.inference_time:.3f}s")

    detector.save_detection_results(
        image_path,
        result,
        result_dir,
        output_name=f"{config['model_type']}_"
    )

if __name__ == "__main__":
    main()