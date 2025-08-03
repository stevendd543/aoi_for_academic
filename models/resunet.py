import torch
import torch.nn as nn
import cv2
import numpy as np
from models.resnet import resnet34
from models.decoder import Decorder

class ResUnet(nn.Module):

    def __init__(self, heads, pretrained, down_ratio, final_kernel, head_conv):
        super(ResUnet, self).__init__()
        assert down_ratio in [1, 2, 4, 8, 16]
        channels = [32, 64, 64, 128, 256, 512]
        self.l1 = int(np.log2(down_ratio))
        self.base_network = resnet34(pretrained=pretrained)
        self.dec_net = Decorder(heads, final_kernel, head_conv, channels[self.l1])


    def forward(self, x):
        x = self.base_network(x)
        dec_dict = self.dec_net(x)
        return dec_dict
    
class create_resunet(object):

    def __init__(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        heads = {'edge': 1}
        self.model = ResUnet(heads=heads,
                                pretrained=True,
                                down_ratio =1,
                                final_kernel=1,
                                head_conv=256).to(self.device)
        try:
            self.load_model(self.model, './weights/resunet.pth')
        except FileNotFoundError as e:
            print(f"Not found model weight: {e}")
        except Exception as e:
            print(f"Error : {e}")

    def load_model(self, model, resume):
        checkpoint = torch.load(resume, map_location=lambda storage, loc: storage) # load model onto device that you svae
        print('loaded weights from {}, epoch {}'.format(resume, checkpoint['epoch']))
        state_dict_ = checkpoint['state_dict']
        model.load_state_dict(state_dict_, strict = False)
        return model
    
    def map_mask_to_image(self, mask, img ,color=None):
        if color is None:
            color = np.random.rand(3)
        mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2)
        mskd = img * mask
        clmsk = np.ones(mask.shape) * mask
        clmsk[:, :, 0] = clmsk[:, :, 0] * color[0] * 256
        clmsk[:, :, 1] = clmsk[:, :, 1] * color[1] * 256
        clmsk[:, :, 2] = clmsk[:, :, 2] * color[2] * 256
        img = img + 1. * clmsk - 1. * mskd
        return np.uint8(img)
    
    def test(self, images, originImg):
        if not isinstance(images, torch.Tensor):
            raise TypeError("images should be a torch.Tensor")
        if not isinstance(originImg, np.ndarray):
            raise TypeError("originImg should be a numpy.ndarray")
        
        if not hasattr(self, 'model') or self.model is None:
            raise AttributeError(f"model is not initialized")
        
        self.model.to(self.device)
        self.model.eval()
        try:
            with torch.no_grad():
                output = self.model(images)
                edge = output['edge']
            edge = edge.squeeze().cpu().numpy()
            edge[edge>=0.5] = 1
            edge[edge<0.5] = 0
            images = images.cpu().numpy()
            edge = cv2.resize(edge, (int(originImg.shape[1]),int(originImg.shape[0])))
        except Exception as e:
            raise RuntimeError(f"Error during model inference: {str(e)}")
        
        return np.clip(originImg * edge, 0, 255)

    def pre_processing(self, image, input_h=512, input_w=512, device='cuda'):
        
        if image is None:
            raise ValueError("image cannot be None")
       
        device = torch.device(device if torch.cuda.is_available() and device == 'cuda' else 'cpu')
        
        try:
            image = cv2.resize(image, (input_w, input_h))
            out_image = image.astype(np.float32) / 255.0
            out_image = out_image - 0.5
            out_image = out_image.transpose(2, 0, 1).reshape(1, 3, input_h, input_w)
            out_image = torch.from_numpy(out_image.copy())  
            out_image = out_image.to(device)
            return out_image
        except Exception as e:
            raise RuntimeError(f"pre_processing failed: {str(e)}")

