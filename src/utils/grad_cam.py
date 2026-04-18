import torch
import torch.nn as nn
import numpy as np
import cv2

def find_last_conv_layer(model):
    """Tìm layer Convolution cuối cùng trong model để áp dụng GradCAM"""
    last_conv = None
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module
    return last_conv

class GradCAM:
    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else find_last_conv_layer(model)
        
        if self.target_layer is None:
            raise ValueError("Không tìm thấy layer Conv2d nào trong model để lấy gradient!")
            
        self.gradients = None
        self.activations = None
        
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        """
        x: ảnh input (1, C, H, W)
        """
        self.model.eval()
        
        # Bật grad để Backward được chạy đến layer lấy feature
        x_requires_grad = x.clone().detach().requires_grad_(True)
        
        output = self.model(x_requires_grad)
        if isinstance(output, tuple): # Handle some architectures like Inception
            output = output[0]
            
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
            
        self.model.zero_grad()
        score = output[0, class_idx]
        score.backward()
        
        if self.gradients is None or self.activations is None:
            # Fallback for models where hook fails
            return np.zeros((x.shape[2], x.shape[3]), dtype=np.float32)

        gradients = self.gradients.cpu().data.numpy()[0]
        activations = self.activations.cpu().data.numpy()[0]
        
        # Pool the gradients (Global Average Pooling 2D)
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weigth the activations
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU
        if np.max(cam) > 0:
            cam = cam / np.max(cam)
            
        input_shape = x.shape[2:] # (H, W)
        cam = cv2.resize(cam, (input_shape[1], input_shape[0]))
        return cam

def overlay_cam_on_image(img, cam, colormap=cv2.COLORMAP_JET):
    """
    img: numpy array, ảnh gốc (H, W) hoặc (H, W, 1), range [0, 1] hoặc [0, 255]
    cam: heatmap array (H, W), range [0, 1]
    Return: RGB image overlay
    """
    # Nếu đang lưu ảnh tensor normalize thì trả về RGB cho hiển thị
    if img.max() <= 1.0 and img.min() >= 0.0:
        img = (img * 255).astype(np.uint8)
    elif img.min() < 0: # Case standard normalization mean=0.5, std=0.5
        img = ((img - img.min()) / (img.max() - img.min()) * 255).astype(np.uint8)
    else:
        img = img.astype(np.uint8)
        
    if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[0] == 1):
        if len(img.shape) == 3:
            img = img.squeeze(0)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif len(img.shape) == 3 and img.shape[0] == 3:
        img = np.transpose(img, (1, 2, 0)) # C,H,W -> H,W,C
        
    # Scale heatmap sang uint8
    heatmap = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(heatmap, colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Kết hợp: 60% ảnh gốc + 40% heatmap
    result = cv2.addWeighted(img, 0.6, heatmap, 0.4, 0)
    return result
