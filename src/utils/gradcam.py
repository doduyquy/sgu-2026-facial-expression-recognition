import torch
import torch.nn.functional as F
import numpy as np
import cv2

class SimpleGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        
        self.gradients = None
        self.activations = None
        
        # Đăng ký hooks để lấy activations và gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        
        # Enable grad tạm thời vì Evaluate đang trong no_grad
        with torch.enable_grad():
            input_tensor.requires_grad = True
            output = self.model(input_tensor)
            
            if target_class is None:
                target_class = output.argmax(dim=1).item()
                
            self.model.zero_grad()
            target = output[0, target_class]
            target.backward(retain_graph=True)
        
        # Lấy gradients và activations
        gradients = self.gradients.detach().cpu().numpy()[0]
        activations = self.activations.detach().cpu().numpy()[0]
        
        # Global Average Pooling trên Gradients -> Trọng số từng Channel
        weights = np.mean(gradients, axis=(1, 2))
        
        # Tính tổ hợp tuyến tính
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
            
        cam = np.maximum(cam, 0) # ReLU step
        
        # Normalize về [0, 1]
        cam -= np.min(cam)
        cam_max = np.max(cam)
        if cam_max != 0:
            cam /= cam_max
            
        # Nội suy lên kích thước ảnh gốc
        input_size = input_tensor.shape[2:]
        cam = cv2.resize(cam, (input_size[1], input_size[0]))
        return cam

def apply_heatmap(img_tensor, cam_matrix):
    """
    Nhận image gốc (Tensor) và heatmap array, trộn lại ra ảnh RGB rực rỡ để vẽ (numpy/tensor). 
    """
    img_np = img_tensor.detach().cpu().numpy()
    
    # Đưa về (H, W, C) array
    if img_np.shape[0] == 1:
        img_np = img_np.squeeze(0)          # -> (H, W)
        img_np = np.stack((img_np,)*3, axis=-1)  # -> (H, W, 3) 
    elif img_np.shape[0] == 3:
        img_np = img_np.transpose(1, 2, 0)
        
    # Tạo màu nhiệt từ OpenCV (BGR -> RGB)
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_matrix), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    
    # Ép kiểu và normalize
    heatmap = np.float32(heatmap) / 255
    img_np = np.float32(img_np)
    if np.max(img_np) > 1.0: # nhỡ ảnh chưa normalize [0, 1]
        img_np = img_np / 255.0
        
    cam_img = heatmap * 0.5 + img_np * 0.5
    cam_img = cam_img / np.max(cam_img)
    
    return cam_img
