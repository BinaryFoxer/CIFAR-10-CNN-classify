import torch
import torch.nn.functional as F
import cv2
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import os

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # 注册钩子
        self._register_hooks()
    
    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output
            
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]
            
        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_backward_hook(backward_hook)
    
    def generate_cam(self, input_tensor, target_class=None):
        """
        生成 Grad-CAM 热力图
        """
        # 前向传播
        output = self.model(input_tensor)
        
        if target_class is None:
            target_class = output.argmax(dim=1)
        
        # 反向传播
        self.model.zero_grad()
        one_hot = torch.zeros_like(output)
        one_hot[0][target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)
        
        # 计算权重
        gradients = self.gradients[0].cpu().data.numpy()
        activations = self.activations[0].cpu().data.numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        # 加权组合
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU 操作
        cam = np.maximum(cam, 0)
        
        # 归一化
        cam = cv2.resize(cam, input_tensor.shape[2:])
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        
        return cam, output

def visualize_gradcam(original_image, cam, alpha=0.5):
    """
    可视化 Grad-CAM 结果
    """
    # 将 CAM 转换为热力图
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    
    # 预处理原始图像
    if original_image.shape[0] == 3:  # CHW -> HWC
        original_image = original_image.transpose(1, 2, 0)
    
    # 反归一化（假设使用 ImageNet 的均值和标准差）
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    original_image = original_image * std + mean
    original_image = np.clip(original_image, 0, 1)
    
    # 叠加热力图
    superimposed_img = heatmap * alpha + original_image
    superimposed_img = superimposed_img / np.max(superimposed_img)
    
    return original_image, heatmap, superimposed_img