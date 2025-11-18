import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
import random

from config.config import DEVICE, LOG_DIR, MODEL_SAVE_DIR
from data.dataset import get_data_loader
from models.ResNet18 import CIFAR_ResNet18
from models.customResNet import customResNet
from utils.logger import setup_logger
from utils.grad_cam import GradCAM, visualize_gradcam  
import torch.nn.functional as F


# logger = setup_logger(log_dir=str(LOG_DIR), log_file='test')
logger = setup_logger(log_dir=str(LOG_DIR), log_file='custom_test')

def denormalize(img):
    mean = np.array([0.4914, 0.4822, 0.4465])
    std  = np.array([0.2023, 0.1994, 0.2010])
    return img * std + mean


def setup_gradcam_model(model):
    target_layer = None
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            target_layer = module  # 最后一个 conv
    if target_layer is None:
        raise ValueError("No convolutional layer found for Grad-CAM")
    
    logger.info(f"Grad-CAM target layer: {target_layer}")
    return GradCAM(model, target_layer)


def save_gradcam_visualization(original_image, heatmap, superimposed_img, 
                              predicted_class, true_class, confidence, save_dir):
    """
    保存 Grad-CAM 可视化结果
    """
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 原始图像
    axes[0].imshow(original_image)
    axes[0].set_title(f'Original\nTrue: {true_class}')
    axes[0].axis('off')
    
    # 热力图
    axes[1].imshow(heatmap)
    axes[1].set_title('Grad-CAM Heatmap')
    axes[1].axis('off')
    
    # 叠加图
    axes[2].imshow(superimposed_img)
    axes[2].set_title(f'Overlay\nPred: {predicted_class} ({confidence:.2f})')
    axes[2].axis('off')
    
    # 保存图像
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"gradcam_{true_class}_pred_{predicted_class}_{timestamp}.png"
    save_path = os.path.join(save_dir, filename)
    
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    
    return save_path

def test_model_with_gradcam(visualize_samples=10):
    """
    Testing model classification with Grad-CAM visualization.
    Uses smart sampling: half correct samples + half wrong samples.
    """
    _, test_loader, classes = get_data_loader()

    # # Load custom model
    # model = customResNet(num_classes=len(classes)).to(DEVICE)

    # Load standerd resnet model
    model = CIFAR_ResNet18(num_classes=len(classes)).to(DEVICE)


    try:
        # checkpoint = torch.load(str(MODEL_SAVE_DIR/'custom_final_model.pth'))
        checkpoint = torch.load(str(MODEL_SAVE_DIR/'resnet_final_model.pth')) # ResNet Model
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("No trained model found!")
        return

    # Setup Grad-CAM
    gradcam = setup_gradcam_model(model)

    # Prepare save directory
    # gradcam_dir = LOG_DIR / "gradcam_visualizations"    # custom
    gradcam_dir = LOG_DIR / "gradcam_visualizations_18"    # original

    os.makedirs(gradcam_dir, exist_ok=True)

    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)

    # New: store samples for smart sampling
    all_correct_samples = []
    all_wrong_samples = []

    logger.info("Collecting samples...")

    # ------------ PASS 1: Collect predictions & store samples ------------
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Scanning Test Set"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            for i in range(labels.size(0)):
                label, pred = labels[i], predicted[i]

                class_total[label.item()] += 1
                if pred == label:
                    class_correct[label.item()] += 1
                    all_correct_samples.append((images[i].detach(), label.detach(), pred.detach()))
                else:
                    all_wrong_samples.append((images[i].detach(), label.detach(), pred.detach()))

    # ------------ SMART SAMPLING ------------
    logger.info(f"Collected {len(all_correct_samples)} correct samples and {len(all_wrong_samples)} wrong samples.")

    def smart_sample(correct_list, wrong_list, total_samples):
        half = total_samples // 2
        selected = []

        # sample correct samples
        if len(correct_list) >= half:
            selected.extend(random.sample(correct_list, half))
        else:
            selected.extend(correct_list)

        # sample wrong samples
        remaining = total_samples - len(selected)
        if len(wrong_list) >= remaining:
            selected.extend(random.sample(wrong_list, remaining))
        else:
            selected.extend(wrong_list)

        return selected

    # choose samples
    samples_to_visualize = smart_sample(
        all_correct_samples,
        all_wrong_samples,
        visualize_samples
    )

    logger.info(f"Selected {len(samples_to_visualize)} samples for Grad-CAM visualization.")

    # ------------ PASS 2: Grad-CAM visualization for selected samples ------------
    model.train(False)

    for img_tensor, label, pred in tqdm(samples_to_visualize, desc="Generating Grad-CAM"):

        img_tensor = img_tensor.unsqueeze(0)  # (1,3,H,W)

        # Important: enable grad for Grad-CAM
        img_tensor = img_tensor.clone().detach().requires_grad_(True)

        cam, output = gradcam.generate_cam(img_tensor, pred)
        confidence = F.softmax(output, dim=1)[0][pred].item()

        original_img = img_tensor[0].detach().permute(1, 2, 0).cpu().numpy()

        # visualize
        original_vis, heatmap_vis, superimposed_vis = visualize_gradcam(
            original_img, cam
        )

        save_path = save_gradcam_visualization(
            original_vis,
            heatmap_vis,
            superimposed_vis,
            classes[pred.item()],
            classes[label.item()],
            confidence,
            gradcam_dir
        )

        logger.info(f"Grad-CAM saved: {save_path}")

    # ------------ REPORT ACCURACY ------------
    accuracy = 100 * correct / total
    logger.info(f"Test Accuracy: {accuracy:.2f}%")
    logger.info(f"Grad-CAM saved to: {gradcam_dir}")

    for i, cls in enumerate(classes):
        if class_total[i] > 0:
            acc = 100 * class_correct[i] / class_total[i]
        else:
            acc = 0
        logger.info(f"Accuracy of {cls:10s}: {acc:.2f}% ({class_correct[i]}/{class_total[i]})")

if __name__ == '__main__':
    test_model_with_gradcam(visualize_samples=10)  # 可视化10个样本