import torch
from tqdm import tqdm

from config.config import DEVICE, LOG_DIR, MODEL_SAVE_DIR
from data.dataset import get_data_loader
from models.ResNet18 import CIFAR_ResNet18
from models.customResNet import customResNet
from utils.logger import setup_logger

# logger = setup_logger(log_dir=str(LOG_DIR), log_file='test')
logger = setup_logger(log_dir=str(LOG_DIR), log_file='custom_test')


def test_model():
    '''
    Testing model classification in test dataset.
    '''
    _, test_loader, classes = get_data_loader()

    # Initialize ResNet model.
    # model = CIFAR_ResNet18(num_classes=len(classes)).to(DEVICE)   # ResNet-18
    model = customResNet(num_classes=len(classes)).to(DEVICE)     # CustomResNet


    try:
        # checkpoint = torch.load(str(MODEL_SAVE_DIR/'resnet_final_model.pth')) # ResNet Model
        checkpoint = torch.load(str(MODEL_SAVE_DIR/'custom_final_model.pth'))   # custom ResNet Model
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        logger.info("Model loaded successfully")
    except FileNotFoundError:
        logger.error("No trained model found!")
        return

    correct = 0
    total = 0
    class_correct = [0] * len(classes)
    class_total = [0] * len(classes)


    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            c = (predicted == labels).reshape(-1)
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += c[i].item()
                class_total[label] += 1


    accuracy = 100 * correct / total
    logger.info(f"Test accuracy:{accuracy:.2f}%")


    for i, cls in enumerate(classes):
        acc = 100 * class_correct[i] / class_total[i]
        logger.info(f"Accuracy of {cls:10s}:{acc:.2f}%({class_correct[i]}/{class_total[i]})")


if __name__ == '__main__':
    test_model()

