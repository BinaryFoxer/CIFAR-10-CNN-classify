import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from tqdm import tqdm
from pathlib import Path
import sys

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.config.config import DEVICE, LEARNING_RATE, EPOCHS, MODEL_SAVE_DIR, MOMENTUM, \
                                WEIGHT_DECAY, USE_COSINE_ANNEALING, MIN_LR, LOG_DIR
from src.data.dataset import get_data_loader
from src.models.ResNet18 import CIFAR_ResNet18
from src.utils.logger import setup_logger

Path(MODEL_SAVE_DIR).mkdir(parents=True, exist_ok=True)

current_file = Path(__file__).name
logger = setup_logger(log_dir=str(LOG_DIR), log_file='train')

def train_model():
    '''
    Training models
    '''
    train_loaders, test_loaders, classes = get_data_loader()

    # Initialize ResNet model.
    model = CIFAR_ResNet18(num_classes=len(classes)).to(DEVICE)

    # Using SGD optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY
    )

    # Learning rate scheduler
    if USE_COSINE_ANNEALING:
        scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=MIN_LR)
    else:
        scheduler = StepLR(optimizer, step_size=60, gamma=0.1)

    lossf = nn.CrossEntropyLoss()

    # training loop
    best_accuracy = 0.0
    train_losses = []
    test_accuracies = []

    logger.info("Starting ResNet-18 training on CIFAR-10...")
    logger.info(f"Using device: {DEVICE}")
    logger.info(f"Training for {EPOCHS} epochs")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        presentbar = tqdm(train_loaders, desc=f'Epoch {epoch+1}/{EPOCHS}')

        for imgs, labels in presentbar:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)

            # forward propagation
            outputs = model(imgs)
            loss = lossf(outputs, labels)

            # backward propagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # statistical result
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # update bar
            presentbar.set_postfix({
                'Loss': f'{loss.item():.3f}',
                'Acc': f'{100.*correct/total:.2f}%',
                'LR': f'{optimizer.param_groups[0]["lr"]:.6f}'
            })

        # Calculate the average loss and training accuracy rate
        avg_loss = running_loss / len(train_loaders)
        train_accuracy = 100 * correct / total
        train_losses.append(avg_loss)

        # update learning rate
        scheduler.step()

        # -------------------------------test----------------------------------
        # evaluation phase
        model.eval()
        test_correct = 0
        test_total = 0

        # Disable gradient calculation. During the testing phase,
        # we usually do not need to calculate gradients because there is 
        # no need for backpropagation to update the model parameters.
        with torch.no_grad():
            for imgs, labels in test_loaders:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                outputs = model(imgs)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += predicted.eq(labels).sum().item()

        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)

        logger.info(f'Epoch [{epoch+1}/{EPOCHS}] | '
                f'Train Loss: {avg_loss:.4f} | Train Acc: {train_accuracy:.2f}% | '
                f'Test Acc: {test_accuracy:.2f}% | '
                f'LR: {optimizer.param_groups[0]["lr"]:.6f}')

        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save({
                'epoch':epoch,
                'model_state_dict':model.state_dict(),
                'optimizer_state_dict':optimizer.state_dict(),
                'scheduler_state_dict':scheduler.state_dict(),
                'train_loss':avg_loss,
                'train_accuracy':train_accuracy,
                'test_accuracy':test_accuracy,
                'train_losses':train_losses,
                'test_accuracies':test_accuracies
            },f'{MODEL_SAVE_DIR}/best_resnet_model.pth')

            logger.info(f'New best model saved with accuracy: {test_accuracy:.2f}%')

        # Save checkpoints every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_loss,
                'train_accuracy': train_accuracy,
                'test_accuracy': test_accuracy,
            }, f'{MODEL_SAVE_DIR}/resnet_checkpoint_epoch_{epoch+1}.pth')

    # training completed
    logger.info(f'Training completed. Best accuracy: {best_accuracy:.2f}%')
    torch.save({
        'model_state_dict':model.state_dict(),
        'optimizer_state_dict':optimizer.state_dict(),
        'test_accuracy':test_accuracy,
        'train_loss': avg_loss,
        'train_accuracy': train_accuracy,
    }, f'{MODEL_SAVE_DIR}/resnet_final_model.pth')




if __name__ == '__main__':
    train_model()
