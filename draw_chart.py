import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from src.config.config import LOG_DIR, CHART_SAVE_DIR


def parse_log_file(log_file_path):
    '''
    Parsing training log, extracting loss、training accuracy,
    testing accuracy and learning rate.
    '''
    epochs = []
    training_losses = []
    train_accs = []
    test_accs = []
    learning_rates = []

    # Matching regular expression pattern to extract key information from log
    epoch_pattern = r'Epoch \[(\d+)/\d+\]'
    loss_pattern = r'Train Loss: ([\d.]+)'
    train_acc_pattern = r'Train Acc: ([\d.]+)%'
    test_acc_pattern = r'Test Acc: ([\d.]+)%'
    lr_pattern = r'LR: ([\d.]+)'
    
    with open(log_file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # check if it is the line that contains key information
            if 'Epoch' in line and 'Train Loss' in line:
                # extract information
                epoch_match = re.search(epoch_pattern, line)
                loss_match = re.search(loss_pattern, line)
                train_acc_match = re.search(train_acc_pattern, line)
                test_acc_match = re.search(test_acc_pattern, line)
                lr_match = re.search(lr_pattern, line)

                # check if this line contains all information we need
                if all([epoch_match, loss_match, train_acc_match, test_acc_match, lr_match]):
                    epochs.append(int(epoch_match.group(1)))    # accquire first ()
                    training_losses.append(float(loss_match.group(1)))
                    train_accs.append(float(train_acc_match.group(1)))
                    test_accs.append(float(test_acc_match.group(1)))
                    learning_rates.append(float(lr_match.group(1)))  

    return {
        'epochs':epochs,
        'train_losses':training_losses,
        'train_accs':train_accs,
        'test_accs':test_accs,
        'learning_rates':learning_rates
    }

def plot_training_curves(data, save_path=None):
    '''
    Drawing training curves
    '''
    epochs = data['epochs']
    train_losses = data['train_losses']
    train_accs = data['train_accs']
    test_accs = data['test_accs']
    learning_rates = data['learning_rates']

    # create subfigure
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

     # train_loss curve
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # accuracy curve
    ax2.plot(epochs, train_accs, 'g-', label='Train Accuracy', linewidth=2)
    ax2.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Test Accuracy')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # learning rate curve
    ax3.plot(epochs, learning_rates, 'purple', label='Learning Rate', linewidth=2)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate')
    ax3.set_title('Learning Rate Schedule')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_yscale('log')

    # only test accuracy curve
    ax4.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Accuracy(%)')
    ax4.set_title('Test Accuracy')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    # mark best accuracy and epoch in plot
    best_test_acc = max(test_accs)
    best_epoch = epochs[test_accs.index(best_test_acc)]
    ax2.annotate(f'Best: {best_test_acc:.2f}%', 
                xy=(best_epoch, best_test_acc), 
                xytext=(best_epoch, best_test_acc - 10),
                arrowprops=dict(facecolor='black', shrink=0.1),
                horizontalalignment='center')
    ax4.annotate(f'Best: {best_test_acc:.2f}%', 
                xy=(best_epoch, best_test_acc), 
                xytext=(best_epoch, best_test_acc - 10),
                arrowprops=dict(facecolor='black', shrink=0.1),
                horizontalalignment='center')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    plt.show()

    print(f"训练摘要:")
    print(f"总训练轮次: {len(epochs)}")
    print(f"最佳测试准确率: {best_test_acc:.2f}% (第 {best_epoch} 轮)")
    print(f"最终测试准确率: {test_accs[-1]:.2f}%")
    print(f"最终训练损失: {train_losses[-1]:.4f}")

if __name__ == "__main__":
    # log_file_path = str(LOG_DIR / 'train')      # ResNet-18
    log_file_path = str(LOG_DIR / 'custom_train')        # customResNet


    print("log parsing...")
    training_data =  parse_log_file(log_file_path)

    print('drawing chart...')
    # plot_training_curves(training_data, save_path=str(CHART_SAVE_DIR/'training_curve.png'))   # ResNet-18
    plot_training_curves(training_data, save_path=str(CHART_SAVE_DIR/'custom_training_curve.png'))     # customResNet


