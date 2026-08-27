import os
import torch
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class MetricTracker:
    def __init__(self, output_dir, model_name):
        self.history_epochs = []
        self.history_map = []
        self.history_rank1 = []
        self.best_mAP = 0.0
        self.best_epoch = 0
        self.output_dir = output_dir
        self.model_name = model_name
        self.logger = logging.getLogger("transreid.tracker")

    def update(self, epoch, mAP, rank1, model):
        """更新指标，并在产生最佳 mAP 时保存模型权重"""
        self.history_epochs.append(epoch)
        self.history_map.append(mAP)
        self.history_rank1.append(rank1)

        if mAP > self.best_mAP:
            self.best_mAP = mAP
            self.best_epoch = epoch
            
            best_path = os.path.join(self.output_dir, f"{self.model_name}_best.pth")
            
            # 兼容单卡和分布式训练的模型权重提取
            state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save(state_dict, best_path)
            
            self.logger.info(f"*** New best model saved at Epoch {epoch} with mAP: {mAP:.1%} ***")

    def plot(self):
        """训练结束后调用，生成折线图"""
        if len(self.history_epochs) == 0:
            return
            
        self.logger.info(f"Training complete. Best result is at Epoch {self.best_epoch}, mAP: {self.best_mAP:.1%}")
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.history_epochs, self.history_map, marker='o', color='red', label='mAP')
        plt.plot(self.history_epochs, self.history_rank1, marker='s', color='blue', label='Rank-1')
        
        plt.title('Re-ID Validation Metrics over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Score (0.0 to 1.0)')
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend(loc='best')
        
        plot_path = os.path.join(self.output_dir, f"{self.model_name}_training_curve.png")
        plt.savefig(plot_path, bbox_inches='tight', dpi=300)
        plt.close()
        self.logger.info(f"Training curve successfully saved to: {plot_path}")