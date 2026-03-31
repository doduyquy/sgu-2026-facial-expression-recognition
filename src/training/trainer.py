import torch
from torch import device
from tqdm import tqdm
import os
import numpy as np 
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics


class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        # push model to deive
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 10)
        self.model_name = config['model'].get('name', 'simple_cnn')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir


    def train_one_epoch(self, epoch_idx):
        # change model mode to train: 
        self.model.train()
        
        running_loss = 0.0
        corrects = 0
        total = 0

        # format show -> pretty, yeah
        pbar = tqdm(self.train_loader, 
                    desc=f"Epoch {epoch_idx}/{self.epochs}",
                    bar_format="{desc} {n_fmt}/{total_fmt} [{bar:30}] - {elapsed} - {rate_fmt}{postfix}",
                    ncols=140,
                    leave=False)  # leave=False: xóa dòng sau khi xong, để fit() in lại đầy đủ
                    
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()              # remove old grad
            outputs = self.model(images)            # pred
            loss = self.criterion(outputs, labels)  # compute loss
            loss.backward()                         # compute grad
            self.optimizer.step()                   # backward, update weights

            # logging 
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1) # ignore: max_props (_)

            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

            pbar.set_postfix_str(f"loss: {running_loss / total:.4f} - accuracy: {(corrects.double() / total).item():.4f}")
        
        pbar.close()

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total # convert Tensor to Float before divide

        return epoch_loss, epoch_acc

    def validate(self):

        running_loss = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():
            self.model.eval() # required! turn off dropout and freeze batchnorm
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)

                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0) # sum loss of batch

                _, preds = torch.max(outputs, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total
    
        return epoch_loss, epoch_acc

    def fit(self): # training
        """ Fit you model
        Return: 
            all_train_loss, all_val_loss
        """
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        # wandb init
        if self.use_wandb:
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_loss = float("inf")
        patience_counter = 0


        all_train_loss = []
        all_val_loss = []
        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):

            train_loss, train_acc = self.train_one_epoch(epoch_idx=(ep+1))
            val_loss, val_acc = self.validate()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            # In dòng log đầy đủ (ghi đè lên dòng trống mà leave=False để lại)
            tqdm.write(
                f"Epoch {ep+1}/{self.epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc.item():.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc.item():.4f}"
            )

            # wandb log
            if self.use_wandb:
                log_metrics({
                    "Epoch": ep + 1,
                    "Train/Loss": train_loss,
                    "Train/Accuracy": train_acc,
                    "Val/Loss": val_loss,
                    "Val/Accuracy": val_acc,
                    "Learning_Rate": self.optimizer.param_groups[0]['lr']
                }, epoch=ep)

            # lr scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss) # need val loss
                else:
                    self.scheduler.step()

            # check loss and save ckpt
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep
                }, self.path_save_ckpt)
                print(f"\n\t--> Save best at {ep+1} ep, val_loss: {val_loss:.4f}, path: {self.path_save_ckpt}\n")

            else:
                patience_counter += 1
                print(f"-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping at ep={ep}")
                    break

        return all_train_loss, all_val_loss



if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    
    print("Test training...")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 7) # Input 10, Output 7 Cảm xúc
        def forward(self, x):
            return self.fc(x)
    class DummyDataset(Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx): 
            # Ảnh là vector 10 chiều, label từ 0..6
            return torch.randn(10), torch.randint(0, 7, (1,)).item()
    mock_config = {
        'training': {'epochs': 3, 'patience': 2},
        'path': {'root': '/tmp/'}, # Lưu rác ở đây cho nhẹ máy
        'model': {'name': 'dummy_model'}
    }
    # Bọc dữ liệu qua Dataloader
    train_loader = DataLoader(DummyDataset(), batch_size=8)
    val_loader = DataLoader(DummyDataset(), batch_size=8)
    
    # Chuẩn bị Model, Loss, Optimizer
    model = DummyModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    try:
        trainer = Trainer(model, train_loader, val_loader, criterion, optimizer, mock_config, device)
        print("Fitting...")
        trainer.fit()
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")