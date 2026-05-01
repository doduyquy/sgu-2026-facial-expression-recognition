import torch
import torch.distributed as dist
from torch import device
import os
import numpy as np 
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics
from src.training.losses import inception_loss
from src.training.optimizer import build_scheduler, build_optimizer
from .sam import SAM

class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.epochs = config['training'].get('epochs', 100)
        self.patience = config['training'].get('patience', 20)
        self.model_name = config['model'].get('name', 'simple_cnn')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir
        self.monitor = config['training'].get('monitor', 'val_loss')
        self.is_distributed = dist.is_available() and dist.is_initialized()
        self.rank = dist.get_rank() if self.is_distributed else 0
        self.is_main_process = self.rank == 0
        if self.monitor not in ['val_loss', 'val_accuracy']:
            raise ValueError("training.monitor must be either 'val_loss' or 'val_accuracy'")

    def _unwrap_model(self):
        return self.model.module if hasattr(self.model, "module") else self.model

    def _reduce_epoch_stats(self, loss_sum, corrects, total, ortho_sum=0.0):
        stats = torch.tensor(
            [loss_sum, float(corrects), float(total), ortho_sum],
            device=self.device,
            dtype=torch.float64,
        )
        if self.is_distributed:
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)

        total = max(stats[2].item(), 1.0)
        epoch_loss = stats[0].item() / total
        epoch_acc = stats[1] / total
        epoch_ortho = stats[3].item() / total if stats[3].item() > 0 else 0.0
        return epoch_loss, torch.tensor(epoch_acc, device=self.device), epoch_ortho

    def _sync_stop_flag(self, should_stop):
        if not self.is_distributed:
            return should_stop

        stop_tensor = torch.tensor(int(should_stop), device=self.device)
        dist.broadcast(stop_tensor, src=0)
        return bool(stop_tensor.item())


    def train_one_epoch(self):
        self.model.train()

        running_loss = 0.0
        running_ortho = 0.0
        corrects = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # -------------
            # Check loại tuple trả về
            if isinstance(outputs, tuple):
                if len(outputs) == 2 and isinstance(outputs[1], torch.Tensor) and outputs[1].dim() == 0:
                    # Trả về (logits, scalar_aux_loss) -> Orthogonal Loss cho RegionAttention
                    main_out, aux_loss = outputs
                    main_loss = self.criterion(main_out, labels)
                    ortho_weight = self.config.get('model', {}).get('ortho_loss_weight', 0.1)
                    loss = main_loss + ortho_weight * aux_loss
                    outputs = main_out
                    running_ortho += aux_loss.item() * images.size(0)
                else:
                    # [Inception] Trả về (main_out, aux_out_logits)
                    main_out, aux_out = outputs
                    loss = inception_loss(main_out, aux_out, labels, criterion=self.criterion)
                    outputs = main_out # Đặt lại outputs -> tinhs accuracy ở dưới
            else:
                loss = self.criterion(outputs, labels)
            # -------------


            if isinstance(self.optimizer, SAM):
                # ── SAM Step 1 ──
                loss.backward()
                self.optimizer.first_step(zero_grad=True)

                # ── SAM Step 2 ──
                outputs_2 = self.model(images)
                if isinstance(outputs_2, tuple):
                    if len(outputs_2) == 2 and isinstance(outputs_2[1], torch.Tensor) and outputs_2[1].dim() == 0:
                        main_out_2, aux_loss_2 = outputs_2
                        main_loss_2 = self.criterion(main_out_2, labels)
                        ortho_weight = self.config.get('model', {}).get('ortho_loss_weight', 0.1)
                        loss_2 = main_loss_2 + ortho_weight * aux_loss_2
                    else:
                        main_out_2, aux_out_2 = outputs_2
                        loss_2 = inception_loss(main_out_2, aux_out_2, labels, criterion=self.criterion)
                else:
                    loss_2 = self.criterion(outputs_2, labels)
                
                loss_2.backward()
                self.optimizer.second_step(zero_grad=True)
            else:
                # ── Standard Optimizer ──
                loss.backward()
                self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, dim=1)
            corrects += torch.sum(preds == labels.data)
            total += labels.size(0)

        return self._reduce_epoch_stats(running_loss, corrects, total, running_ortho)


    def validate(self):
        self.model.eval()

        running_loss = 0.0
        corrects = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(outputs, dim=1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)

        epoch_loss, epoch_acc, _ = self._reduce_epoch_stats(running_loss, corrects, total)
        return epoch_loss, epoch_acc


    def fit(self):
        """ Fit your model
        Return:
            all_train_loss, all_val_loss
        """
        if self.is_main_process:
            print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb and self.is_main_process:
            init_wandb(config=self.config, run_name=self.run_name)

        best_score = float("inf") if self.monitor == 'val_loss' else -float("inf")
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        if self.is_main_process:
            print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):
            if self.is_distributed and hasattr(self.train_loader.sampler, "set_epoch"):
                self.train_loader.sampler.set_epoch(ep)

            # ── Transfer Learning: kiểm tra có cần mở băng backbone không ──
            base_model = self._unwrap_model()
            if hasattr(base_model, 'check_unfreeze'):
                should_rebuild = base_model.check_unfreeze(ep)
                if should_rebuild:
                    # Rebuild optimizer với LR nhỏ hơn cho fine-tuning
                    finetune_lr = self.config['training'].get('finetune_lr', 1e-5)
                    
                    # Update config temporarily to use build_optimizer
                    old_lr = self.config['training']['lr']
                    self.config['training']['lr'] = finetune_lr
                    self.optimizer = build_optimizer(self.model, self.config)
                    self.config['training']['lr'] = old_lr # Restore
                    
                    # REBUILD scheduler to link to NEW optimizer
                    self.scheduler = build_scheduler(self.optimizer, self.config)
                    
                    # RESET bộ đếm Early Stopping để Phase 2 được chạy đủ
                    patience_counter = 0
                    if self.is_main_process:
                        print(f"[Trainer] Rebuilt optimizer & scheduler with finetune_lr={finetune_lr} and reset patience.")

            train_loss, train_acc, train_ortho_loss = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            if self.is_main_process:
                print(
                    f"Epoch {ep+1}/{self.epochs} - "
                    f"loss: {train_loss:.4f} (ortho: {train_ortho_loss:.4f}) - accuracy: {train_acc.item():.4f} - "
                    f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc.item():.4f}"
                )

            # wandb log
            if self.use_wandb and self.is_main_process:
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
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # save checkpoint
            current_score = val_loss if self.monitor == 'val_loss' else val_acc.item()
            improved = current_score < best_score if self.monitor == 'val_loss' else current_score > best_score

            should_stop = False
            if improved:
                best_score = current_score
                patience_counter = 0

                if self.is_main_process:
                    torch.save({
                        "model_state_dict": self._unwrap_model().state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "epoch": ep,
                        "val_loss": val_loss,
                        "val_accuracy": val_acc.item(),
                        "monitor": self.monitor,
                        "best_score": best_score,
                    }, self.path_save_ckpt)
                    print(
                        f"\t--- Save best at ep {ep+1}, "
                        f"val_loss: {val_loss:.4f}, val_accuracy: {val_acc.item():.4f}, "
                        f"monitor: {self.monitor}, path: {self.path_save_ckpt} ---"
                    )

            else:
                patience_counter += 1
                if self.is_main_process:
                    print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    if self.is_main_process:
                        print(f"\t-_- Early stopping at ep={ep+1}")
                    should_stop = True

            if self._sync_stop_flag(should_stop):
                break

        return all_train_loss, all_val_loss



if __name__ == "__main__":
    from torch.utils.data import DataLoader, Dataset
    import torch.nn as nn
    
    print("Test training...")

    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(10, 7)
        def forward(self, x):
            return self.fc(x)

    class DummyDataset(Dataset):
        def __len__(self): return 16
        def __getitem__(self, idx):
            return torch.randn(10), torch.randint(0, 7, (1,)).item()

    mock_config = {
        'training': {'epochs': 3, 'patience': 2},
        'path': {'root': '/tmp/'},
        'model': {'name': 'dummy_model'}
    }

    train_loader = DataLoader(DummyDataset(), batch_size=8)
    val_loader = DataLoader(DummyDataset(), batch_size=8)

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
