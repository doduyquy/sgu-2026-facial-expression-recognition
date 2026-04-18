import torch
from src.utils.logger_wandb import init_wandb, log_metrics
from src.training.losses import inception_loss
from src.training.optimizer import build_scheduler, build_optimizer
from .sam import SAM
from src.utils.checkpoint import save_checkpoint, unwrap_model

class Trainer:
    """Forward -> Compute loss -> zero_grad -> Backward -> Update weights (step)"""
    def __init__(self, model, train_loader, val_loader, criterion, optimizer, scheduler, config, device, run_name, save_dir):
        self.model = model
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


    def train_one_epoch(self):
        self.model.train()

        running_loss = 0.0
        corrects = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            
            # -------------
            # [Inception]Vì incpetion trả về tuple của trong lúc training
            if isinstance(outputs, tuple):
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
                    main_out, aux_out = outputs_2
                    loss_2 = inception_loss(main_out, aux_out, labels, criterion=self.criterion)
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

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        return epoch_loss, epoch_acc


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

        epoch_loss = running_loss / total
        epoch_acc = corrects.double() / total

        return epoch_loss, epoch_acc


    def fit(self):
        """ Fit your model
        Return:
            all_train_loss, all_val_loss
        """
        print(f'\n--> Train on {len(self.train_loader.dataset)} samples, validate on {len(self.val_loader.dataset)} samples')

        if self.use_wandb:
            init_wandb(config=self.config, run_name=self.run_name)

        best_val_acc = 0.0
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):
            base_model = unwrap_model(self.model)

            # ── Transfer Learning: kiểm tra có cần mở băng backbone không ──
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
                    print(f"[Trainer] Rebuilt optimizer & scheduler with finetune_lr={finetune_lr} and reset patience.")

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            print(
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
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()

            # save checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0

                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=ep,
                    checkpoint_path=self.path_save_ckpt,
                )
                print(f"\t--- Save best at ep {ep+1}, val_accuracy: {val_acc:.4f}, path: {self.path_save_ckpt} ---")

            else:
                patience_counter += 1
                print(f"\t-!- No improvement: {patience_counter}/{self.patience}")
                if patience_counter >= self.patience:
                    print(f"\t-_- Early stopping at ep={ep+1}")
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
