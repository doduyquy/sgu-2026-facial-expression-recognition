import torch
from torch import device
import os
import numpy as np 
from datetime import datetime
from src.utils.logger_wandb import init_wandb, log_image_to_wandb, log_metrics


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
        self.patience = config['training'].get('patience', 10)
        self.model_name = config['model'].get('name', 'simple_cnn')
        self.use_wandb = config['logging'].get('use_wandb', True)
        self.run_name = run_name
        self.config = config
        self.path_save_ckpt = save_dir
        self.landmark_diversity_lambda = config['training'].get('landmark_diversity_lambda', 0.3)
        self.landmark_entropy_lambda = config['training'].get(
            'landmark_entropy_lambda',
            config['training'].get('landmark_sparsity_lambda', 0.1),
        )
        self.landmark_coord_spread_lambda = config['training'].get('landmark_coord_spread_lambda', 0.1)
        self.landmark_edge_align_lambda = config['training'].get('landmark_edge_align_lambda', 0.02)

    @staticmethod
    def _extract_logits(outputs):
        if isinstance(outputs, dict):
            return outputs.get("logits")
        if isinstance(outputs, (list, tuple)) and len(outputs) > 0:
            return outputs[0]
        return outputs

    def _extract_aux_losses(self, outputs):
        if isinstance(outputs, dict):
            aux = outputs.get("aux_losses", None)
            if isinstance(aux, dict):
                return aux
        getter = getattr(self.model, "get_aux_losses", None)
        if callable(getter):
            aux = getter()
            if isinstance(aux, dict):
                return aux
        return {}


    def train_one_epoch(self):
        self.model.train()

        running_loss = 0.0
        corrects = 0
        total = 0

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = self._extract_logits(outputs)

            cls_loss = self.criterion(logits, labels)
            aux_losses = self._extract_aux_losses(outputs)

            div_loss = aux_losses.get("landmark_diversity", torch.tensor(0.0, device=self.device))
            entropy_loss = aux_losses.get(
                "landmark_entropy",
                aux_losses.get("landmark_sparsity", torch.tensor(0.0, device=self.device)),
            )
            coord_spread_loss = aux_losses.get("landmark_coord_spread", torch.tensor(0.0, device=self.device))
            edge_align_loss = aux_losses.get("landmark_edge_align", torch.tensor(0.0, device=self.device))

            loss = (
                cls_loss
                + (self.landmark_diversity_lambda * div_loss)
                + (self.landmark_entropy_lambda * entropy_loss)
                + (self.landmark_coord_spread_lambda * coord_spread_loss)
                + (self.landmark_edge_align_lambda * edge_align_loss)
            )
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(logits, dim=1)
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
                logits = self._extract_logits(outputs)
                cls_loss = self.criterion(logits, labels)
                aux_losses = self._extract_aux_losses(outputs)
                div_loss = aux_losses.get("landmark_diversity", torch.tensor(0.0, device=self.device))
                entropy_loss = aux_losses.get(
                    "landmark_entropy",
                    aux_losses.get("landmark_sparsity", torch.tensor(0.0, device=self.device)),
                )
                coord_spread_loss = aux_losses.get("landmark_coord_spread", torch.tensor(0.0, device=self.device))
                edge_align_loss = aux_losses.get("landmark_edge_align", torch.tensor(0.0, device=self.device))
                loss = (
                    cls_loss
                    + (self.landmark_diversity_lambda * div_loss)
                    + (self.landmark_entropy_lambda * entropy_loss)
                    + (self.landmark_coord_spread_lambda * coord_spread_loss)
                    + (self.landmark_edge_align_lambda * edge_align_loss)
                )
                running_loss += loss.item() * images.size(0)

                _, preds = torch.max(logits, dim=1)
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

        best_val_loss = float("inf")
        patience_counter = 0
        all_train_loss = []
        all_val_loss = []

        print(f'\n--> Start training in total {self.epochs} epochs with {self.device} device. Start...\n')

        for ep in range(self.epochs):
            progress = ep / max(self.epochs - 1, 1)
            set_progress = getattr(self.model, "set_training_progress", None)
            if callable(set_progress):
                set_progress(progress)

            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.validate()

            all_train_loss.append(train_loss)
            all_val_loss.append(val_loss)

            print(
                f"Epoch {ep+1}/{self.epochs} - "
                f"loss: {train_loss:.4f} - accuracy: {train_acc.item():.4f} - "
                f"val_loss: {val_loss:.4f} - val_accuracy: {val_acc.item():.4f}"
            )
            get_prior = getattr(self.model, "get_current_prior_strength", None)
            if callable(get_prior):
                current_prior = get_prior()
                if current_prior is not None:
                    print(f"\tlandmark_prior_strength(now): {current_prior:.4f}")

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
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0

                torch.save({
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "epoch": ep
                }, self.path_save_ckpt)
                print(f"\t--- Save best at ep {ep+1}, val_loss: {val_loss:.4f}, path: {self.path_save_ckpt} ---")

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