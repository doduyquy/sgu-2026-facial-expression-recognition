import os
import torch
import numpy as np 
import torchvision.transforms.functional as TF
import torch.nn.functional as F
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
        # Tuned defaults to avoid over-constraint (Sunset suggestions)
        self.landmark_diversity_lambda = config['training'].get('landmark_diversity_lambda', 0.15)
        self.landmark_entropy_lambda = config['training'].get(
            'landmark_entropy_lambda',
            config['training'].get('landmark_sparsity_lambda', 0.02),
        )
        # keep edge_align disabled by default
        self.landmark_edge_align_lambda = config['training'].get('landmark_edge_align_lambda', 0.0)
        # per-keypoint edge consistency (weaker default)
        self.landmark_edge_consistency_lambda = config['training'].get('landmark_edge_consistency_lambda', 0.01)
        # small regularization on learned edge conv; disable TV by default
        self.landmark_edge_conv_reg_lambda = config['training'].get('landmark_edge_conv_reg_lambda', 1e-5)
        self.landmark_edge_tv_lambda = config['training'].get('landmark_edge_tv_lambda', 0.0)
        # augment consistency lambda (pred(Aug(x)) ≈ Aug(pred(x))) - conservative default
        self.landmark_augment_consistency_lambda = config['training'].get('landmark_augment_consistency_lambda', 0.01)
        # probability to run augment-consistency per batch (to save compute)
        self.landmark_augment_consistency_prob = config['training'].get('landmark_augment_consistency_prob', 0.3)
        # Target entropy for attention maps; regularize toward this value (abs diff)
        self.landmark_target_entropy = config['training'].get('landmark_target_entropy', 2.0)
        # auxiliary classification head weight for landmark features
        self.landmark_aux_cls_lambda = config['training'].get('landmark_aux_cls_lambda', 0.3)
        # auxiliary logits consistency (KL) weight
        self.landmark_aux_consistency_lambda = config['training'].get('landmark_aux_consistency_lambda', 0.1)

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

        # runtime lambdas (may be set by fit() for staged schedule)
        div_lambda = getattr(self, '_runtime_diversity_lambda', self.landmark_diversity_lambda)
        entropy_lambda = getattr(self, '_runtime_entropy_lambda', self.landmark_entropy_lambda)
        edge_consistency_lambda = getattr(self, '_runtime_edge_consistency_lambda', self.landmark_edge_consistency_lambda)
        augment_lambda = getattr(self, '_runtime_augment_lambda', self.landmark_augment_consistency_lambda)
        aux_cls_lambda = getattr(self, '_runtime_aux_cls_lambda', self.landmark_aux_cls_lambda)

        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images)
            logits = self._extract_logits(outputs)

            cls_loss = self.criterion(logits, labels)
            aux_losses = self._extract_aux_losses(outputs)

            # (no target) use raw entropy directly for both train and val

            div_loss = aux_losses.get("landmark_diversity", torch.tensor(0.0, device=self.device))
            entropy_loss = aux_losses.get(
                "landmark_entropy",
                aux_losses.get("landmark_sparsity", torch.tensor(0.0, device=self.device)),
            )
            # normalize entropy by heatmap area to make it scale-invariant: entropy / log(H*W)
            heatmaps_now, _ = self.model.get_landmark_outputs()
            if heatmaps_now is not None:
                try:
                    _, _, H_att, W_att = heatmaps_now.shape
                    denom = float(np.log(max(1, H_att * W_att)))
                    if denom > 0:
                        entropy_reg = entropy_loss / denom
                    else:
                        entropy_reg = entropy_loss
                except Exception:
                    entropy_reg = entropy_loss
            else:
                entropy_reg = entropy_loss
            edge_align_loss = aux_losses.get("landmark_edge_align", torch.tensor(0.0, device=self.device))
            edge_consistency_loss = aux_losses.get("landmark_edge_consistency", torch.tensor(0.0, device=self.device))
            edge_conv_reg = aux_losses.get("landmark_edge_conv_reg", torch.tensor(0.0, device=self.device))
            edge_tv = aux_losses.get("landmark_edge_tv", torch.tensor(0.0, device=self.device))
            # Compose base loss (classification + landmark auxes) using runtime lambdas
            # Note: we intentionally exclude conv/TV regularizers from the main loss to avoid over-constraint
            loss = (
                cls_loss
                + (div_lambda * div_loss)
                + (entropy_lambda * entropy_reg)
                + (edge_consistency_lambda * edge_consistency_loss)
            )

            # Auxiliary classification on landmark features (encourage feat_k to be useful)
            aux_logits_getter = getattr(self.model, 'get_landmark_aux_logits', None)
            if callable(aux_logits_getter):
                aux_logits = aux_logits_getter()
            else:
                aux_logits = None
            if aux_logits is not None:
                try:
                    if aux_cls_lambda > 0.0:
                        aux_cls_loss = F.cross_entropy(aux_logits, labels)
                        loss = loss + (aux_cls_lambda * aux_cls_loss)
                    # KL consistency: make aux logits follow main logits' decision
                    aux_consistency_lambda = getattr(self, '_runtime_aux_consistency_lambda', self.landmark_aux_consistency_lambda)
                    if aux_consistency_lambda > 0.0:
                        # mutual learning: symmetric KL between main and aux (average of both directions)
                        p_main = F.softmax(logits.detach(), dim=1)
                        p_aux = F.softmax(aux_logits, dim=1)
                        kl1 = F.kl_div(F.log_softmax(aux_logits, dim=1), p_main, reduction='batchmean')
                        kl2 = F.kl_div(F.log_softmax(logits, dim=1), p_aux.detach(), reduction='batchmean')
                        kl = 0.5 * (kl1 + kl2)
                        loss = loss + (aux_consistency_lambda * kl)
                except Exception:
                    pass

            # Augment-consistency: pred(Aug(x)) ≈ Aug(pred(x)) using heatmaps
            # probabilistically run augment-consistency to save compute (and only when enabled)
            if augment_lambda > 0.0 and getattr(self.model, 'use_learned_landmark_branch', False) and (np.random.rand() < getattr(self, 'landmark_augment_consistency_prob', 0.3)):
                try:
                    # get latest landmark heatmaps from original forward
                    heatmaps_orig, coords_orig = self.model.get_landmark_outputs()
                    if heatmaps_orig is not None:
                        # build an augmented batch (same random params for whole batch)
                        bsz, c, H, W = heatmaps_orig.shape
                        # sample milder random affine params to avoid heavy misalignment on small images
                        angle = float(np.random.uniform(-10, 10))
                        max_tx = max(1, int(0.05 * W))
                        max_ty = max(1, int(0.05 * H))
                        translate = (int(np.random.randint(-max_tx, max_tx + 1)), int(np.random.randint(-max_ty, max_ty + 1)))
                        scale = 1.0
                        shear = 0.0

                        # apply same transform to input images
                        images_aug = torch.stack([TF.affine(img, angle=angle, translate=translate, scale=scale, shear=shear, fill=0) for img in images])

                        # forward pass on augmented images without updating BN running stats or grads
                        was_training = self.model.training
                        self.model.eval()
                        with torch.no_grad():
                            _ = self.model(images_aug)
                        heatmaps_aug, coords_aug = self.model.get_landmark_outputs()
                        if was_training:
                            self.model.train()

                        if heatmaps_aug is not None:
                            # transform original heatmaps (detach to use as pseudo-target)
                            heatmaps_orig_det = heatmaps_orig.detach()
                            transformed = []
                            for i in range(heatmaps_orig_det.size(0)):
                                # heatmaps_orig_det[i]: (K, H, W) -> apply TF.affine per-channel
                                chs = []
                                for k in range(heatmaps_orig_det.size(1)):
                                    hm = heatmaps_orig_det[i, k:k+1]
                                    hm_t = TF.affine(hm, angle=angle, translate=translate, scale=scale, shear=shear, fill=0)
                                    chs.append(hm_t)
                                transformed.append(torch.cat(chs, dim=0))
                            transformed = torch.stack(transformed, dim=0).to(heatmaps_aug.dtype)

                            augment_consistency_loss = F.l1_loss(heatmaps_aug, transformed, reduction='mean')
                            loss = loss + (augment_lambda * augment_consistency_loss)
                except Exception:
                    # if any issue with augment or TF, skip augment consistency for this batch
                    pass
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
                # use raw entropy as regularizer (no target -- temperature already controls sharpness)
                entropy_reg = entropy_loss
                edge_align_loss = aux_losses.get("landmark_edge_align", torch.tensor(0.0, device=self.device))
                edge_consistency_loss = aux_losses.get("landmark_edge_consistency", torch.tensor(0.0, device=self.device))
                edge_conv_reg = aux_losses.get("landmark_edge_conv_reg", torch.tensor(0.0, device=self.device))
                edge_tv = aux_losses.get("landmark_edge_tv", torch.tensor(0.0, device=self.device))
                # Use runtime lambdas if scheduled by fit(), otherwise fall back to configured defaults
                div_lambda = getattr(self, '_runtime_diversity_lambda', self.landmark_diversity_lambda)
                entropy_lambda = getattr(self, '_runtime_entropy_lambda', self.landmark_entropy_lambda)
                edge_consistency_lambda = getattr(self, '_runtime_edge_consistency_lambda', self.landmark_edge_consistency_lambda)
                edge_conv_reg_lambda = getattr(self, '_runtime_edge_conv_reg_lambda', self.landmark_edge_conv_reg_lambda)
                loss = (
                    cls_loss
                    + (div_lambda * div_loss)
                    + (entropy_lambda * entropy_reg)
                    + (edge_consistency_lambda * edge_consistency_loss)
                    + (edge_conv_reg_lambda * edge_conv_reg)
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
                try:
                    set_progress(progress)
                except Exception:
                    pass

            # apply 3-phase staged lambda schedule (Sunset recommendation)
            if progress <= 0.3:
                # Phase 1: focus on classification only
                self._runtime_diversity_lambda = 0.0
                self._runtime_entropy_lambda = 0.0
                self._runtime_augment_lambda = 0.0
                self._runtime_edge_consistency_lambda = 0.0
                self._runtime_aux_cls_lambda = 0.0
                self._runtime_aux_consistency_lambda = 0.0
            elif progress <= 0.7:
                # Phase 2: start lightweight landmark usefulness training
                self._runtime_diversity_lambda = 0.05
                self._runtime_entropy_lambda = 0.0
                self._runtime_augment_lambda = 0.0
                self._runtime_edge_consistency_lambda = 0.0
                self._runtime_aux_cls_lambda = 0.1
                self._runtime_aux_consistency_lambda = 0.1
            else:
                # Phase 3: refine attention, weak regularizers
                self._runtime_diversity_lambda = 0.2
                self._runtime_entropy_lambda = 0.01
                self._runtime_augment_lambda = 0.01
                self._runtime_edge_consistency_lambda = 0.005
                self._runtime_aux_cls_lambda = 0.2
                self._runtime_aux_consistency_lambda = 0.1

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