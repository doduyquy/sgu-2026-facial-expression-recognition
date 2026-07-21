import sys
sys.path.append('.')
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
from src.training.trainer import Trainer

print("Test training with SAM...")

class DummyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 7)
    def forward(self, x):
        return self.fc(x)
    def get_landmark_outputs(self):
        return None, None
    def get_aux_losses(self):
        return {}

class DummyDataset(Dataset):
    def __len__(self): return 16
    def __getitem__(self, idx):
        return torch.randn(10), torch.randint(0, 7, (1,)).item()

mock_config = {
    'training': {
        'epochs': 2, 
        'patience': 2,
        'optimizer': 'sam_adam',
        'sam_rho': 0.05,
        'lr': 0.01,
        'use_scn': False,
        'mixup_alpha': 0.0,
        'loss': 'cross_entropy'
    },
    'path': {'root': '/tmp/'},
    'model': {'name': 'dummy_model'},
    'logging': {'use_wandb': False}
}

train_loader = DataLoader(DummyDataset(), batch_size=8)
val_loader = DataLoader(DummyDataset(), batch_size=8)

model = DummyModel()
criterion = nn.CrossEntropyLoss()
from src.training.optimizer import build_optimizer
optimizer = build_optimizer(model, mock_config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

try:
    trainer = Trainer(
        model, train_loader, val_loader, criterion, optimizer, None, 
        mock_config, device, "debug_sam", "checkpoint.pth"
    )
    print("Fitting...")
    trainer.fit()
    print("Done SAM!")
except Exception as e:
    import traceback
    traceback.print_exc()
