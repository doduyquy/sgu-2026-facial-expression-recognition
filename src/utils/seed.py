import os
import random
import numpy as np
import torch
import numpy as np

def set_seed(seed=21):
    """Setup seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    # Cấu hình CUDNN cho deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    print(f"[OK] Seed set to {seed}")

if __name__ == "__main__":
    set_seed(21)
    print(random.randint(1, 10))
    print(np.random.randint(1, 10))
    print(torch.randint(1, 10, (1,)).item())