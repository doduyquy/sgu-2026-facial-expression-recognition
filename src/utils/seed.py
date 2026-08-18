import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=21):
    """Setup seed for reproducibility"""
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
    tf.random.set_seed(seed)
    
    # Cấu hình TF cho deterministic behavior
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'  # Gây lỗi với CropAndResizeBackpropImage
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
    
    print(f"[OK] Seed set to {seed}")

if __name__ == "__main__":
    set_seed(21)
    print(random.randint(1, 10))
    print(np.random.randint(1, 10))
    print(tf.random.uniform((1,), minval=1, maxval=10, dtype=tf.int32).numpy()[0])