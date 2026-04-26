import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch

ckpt_path = r"d:\HocTap\Phân tích  và xử lý ảnh\sgu-2026-facial-expression-recognition\checkpoints\resnet152_rot30_2019Nov14_12.47"

if not os.path.exists(ckpt_path):
    print(f"File {ckpt_path} not found.")
else:
    try:
        ckpt = torch.load(ckpt_path, map_location='cpu')
        print("Type of checkpoint:", type(ckpt))
        if isinstance(ckpt, dict):
            print("\n--- Metadata in Checkpoint ---")
            for key in ['num_classes', 'image_size', 'in_channels', 'arch', 'model_name']:
                if key in ckpt:
                    print(f"  {key}: {ckpt[key]}")
            
            # Kiểm tra xem có lưu class names không
            if 'class_names' in ckpt:
                print(f"  class_names: {ckpt['class_names']}")
            
            state_dict = None
            if 'net' in ckpt:
                print("'net' key found, likely contains state_dict.")
                state_dict = ckpt['net']
            elif 'model_state_dict' in ckpt:
                print("'model_state_dict' key found.")
                state_dict = ckpt['model_state_dict']
            else:
                print("Using checkpoint dict as state_dict.")
                state_dict = ckpt
            
            if isinstance(state_dict, dict):
                # Print some layer names to see architecture
                print("\nFirst 10 layer names in state_dict:")
                keys = list(state_dict.keys())
                for k in keys[:10]:
                    val = state_dict[k]
                    if hasattr(val, 'shape'):
                        print(f"  {k}: {val.shape}")
                    else:
                        print(f"  {k}: {type(val)}")
                
                print("\nLast 10 layer names in state_dict:")
                for k in keys[-10:]:
                    val = state_dict[k]
                    if hasattr(val, 'shape'):
                        print(f"  {k}: {val.shape}")
                    else:
                        print(f"  {k}: {type(val)}")
            else:
                print("state_dict is not a dict, it is:", type(state_dict))
        else:
            print("Checkpoint is not a dict.")
    except Exception as e:
        print("Error loading checkpoint:", e)
