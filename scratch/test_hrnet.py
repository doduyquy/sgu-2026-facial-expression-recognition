import os
import time
import torch
import torch.nn.functional as F

import sys
# Add src to path so we can import
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.semantic_roi_graph import SemanticRoiGraphConfig, SemanticROIGraphFER

def test_model(backbone_type="resnet50", use_cuda=False):
    print(f"\n======================================")
    print(f"--- Testing {backbone_type} ---")
    print(f"======================================")
    config = SemanticRoiGraphConfig(backbone_type=backbone_type)
    model = SemanticROIGraphFER(config)
    
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    # 1. Test 3-channel input
    B = 2
    dummy_input_3 = torch.randn(B, 3, 48, 48).to(device)
    
    # 9 default ROIs (B, 9, 4)
    bboxes = torch.tensor([
        [[8, 0, 40, 10], [5, 8, 18, 18], [30, 8, 43, 18], [18, 12, 30, 22], [6, 16, 20, 30], [28, 16, 42, 30], [14, 20, 34, 38], [8, 30, 22, 43], [26, 30, 40, 43]]
    ], dtype=torch.float32).repeat(B, 1, 1).to(device)
    
    print("Testing 3-channel input shape (2, 3, 48, 48)...")
    with torch.no_grad():
        backbone_feat = model.backbone(dummy_input_3)
        print(f"[Backbone Output Shape]: {backbone_feat.shape}")
        assert backbone_feat.shape == (B, 256, 12, 12), "Backbone shape mismatch!"
        
        out_dict = model(dummy_input_3, bboxes=bboxes)
        
    print(f"[Model Total Output Shape]: {out_dict['logits'].shape}")

    # 2. Memory & Speed Test
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Warmup
    print("\nWarming up...")
    for _ in range(3):
        out_dict = model(dummy_input_3, bboxes=bboxes)
        loss = out_dict['logits'].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    if use_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        
    start = time.time()
    iters = 10
    
    print(f"Running {iters} iterations of Forward + Backward...")
    for _ in range(iters):
        out_dict = model(dummy_input_3, bboxes=bboxes)
        loss = out_dict['logits'].sum()
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
    if use_cuda:
        torch.cuda.synchronize()
        
    duration = time.time() - start
    print(f">>> Time for {iters} training steps: {duration:.4f} seconds")
    if use_cuda:
        print(f">>> Peak CUDA memory: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")

if __name__ == "__main__":
    use_cuda = torch.cuda.is_available()
    print(f"Using CUDA: {use_cuda}")
    try:
        test_model("resnet50", use_cuda)
        test_model("hrnet_w18", use_cuda)
    except Exception as e:
        import traceback
        traceback.print_exc()
