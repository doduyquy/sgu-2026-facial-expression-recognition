import torch
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

def visualize_selected_subgraphs(model, image_tensor, label, emotion_dict, device):
    """
    image_tensor: (1, 1, 48, 48)
    """
    model.eval()
    model.to(device)
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        logits, top_k_idx, centers, scores = model(image_tensor, return_selection=True)
    
    preds = torch.argmax(logits, dim=1).item()
    top_k_idx = top_k_idx[0].cpu().numpy() # (top_k,)
    
    # Image for display
    img = image_tensor[0, 0].cpu().numpy()
    
    plt.figure(figsize=(10, 5))
    
    # 1. Show original image with selected regions
    plt.subplot(1, 2, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"True: {emotion_dict[label]}, Pred: {emotion_dict[preds]}")
    
    # Feature map size (H, W) is 6x6
    H_feat, W_feat = 6, 6
    patch_size = 48 // H_feat # 8
    
    for idx in top_k_idx:
        # idx is the index in the candidates list
        # candidates are extracted from centers (1,1) to (H-2, W-2)
        center_i, center_j = centers[idx]
        
        # Center in pixel coordinates
        y_c = center_i * patch_size + patch_size // 2
        x_c = center_j * patch_size + patch_size // 2
        
        # 3x3 patch in feature map corresponds to 3*patch_size in pixels
        half_box = (3 * patch_size) // 2
        
        rect = plt.Rectangle(
            (x_c - half_box, y_c - half_box), 
            3 * patch_size, 3 * patch_size, 
            fill=False, color='red', linewidth=2
        )
        plt.gca().add_patch(rect)
        
    # 2. Show Motif Similarity Heatmap
    plt.subplot(1, 2, 2)
    # scores: (1, num_cands, Total_Motifs)
    # Take max similarity across motifs of the predicted class
    pred_class_motifs = scores[0, :, preds*8 : (preds+1)*8] # (num_cands, 8)
    heatmap_vals = pred_class_motifs.max(dim=-1)[0].cpu().numpy()
    
    # Reshape heatmap to (H-2, W-2) = (4, 4)
    heatmap = heatmap_vals.reshape(4, 4)
    plt.imshow(heatmap, cmap='viridis', interpolation='bilinear')
    plt.title("Motif Matching Confidence")
    plt.colorbar()
    
    plt.tight_layout()
    save_path = "motif_visualization.png"
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    # plt.show()

if __name__ == "__main__":
    from src.data.emotions_dict import EMOTION_DICT
    from src.models.motif_graph_fer import MotifGraphModel
    
    config = {
        'feat_dim': 64,
        'num_classes': 7,
        'motifs_per_class': 8,
        'top_k': 4
    }
    model = MotifGraphModel(config)
    dummy_img = torch.randn(1, 1, 48, 48)
    visualize_selected_subgraphs(model, dummy_img, 3, EMOTION_DICT, 'cpu')
