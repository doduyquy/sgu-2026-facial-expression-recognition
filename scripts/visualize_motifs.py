import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
import yaml
import os
from src.models.motif_graph_fer import MotifGraphModel
from src.data.dataset import get_dataloaders # Giả định có hàm này

def visualize_inference(model_path, config_path, image_path=None):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Init model
    model = MotifGraphModel(config['model'])
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.to(device)
    model.eval()

    # Get sample image
    if image_path:
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (48, 48))
        img_tensor = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
    else:
        # Dummy or get from loader
        img_tensor = torch.randn(1, 1, 48, 48)
        img = (img_tensor[0,0].numpy() * 255).astype(np.uint8)

    img_tensor = img_tensor.to(device)

    # Forward
    with torch.no_grad():
        logits, top_k_idx, centers, scores = model(img_tensor, return_selection=True)
        probs = torch.softmax(logits, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()

    # Process results
    # centers is a list of (i, j) for ALL candidates
    # top_k_idx: (1, K)
    # scores: (1, num_cands, num_motifs)
    
    K = top_k_idx.shape[1]
    selected_indices = top_k_idx[0].cpu().numpy()
    
    # Class mapping (FER2013 standard)
    classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    
    plt.figure(figsize=(15, 5))
    
    # 1. Original Image with Selected Regions
    plt.subplot(1, 3, 1)
    plt.title(f"Prediction: {classes[pred_class]} ({probs[0, pred_class]:.2f})")
    display_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    
    # Draw selected centers (approximated from feature map grid to 48x48)
    # Feature map is roughly 6x6, so each pixel is ~8x8 in 48x48 image
    scale = 48 / 6 
    
    for idx in selected_indices:
        center_y, center_x = centers[idx]
        # Draw a 3x3 window (which is ~24x24 pixels in 48x48 image)
        y1, x1 = int((center_y-1) * scale), int((center_x-1) * scale)
        y2, x2 = int((center_y+2) * scale), int((center_x+2) * scale)
        cv2.rectangle(display_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        cv2.circle(display_img, (int((center_x+0.5)*scale), int((center_y+0.5)*scale)), 2, (255, 0, 0), -1)

    plt.imshow(display_img)
    plt.axis('off')

    # 2. Motif Matching Scores for the Best Region
    plt.subplot(1, 3, 2)
    best_cand_idx = selected_indices[0]
    cand_scores = scores[0, best_cand_idx].cpu().numpy()
    plt.bar(range(len(cand_scores)), cand_scores)
    plt.title(f"Motif Scores for Top Region")
    plt.xlabel("Motif Index")
    plt.ylabel("Similarity")

    # 3. Class-wise Similarity (Logits breakdown)
    plt.subplot(1, 3, 3)
    class_logits = logits[0].cpu().numpy()
    plt.bar(classes, class_logits, color='orange')
    plt.title("Similarity to Class Prototypes")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('motif_visualization.png')
    print("Visualization saved to motif_visualization.png")
    plt.show()

if __name__ == "__main__":
    # Update these paths to your actual files
    MODEL_PATH = "checkpoints/best_motif_model.pt"
    CONFIG_PATH = "configs/motif_config.yaml"
    visualize_inference(MODEL_PATH, CONFIG_PATH)