import os
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor

# 1. BỘ PROMPT GEOMETRY TRỰC GIAO & PHÂN VÙNG (8 Prompts/Class)
# Các vùng: [1. Lông mày, 2. Mắt, 3. Mũi/Má, 4. Miệng (Môi trên), 5. Miệng (Khóe môi), 6. Hàm/Cằm, 7. Trán, 8. Tổng thể cơ]
AU_PROMPTS = {
    0: [ # Anger (Tense, lowered, compressed)
        "lowered inner eyebrows pulled together", "narrowed eyes with tensed lower eyelids", "flared nostrils with tense cheeks", 
        "tightly pressed upper lip", "downward pulled mouth corners", "clenched jaw pointing forward",
        "deep vertical wrinkles between eyebrows", "rigid tense facial muscles"
    ],
    1: [ # Disgust (Wrinkled, asymmetrical, raised)
        "lowered eyebrows pulling down", "squinting eyes with pushed up lower lids", "deeply wrinkled nose bridge", 
        "asymmetrically raised upper lip", "tense mouth with exposed upper teeth", "raised cheeks creating nasolabial folds",
        "horizontal wrinkles across upper nose", "tense pulled-up central face muscles"
    ],
    2: [ # Fear (Tense, stretched horizontally, raised)
        "raised flat inner eyebrows", "widened eyes showing upper white sclera", "tense flared nostrils", 
        "horizontally stretched lips", "pulled back tense mouth corners", "jaw pulled backward",
        "horizontal wrinkles across forehead", "stiff tense facial structure"
    ],
    3: [ # Happiness (Relaxed, curved, upward)
        "neutral level eyebrows", "narrowed eyes with crinkled outer corners", "raised cheeks pushing up eyes", 
        "raised upper lip showing teeth", "sharply upturned smiling mouth corners", "relaxed dropped jaw",
        "smooth flat forehead", "relaxed expanded facial muscles"
    ],
    4: [ # Sadness (Drooping, drawn up inner, relaxed)
        "inner eyebrow corners drawn up and together", "drooping relaxed upper eyelids", "flattened relaxed cheeks", 
        "pouting protruding lower lip", "subtle downward curve of mouth corners", "trembling relaxed chin",
        "vertical tension lines between brows", "sagging downward facial muscles"
    ],
    5: [ # Surprise (Relaxed, round, arched)
        "highly arched raised separated eyebrows", "perfectly round widened eyes", "neutral relaxed nose", 
        "relaxed open oval mouth", "neutral unpulled mouth corners", "dropped open jaw with no tension",
        "curved horizontal wrinkles on high forehead", "completely relaxed extended facial structure"
    ],
    6: [ # Neutral (Flat, level, symmetrical)
        "flat level resting eyebrows", "naturally open relaxed eyes", "smooth flat resting cheeks", 
        "gently closed symmetrical lips", "level horizontal mouth corners", "relaxed closed jaw",
        "smooth unwrinkled forehead", "symmetrical expressionless facial muscles"
    ]
}

def generate_clip_embeddings(output_path="dataset/clip_au_embeddings.pt"):
    print("--> Starting Extractor...")
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    num_classes = 7
    motifs_per_class = 8 # Update to 8 Semantic Motifs
    embed_dim = 512

    model_name = "openai/clip-vit-base-patch16"
    print(f"Loading CLIP model ({model_name})...")
    model = CLIPModel.from_pretrained(model_name).eval()
    processor = CLIPProcessor.from_pretrained(model_name)

    templates = ["a facial expression showing {}", "a close-up photo of a face with {}", "a portrait showing {}"]
    flat_prompts = [tpl.format(au) for c in range(num_classes) for au in AU_PROMPTS[c] for tpl in templates]

    inputs = processor(text=flat_prompts, return_tensors="pt", padding=True, truncation=True)
    
    with torch.no_grad():
        out = model.get_text_features(**inputs) # (168, 512)
        if isinstance(out, torch.Tensor):
            raw_features = out
        elif hasattr(out, 'text_embeds'):
            raw_features = out.text_embeds
        elif hasattr(out, 'pooler_output'):
            raw_features = out.pooler_output
        elif isinstance(out, tuple):
            raw_features = out[0]
        else:
            raw_features = out
            
        # 1. Reshape & Average Templates
        raw_features = raw_features.view(num_classes * motifs_per_class, len(templates), embed_dim)
        averaged_features = raw_features.mean(dim=1) # (56, 512)
        
        # 2. ISOTROPIC CENTERING (PCA Whitening Light) - CỰC KỲ QUAN TRỌNG
        # Trừ đi mean của toàn bộ các vector để phân tán chúng ra xung quanh gốc tọa độ
        mean_feat = averaged_features.mean(dim=0, keepdim=True)
        centered_features = averaged_features - mean_feat
        
        # 3. Final Normalize
        clip_tensor = F.normalize(centered_features, p=2, dim=-1)
        clip_tensor = clip_tensor.view(num_classes, motifs_per_class, embed_dim) # (7, 8, 512)
        
    # Check tính trực giao (Orthogonality check)
    sim_matrix = torch.matmul(clip_tensor.view(56, 512), clip_tensor.view(56, 512).T)
    print(f"Mean inter-prompt similarity (should be close to 0): {sim_matrix.mean().item():.4f}")

    torch.save(clip_tensor, output_file)
    print(f"--> Saved at: {output_file.resolve()} | Shape: {clip_tensor.shape}")

if __name__ == "__main__":
    # Tìm gốc dự án
    root_dir = Path(__file__).resolve().parent.parent
    target_path = root_dir / "dataset" / "clip_au_embeddings.pt"
    generate_clip_embeddings(str(target_path))
