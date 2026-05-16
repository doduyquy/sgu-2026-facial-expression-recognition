import os
import argparse
import numpy as np
import pandas as pd
from PIL import Image
import cv2

try:
    import mediapipe as mp
    # Import trực tiếp từ đường dẫn lõi python.solutions để tránh lỗi AttributeError/ImportError trên Kaggle
    from mediapipe.python.solutions import face_mesh as mp_face_mesh
except Exception as e:
    print(f"[WARNING] MediaPipe import failed with error: {e}")
    print("[WARNING] Please ensure mediapipe is installed correctly (pip install mediapipe).")
    mp = None
    mp_face_mesh = None

def parse_args():
    parser = argparse.ArgumentParser(description="Offline MediaPipe Landmark Detection for FER2013")
    parser.add_argument('--data_path', type=str, default='dataset/fer13-split', help='Path to FER2013 split directory')
    parser.add_argument('--output_dir', type=str, default='dataset/landmarks', help='Directory to save landmark CSV files')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test', 'all'], help='Dataset split to process')
    parser.add_argument('--vis_samples', type=int, default=20, help='Number of sample images to export with drawn landmark points')
    parser.add_argument('--vis_dir', type=str, default='dataset/landmarks_vis', help='Directory to save visualization images')
    return parser.parse_args()

def process_split(data_path, output_dir, split_name, vis_samples=20, vis_dir='dataset/landmarks_vis'):
    csv_path = os.path.join(data_path, f"{split_name}.csv")
    if not os.path.exists(csv_path):
        print(f"[ERROR] CSV file not found at {csv_path}")
        return

    print(f"\n--> [MediaPipe] Loading {split_name} dataset from {csv_path}...")
    df = pd.read_csv(csv_path, usecols=[0, 1])
    pixels_list = df.iloc[:, 1].tolist()
    
    # Định nghĩa 6 điểm neo sinh học tương ứng với MediaPipe Face Mesh (468 points)
    # 151: Trán (Forehead center)
    # 159: Mắt trái (Left eye upper center)
    # 386: Mắt phải (Right eye upper center)
    # 1: Chóp mũi (Nose tip)
    # 61: Mép trái (Left mouth corner)
    # 291: Mép phải (Right mouth corner)
    LANDMARK_INDICES = {
        'forehead': 151,
        'left_eye': 159,
        'right_eye': 386,
        'nose': 1,
        'left_mouth': 61,
        'right_mouth': 291
    }

    if mp_face_mesh is None:
        print("[ERROR] Cannot run detection without MediaPipe.")
        return

    # Sử dụng trực tiếp mô-đun mp_face_mesh đã import ở đầu file
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=False,
        min_detection_confidence=0.5
    )

    results_list = []
    successful_coords = []
    failed_indices = []

    print(f"--> Processing {len(pixels_list)} images for {split_name} (Resize 48->256, Detect, Scale 256->48)...")
    
    for idx, p in enumerate(pixels_list):
        # 1. Parse chuỗi pixel thành ảnh 48x48 numpy array (Loại bỏ DeprecationWarning)
        img_48 = np.array(p.split(), dtype=np.uint8).reshape(48, 48)
        
        # 2. Chuyển sang ảnh màu RGB và resize lên 256x256 để MediaPipe hoạt động chính xác
        img_rgb = cv2.cvtColor(img_48, cv2.COLOR_GRAY2RGB)
        img_256 = cv2.resize(img_rgb, (256, 256), interpolation=cv2.INTER_LINEAR)
        
        # 3. Chạy MediaPipe Face Mesh
        results = face_mesh.process(img_256)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            coords = {}
            for name, l_idx in LANDMARK_INDICES.items():
                # Tọa độ landmark trả về ở dạng chuẩn hóa [0.0, 1.0].
                # Nhân với 256 để lấy tọa độ trên ảnh 256x256, sau đó nhân với (48/256) để lùi về chuẩn 48x48.
                # Bản chất chính là nhân trực tiếp với 48!
                x_48 = landmarks[l_idx].x * 48.0
                y_48 = landmarks[l_idx].y * 48.0
                coords[f'x_{name}'] = x_48
                coords[f'y_{name}'] = y_48
            
            row = {'sample_id': idx, 'status': 'success'}
            row.update(coords)
            results_list.append(row)
            successful_coords.append([coords[k] for k in coords.keys()])
            
            if idx < vis_samples:
                # Vẽ 6 điểm neo lên ảnh 256x256 để kiểm chứng trực quan
                vis_img = img_256.copy()
                # Màu sắc BGR cho 6 điểm
                COLORS = {
                    'forehead': (255, 0, 0),    # Blue
                    'left_eye': (0, 255, 0),    # Green
                    'right_eye': (255, 255, 0), # Cyan
                    'nose': (0, 255, 255),      # Yellow
                    'left_mouth': (255, 0, 255),# Magenta
                    'right_mouth': (0, 0, 255)  # Red
                }
                for name, l_idx in LANDMARK_INDICES.items():
                    pt_x = int(landmarks[l_idx].x * 256.0)
                    pt_y = int(landmarks[l_idx].y * 256.0)
                    cv2.circle(vis_img, (pt_x, pt_y), radius=4, color=COLORS[name], thickness=-1)
                
                os.makedirs(vis_dir, exist_ok=True)
                vis_path = os.path.join(vis_dir, f"{split_name}_sample_{idx}_success.png")
                cv2.imwrite(vis_path, vis_img)
        else:
            # Ghi nhận thất bại để đắp tọa độ trung bình (Mean Coordinates) sau
            row = {'sample_id': idx, 'status': 'failed'}
            # Khởi tạo tạm bằng 0.0
            for name in LANDMARK_INDICES.keys():
                row[f'x_{name}'] = 0.0
                row[f'y_{name}'] = 0.0
            results_list.append(row)
            failed_indices.append(idx)
            
        if (idx + 1) % 5000 == 0 or (idx + 1) == len(pixels_list):
            print(f"    [Progress] Processed {idx + 1}/{len(pixels_list)} samples (Failed: {len(failed_indices)})")

    face_mesh.close()

    # 4. XỬ LÝ NGOẠI LỆ (FALLBACK TO MEAN COORDINATES)
    print(f"\n--> [Exception Handling] Calculating Mean Coordinates from {len(successful_coords)} successful detections...")
    if len(successful_coords) > 0:
        mean_matrix = np.mean(successful_coords, axis=0) # shape: (12,)
    else:
        # Fallback khẩn cấp về tọa độ sinh học lý tưởng trên lưới 48x48 nếu toàn bộ thất bại
        # 6 điểm neo trên lưới 12x12 tương ứng với tọa độ 48x48: (y*4, x*4)
        # [(2,5), (4,3), (4,8), (6,5), (9,3), (9,8)] -> nhân 4 -> y: [8, 16, 16, 24, 36, 36], x: [20, 12, 32, 20, 12, 32]
        mean_matrix = np.array([
            20.0, 8.0,   # forehead
            12.0, 16.0,  # left_eye
            32.0, 16.0,  # right_eye
            20.0, 24.0,  # nose
            12.0, 36.0,  # left_mouth
            32.0, 36.0   # right_mouth
        ])

    keys_order = [
        'x_forehead', 'y_forehead',
        'x_left_eye', 'y_left_eye',
        'x_right_eye', 'y_right_eye',
        'x_nose', 'y_nose',
        'x_left_mouth', 'y_left_mouth',
        'x_right_mouth', 'y_right_mouth'
    ]

    print(f"--> Mean Coordinates: {dict(zip(keys_order, np.round(mean_matrix, 2)))}")
    print(f"--> Applying Mean Coordinates to {len(failed_indices)} failed samples ({(len(failed_indices)/len(pixels_list))*100:.1f}%)...")

    for idx in failed_indices:
        for k_idx, key in enumerate(keys_order):
            results_list[idx][key] = mean_matrix[k_idx]

    # 5. Lưu kết quả ra file CSV
    os.makedirs(output_dir, exist_ok=True)
    out_csv = os.path.join(output_dir, f"landmarks_{split_name}.csv")
    df_out = pd.DataFrame(results_list)
    df_out.to_csv(out_csv, index=False)
    print(f"[SUCCESS] Saved landmark coordinates to {out_csv}")

if __name__ == "__main__":
    args = parse_args()
    
    if args.split == 'all':
        for s in ['train', 'val', 'test']:
            process_split(args.data_path, args.output_dir, s, vis_samples=args.vis_samples, vis_dir=args.vis_dir)
    else:
        process_split(args.data_path, args.output_dir, args.split, vis_samples=args.vis_samples, vis_dir=args.vis_dir)
