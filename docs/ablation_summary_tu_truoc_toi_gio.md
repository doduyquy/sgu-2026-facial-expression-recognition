# Bang thong ke ablation tu truoc toi gio

Cap nhat: 2026-05-22 19:24 +07.

Pham vi: tong hop cac ablation/ket qua quan trong dang co trong repo nay. Cot `Trang thai` phan biet ket qua da co artifact eval/checkpoint voi config moi tao nhung chua thay metric local.

## Moc so sanh nhanh

| Nhom | Ablation / model | Config / artifact chinh | Val best (%) | Test / eval (%) | Trang thai | Ghi chu |
|---|---|---|---:|---:|---|---|
| Baseline CNN | ConvNeXt-Tiny plain AdamW | `checkpoints/ensemble/convnext_tiny_fer2013_15052026_accuracy72.97.pth` | - | 72.97 | Da co checkpoint | Baseline ConvNeXt truoc SAM. |
| Baseline CNN | ConvNeXt-Tiny + SAM cosine | `configs/convnext_tiny_fer2013_sam_cosine.yaml` | 73.59 | 74.6169 | Da co eval | Moc manh nhat cua ConvNeXt CNN don le cu: `outputs/ensemble/eval_convnext_tiny_sam_test_20260516_183522/metrics_summary.json`. |
| ResNet region | ResNet152 region attention base | `checkpoints/ensemble/resnet152_region_attention_base_73.36.pth` | - | 73.36 | Da co checkpoint | Moc ResNet region base. |
| ResNet region | ResNet152 region attention SAM | `checkpoints/ensemble/resnet152_region_attention_20260513_accuracy73.47.pth` | - | 73.47 | Da co checkpoint/eval XAI | Dung trong ensemble va XAI. |
| ResNet region | ResNet152 layer3+4 SAM | `checkpoints/ensemble/resnet152_region_attention_sam_layer34_accuracy73.39.pth` | - | 73.39 | Da co checkpoint | Mot thanh vien ensemble. |
| ConvNeXt region | Region-attention gan checkpoint 74.62 | `checkpoints/convnext_tiny_region_attention_16052026_1636_best.pth` | - | ~74.51 | Da co checkpoint | Gan bang 74.62, chua phai win ro. |
| Mask-guided | Mask-guided ImageNet/stable | `checkpoints/version best/convnext_tiny_mask_guided_region_attention_17052026_75.01.pth` | 73.8162 | 75.0070 | Da co checkpoint/eval | Thanh vien `version best`, checkpoint 17/05. |
| Mask-guided | Mask-guided FER-seeded learned-only | `checkpoints/version best/convnext_tiny_mask_guided_region_attention_18052026_75.12.pth` | 73.5933 | 75.1184 | Da co checkpoint/eval | Thanh vien `version best`, checkpoint 18/05. |

## Ensemble / parameter search

| Nhom | Ablation | Artifact | Test (%) | Balanced (%) | Macro-F1 (%) | Trang thai | Ghi chu |
|---|---|---|---:|---:|---:|---|---|
| Ensemble 6 checkpoint | Best temperature/logit search 3 members | `outputs/ensemble/eval_6_checkpoint_parameter_search_20260516_191512/.../metrics_summary.json` | 76.0379 | 74.8461 | 75.4662 | Da co eval | Ket qua cao nhat trong `metrics_summary.json` hien co. |
| Ensemble 6 checkpoint | All-6 equal logit avg baseline | `outputs/ensemble/eval_6_checkpoint_parameter_search_20260516_191512/ensembles/baseline_all6_equal_logit_avg/metrics_summary.json` | 75.4249 | 74.2556 | 74.8063 | Da co eval | Baseline ensemble tat ca 6 checkpoint. |
| Ensemble 2 checkpoint | Version-best parameter search | `outputs/ensemble/eval_version_best_2_checkpoint_logits_20260520_095839` | 75.3692 | 74.0973 | 74.6538 | Da co eval | Best quanh 2 checkpoint 75.01 va 75.12, khong vuot 6-checkpoint search. |
| Ensemble 2 checkpoint | Version-best equal prob avg | `outputs/ensemble/eval_version_best_2_checkpoint_logits_20260520_095839/ensembles/baseline_equal_prob_avg/metrics_summary.json` | 75.2020 | 73.9399 | 74.5048 | Da co eval | Baseline cho notebook `version best`. |
| Ensemble 2 checkpoint | ResNet152 + ConvNeXt SAM temp search | `outputs/ensemble/eval_2_checkpoint_resnet152_convnext_sam_20260516_190327` | 75.5085 | 74.0880 | 74.9699 | Da co eval | 2-model ensemble co loi ich hon tung model don. |

## Nhanh ConvNeXt aux-CNN / CLIP / CNN-main moi

| Nhom | Ablation | Config / checkpoint | Val best checkpoint (%) | Val eval best (%) | Test eval best (%) | Trang thai | Ket luan nhanh |
|---|---|---|---:|---:|---:|---|---|
| CNN-main | CNN-main logits, fixed 0.8 CNN + 0.2 region | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits.yaml`; `checkpoints/convnext_tiny_mask_guided_region_attention_CNN0.8main.pth` | 73.7326 | 73.6138 | 74.2547 | Da co eval | Tot nhat trong nhom moi, nhung van thap hon ConvNeXt SAM 74.6169 va thap hon mask-guided 75.x. |
| CNN-main sweep | Logit fusion recommended CNN=0.8 | `outputs/evaluation/convnext_tiny_mask_guided_region_attention_CNN08main_weight_sweep/.../recommended_cnn_0.800` | 73.7326 | 73.6138 | 74.2547 | Da co eval | Bang voi fixed config; TTA hflip tot hon no-TTA. |
| CNN-main sweep | Prob fusion recommended CNN=0.65 | `outputs/evaluation/convnext_tiny_mask_guided_region_attention_CNN08main_weight_sweep/.../prob/recommended_cnn_0.650` | 73.7326 | 73.8089 | 73.8367 | Da co eval | Val nhin dep hon nhung test giam, khong nen chon. |
| CNN-base logits | Export/eval base CNN logits | `outputs/evaluation/convnext_tiny_mask_guided_region_attention_CNN08main_cnn_base_logits` | 73.7326 | 73.7531 | 74.0039 | Da co eval | Khong bang `CNN-main` 0.8 tren test. |
| Multi-scale SE | Stage3+stage4 SE fusion | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_multiscale_se.yaml`; `checkpoints/convnext_tiny_mask_guided_region_attention_SE.pth` | 73.0362 | 72.8615 | 73.8646 | Da co eval | Thap hon CNN-main, nen xem la ablation khong tot. |
| Multi-scale ECA | SE -> ECA, gate nhe hon | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_multiscale_eca.yaml` | - | - | - | Moi co config | Chua thay checkpoint/eval local. |
| Layer4-only SE | Chi gate feature layer4 | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_layer4_se.yaml` | - | - | - | Moi co config | Chua thay checkpoint/eval local. |
| Avg+Max full | CNN auxiliary pooling avg+max, one-file full config | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_avgmax_full.yaml` | - | - | - | Moi co config | Chua thay checkpoint/eval local. |
| ConvNeXt-Small | Doi backbone tiny -> small | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_convnext_small.yaml` | - | - | - | Moi co config | Chua thay checkpoint/eval local; canh OOM. |
| EfficientNet-B3 | Doi backbone ConvNeXt -> EfficientNet-B3 | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_efficientnet_b3.yaml` | - | - | - | Moi co config | Chua thay checkpoint/eval local; batch da giam 12. |
| Learnable fusion | Hoc trong so final-logit fusion | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_learnable_fusion.yaml` | - | - | - | Moi co config | Chua thay checkpoint/eval local. |

## Config da tao nhung chua co ket qua ro trong outputs

| Nhom | Config | Muc dich | Trang thai |
|---|---|---|---|
| Aux-CNN clean ImageNet | `configs/convnext_aux_cnn_one_run/kaggle_aux_cnn.yaml` | Aux CNN loss tren clean ImageNet-only line | Moi co config / chua thay eval local |
| Aux-CNN + CLIP | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip.yaml` | Bat CLIP + learnable region tokens | Moi co config / chua thay eval local |
| CNN logits nhe | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_logits.yaml` | Final logits them CNN auxiliary logits nhe | Moi co config / chua thay eval local |
| AvgMax compact | `configs/convnext_aux_cnn_clip_one_run/kaggle_aux_cnn_clip_cnn_main_logits_avgmax.yaml` | CNN-main voi avg+max pooling, file ke thua ngan | Moi co config / chua thay eval local |

## Nhan xet ngan

1. Neu tinh tat ca artifact trong repo, ket qua cao nhat hien tai van la ensemble search `76.0379%`.
2. Neu tinh single checkpoint/model, nhom mask-guided `75.01/75.12` dang cao hon ConvNeXt SAM `74.6169%`.
3. Nhanh CNN-main moi co ket qua tot nhat la `74.2547%`, chua vuot moc ConvNeXt SAM `74.6169%`.
4. Multi-scale SE dang giam diem, nen khong uu tien tiep neu muc tieu la diem so.
5. Cac config ECA, Layer4-SE, AvgMax full, ConvNeXt-Small, EfficientNet-B3, learnable fusion moi la ung vien tiep theo; chua nen ghi la co ket qua khi chua co checkpoint/eval.
