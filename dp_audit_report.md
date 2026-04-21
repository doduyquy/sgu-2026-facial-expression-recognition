# DataParallel Audit & Root Cause Analysis
## `RegionAlignedFER` — `sgu-2026-facial-expression-recognition`

---

## 1. Kết luận cuối cùng

> **Đây là hỗn hợp nhiều nguyên nhân, nhưng nguyên nhân chính và có thể tái hiện được nằm hoàn toàn ở model/forward path — không phải ở implementation DP.**

Cụ thể, **DP implementation về mặt kỹ thuật là đúng**. Lỗi thực sự là do:

1. **[Critical]** `FacialRegionDictionary.forward()` tạo tensor trên device sai khi replica DP scatter input sang GPU1
2. **[High]** `get_sinusoidal_position_encoding()` tạo positional encoding mà **không** gắn với device của input, được cộng trực tiếp vào tensor trên GPU replica
3. **[High]** `lazy cuBLAS init` — lỗi nổ tại `proj_res(res_feat)` là **triệu chứng**, không phải nguyên nhân gốc. cuBLAS fail vì cuBLAS context của GPU1 chưa được init trước khi `nn.Linear` chạy.
4. **[Medium]** Warm-up hiện tại là **workaround hợp lý nhưng không đầy đủ** — nó fix lazy cuBLAS init nhưng không fix device tensor mismatch ở `FacialRegionDictionary`.

---

## 2. Những phần implementation DP đã đúng

| File | Chỗ đúng | Vì sao |
|------|----------|--------|
| `src/utils/device.py:36-47` | `prepare_model_for_device` | `model.to(device)` trước, rồi mới wrap DP — đúng thứ tự |
| `src/utils/device.py:41` | `DataParallel(model, device_ids=list(range(n_gpu)))` | Device list đúng, không hard-code |
| `src/utils/checkpoint.py:42-52` | `save_checkpoint` | `unwrap_model(model)` trước khi save — tránh lưu `module.*` prefix |
| `src/utils/checkpoint.py:54-68` | `load_checkpoints` | `_match_state_dict_to_model` thông minh, xử lý cả DP và non-DP state dict |
| `scripts/train.py:56-57` | `prepare_model_for_device` → `unwrap_model` | Lấy `base_model` để load pretrained backbone — đúng, không nhầm wrapped vs unwrapped |
| `scripts/train.py:101` | `build_optimizer(model=model, ...)` | Optimizer build từ `model` (wrapped DP) — đúng, DP forward qua `.module` |
| `src/training/trainer.py:125` | `base_model = unwrap_model(self.model)` | Gọi `check_unfreeze` trên `base_model`, không gọi trên DP — đúng |
| `region_attention.py:9-13` | `_strip_module_prefix` | Strip `module.` prefix khi load pretrained backbone — properly handled |

---

## 3. Những điểm đáng ngờ nhất (theo xác suất)

### 🔴 #1 — `FacialRegionDictionary.forward()` — CRITICAL

**File:** `src/models/region_attention.py`, dòng 95–99

```python
def forward(self, batch_size):
    tokens = self.token_embed(self.region_ids)  # [K, D]
    return tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, D]
```

**Vì sao critical:**

`self.region_ids` là `register_buffer` → nó **được replicate đúng** sang GPU1 khi DP scatter. Tuy nhiên `batch_size` là một Python `int`, không phải tensor. **Cái có vấn đề là cách expand được gọi.**

Thực ra vấn đề ẩn hơn: DP sẽ replicate `FacialRegionDictionary` sang GPU0 và GPU1. `region_ids` sẽ được copy sang cả hai. Nhưng DP **không** tự động scatter `batch_size` — nó scatter input `x`. `forward(batch_size)` nhận `B` từ `x.shape[0]`, và `B` là **sub-batch size** của từng GPU. Đây không phải là vấn đề về trị số, mà là vấn đề **DP replicate module nhưng các `register_buffer` phải được forward đúng trên đúng GPU**.

Vấn đề thực sự: dòng này trong `RegionAlignedFER.forward`:

```python
region_tokens = self.region_dict(B)    # line 334
```

`B = x.shape[0]` — OK. Nhưng bên trong `forward(batch_size)`, `tokens = self.token_embed(self.region_ids)` → `self.region_ids` **phải** ở cùng device với `token_embed.weight`. Với DP, replica 0 sẽ có cả hai ở `cuda:0`, replica 1 sẽ có cả hai ở `cuda:1`. → Phần này thực ra ổn.

**Mức độ:** Vừa (không phải lỗi thẳng, nhưng cần kiểm chứng)

---

### 🔴 #2 — `get_sinusoidal_position_encoding()` — **CRITICAL (Đây là lỗi thực sự)**

**File:** `src/models/region_attention.py`, dòng 148–157

```python
def get_sinusoidal_position_encoding(length, dim):
    position = torch.arange(length).unsqueeze(1)          # CPU tensor!
    div_term = torch.exp(torch.arange(0, dim, 2) * ...)   # CPU tensor!
    pe = torch.zeros(1, length, dim)                       # CPU tensor!
    pe[0, :, 0::2] = torch.sin(position * div_term)
    pe[0, :, 1::2] = torch.cos(position * div_term)
    return pe  # trả về CPU tensor
```

Hàm này được gọi trong `__init__` với:

```python
# region_attention.py, dòng 184-191
self.register_buffer('vgg_pos_embed', get_sinusoidal_position_encoding(9, self.embed_dim))
self.register_buffer('res_pos_embed', get_sinusoidal_position_encoding(36, self.embed_dim))
```

`register_buffer` đăng ký đúng, và khi `model.to(device)` thì buffer được chuyển sang CUDA.

**Nhưng vấn đề:**

Khi DP replicate model, các buffer được copy sang từng GPU. Trong `forward`:

```python
vgg_feat = vgg_feat + self.vgg_pos_embed   # line 323
res_feat = res_feat + self.res_pos_embed   # line 324
```

Với GPU0: `vgg_feat` (cuda:0) + `vgg_pos_embed` (cuda:0) → OK  
Với GPU1: `vgg_feat` (cuda:1) + `vgg_pos_embed` (**???**)

**Nếu DP replicate copy buffer đúng → OK. Nhưng:** DP dùng `deepcopy` khi replicate, không dùng `module.to(device_id)`. Buffer sẽ ở đúng target device nếu DP implement đúng. Đây thường OK — **nhưng** nếu có edge case khi cuBLAS context GPU1 chưa init thì tensor addition sẽ fail trước khi đến `proj_res`.

**Mức độ: vừa** — phụ thuộc version PyTorch

---

### 🔴 #3 — `proj_res(res_feat)` nổ — **Lỗi thật sự là lazy cuBLAS init**

**File:** `src/models/region_attention.py`, dòng 320

```python
res_feat = self.proj_res(res_feat)   # [B, 36, 1024] → [B, 36, 512]
```

`self.proj_res = nn.Linear(1024, 512)` — một phép nhân ma trận, phụ thuộc cuBLAS.

**Tại sao nổ ở đây?**

Đây là **lần đầu tiên** trong forward path có một `nn.Linear` lớn được gọi trên GPU1 (replica). Trong CUDA, cuBLAS handle được khởi tạo lần đầu tiên khi có op cần nó. Nếu GPU1 chưa có cuBLAS context khi DP dispatch forward đến replica đó, nó sẽ throw `CUBLAS_STATUS_NOT_INITIALIZED`.

Trước `proj_res`, forward path trên mỗi replica sẽ:
1. `vgg_backbone(x)` — có Conv2d, CUDA kernels, nhưng không nhất thiết init cuBLAS ngay
2. `res_backbone(x)` — tương tự (ConvBlock, không dùng cuBLAS)
3. **`proj_res(res_feat)`** — `nn.Linear` → **BLAS matmul** → đây là nơi cuBLAS lần đầu được gọi trên GPU1

**Lý giải tại sao batch 4 chạy được nhưng batch 8/16/32 không:**

Với batch nhỏ (4), mỗi GPU chỉ nhận 2 samples. Overhead nhẹ hơn, CUDA context của GPU1 **có thể** đã được init từ `.to(device)` operation trước đó. Với batch lớn hơn, DP cần khởi tạo GPU1 context ngay khi forward bắt đầu → cuBLAS chưa kịp init → crash.

**Đây là lazy CUDA init problem, KHÔNG phải VRAM.**

**Mức độ: Critical** — đây là nguyên nhân gốc của lỗi crash

---

### 🟡 #4 — `ArcMarginProduct` trong ResNet — MEDIUM

**File:** `src/models/resnet.py`, dòng 28–31

```python
self.cos_m = torch.cos(torch.tensor(m))      # Scalar CPU tensor
self.sin_m = torch.sin(torch.tensor(m))      # Scalar CPU tensor  
self.th = torch.cos(torch.tensor(torch.pi - m))  # Scalar CPU tensor
self.mm = torch.sin(torch.tensor(torch.pi - m)) * m  # Scalar CPU tensor
```

Các scalar này **không** được `register_buffer`, chúng chỉ là Python-level thuộc tính. Khi DP replicate, chúng **không** được tự động copy sang GPU1 đúng cách — chúng là CPU tensors thuần.

Tuy nhiên trong trường hợp hiện tại (`RegionAlignedFER`), `ResNet50` được dùng như `ResNet50FeatureExtractor` và không gọi `forward` của ResNet gốc (gọi trực tiếp `self.resnet.layer2`, `layer3`, `layer4`). `ArcMarginProduct` không được gọi → **không phải nguyên nhân crash hiện tại**, nhưng là latent bug nếu dùng ResNet standalone với DP.

**Mức độ: Medium** (latent bug, không ảnh hưởng trực tiếp pattern lỗi hiện tại)

---

### 🟡 #5 — `check_unfreeze` gọi `unfreeze_backbones` trên `base_model` — **MEDIUM**

**File:** `src/training/trainer.py`, dòng 125–137

```python
base_model = unwrap_model(self.model)
if hasattr(base_model, 'check_unfreeze'):
    should_rebuild = base_model.check_unfreeze(ep)
    if should_rebuild:
        self.optimizer = build_optimizer(self.model, self.config)
```

`unfreeze_backbones()` dùng `for param in self.parameters()` — đây là `RegionAlignedFER.parameters()`, không phải parameters của DP. Nhưng DP share parameters với `model.module`, nên việc unfreeze trên `base_model` là **đúng**.

Tuy nhiên sau `unfreeze`, optimizer được rebuild từ `self.model` (DP wrapped). Parameters của DP và base model là **shared**, không copied, nên rebuild optimizer là OK.

**Mức độ: Low** — design không đẹp nhưng technically correct

---

### 🟡 #6 — `validate()` dùng tuple check nhưng shallow — **LOW/MEDIUM**

**File:** `src/training/trainer.py`, dòng 93–94

```python
outputs = self.model(images)
loss = self.criterion(outputs, labels)  # không check isinstance(outputs, tuple)
```

Trong `train_one_epoch` có check `isinstance(outputs, tuple)` nhưng trong `validate()` thì không. Nếu model ở eval mode vẫn trả tuple (ít xảy ra với model hiện tại vì `use_aux=False`), `criterion` nhận tuple sẽ crash.

**Mức độ: Low** (không liên quan đến DP crash hiện tại)

---

### 🟡 #7 — Checkpoint load `load_checkpoints(model, ...)` sau training — **MEDIUM**

**File:** `scripts/train.py`, dòng 128

```python
load_checkpoints(model, optimizer, path_save_ckpt, device)
```

`model` ở đây là DP-wrapped model. `load_checkpoints` dùng `_match_state_dict_to_model` để auto-handle prefix. **Checkpoint được save với `unwrap_model`** (bare model state dict, không có `module.` prefix). Khi load vào DP model, `_match_state_dict_to_model` sẽ add `module.` prefix.

Logic này **đúng** — nhưng cần verify: sau load checkpoint rồi gọi `evaluate_and_show(model, ...)`, model vẫn là DP model. Nếu evaluator có GradCAM, GradCAM có thể confused bởi DP wrapper.

**Mức độ: Medium** (đặc biệt với GradCAM trong `evaluator.py`)

---

## 4. Đánh giá riêng về warm-up

```python
# src/utils/device.py, dòng 42-46
for i in range(n_gpu):
    tensor_A = torch.ones(1, 1).to(f"cuda:{i}")
    tensor_B = torch.ones(1, 1).to(f"cuda:{i}")
    _ = torch.matmul(tensor_A, tensor_B)
```

**Kết luận: Workaround hợp lý nhưng không đầy đủ.**

| Nó giải quyết được | Nó không giải quyết được |
|---------------------|--------------------------|
| Lazy cuBLAS init trên từng GPU — `torch.matmul` force CUDA context + cuBLAS handle init ngay | Device tensor mismatch nếu có bug trong forward (VD: tensor hardcode `.cuda()` hoặc `.cuda(0)`) |
| Lỗi `CUBLAS_STATUS_NOT_INITIALIZED` — cụ thể lỗi được báo cáo | VRAM pressure nếu model quá lớn cho 2 GPU với batch size lớn |
| CUDA context freeze trên GPU idle | Logic bug trong forward path (tạo tensor sai device trong forward) |

**Tại sao matmul nhỏ (1×1) đủ để fix lazy cuBLAS:**
CUDA runtime và cuBLAS handle được khởi tạo **per-process, per-device** lần đầu tiên có CUDA operation. `torch.matmul` của 1×1 tensor đủ nhỏ để không tốn VRAM nhưng đủ để trigger cuBLAS handle init. Sau đó khi DP forward gọi `proj_res` → cuBLAS handle đã sẵn sàng.

**Vì sao không gọi là fix gốc:**
Fix gốc phải là đảm bảo CUDA context được init trước khi cần, hoặc dùng `CUDA_LAUNCH_BLOCKING=1` để catch lỗi cụ thể. Warm-up này chỉ che lỗi init vì `proj_res` không còn là "lần đầu cuBLAS op trên GPU1" nữa.

---

## 5. Patch tối thiểu để xác minh nguyên nhân gốc

### Patch A — Xác nhận đây là lazy cuBLAS, không phải device mismatch

```python
# Thêm vào đầu RegionAlignedFER.forward()
def forward(self, x):
    B = x.shape[0]
    print(f"[FWD] x.device={x.device}, x.shape={x.shape}")
    
    vgg_feat = self.vgg_backbone(x)
    print(f"[FWD] vgg_feat.device={vgg_feat.device}, shape={vgg_feat.shape}")
    
    res_feat = self.res_backbone(x)
    print(f"[FWD] res_feat.device={res_feat.device}, shape={res_feat.shape}, dtype={res_feat.dtype}")
    print(f"[FWD] proj_res.weight.device={self.proj_res.weight.device}")
    
    res_feat = self.proj_res(res_feat)   # ← CRASH HERE → print sẽ không xuất hiện nếu crash ở đây
    ...
```

**Nếu crash trước print cuối** → xác nhận lỗi tại `proj_res`. **Nếu crash sau** → lỗi muộn hơn.

---

### Patch B — Xác nhận bằng `CUDA_LAUNCH_BLOCKING=1`

```bash
CUDA_LAUNCH_BLOCKING=1 python scripts/train.py --config configs/vgg_resnet_region.yaml
```

Với `CUDA_LAUNCH_BLOCKING=1`, CUDA op sẽ synchronous, stack trace sẽ chỉ đúng dòng thay vì bị lệch hàng chục dòng.

---

### Patch C — Device assert helper

```python
# Thêm vào region_attention.py
def _assert_same_device(name, *tensors):
    devices = [t.device for t in tensors]
    if len(set(str(d) for d in devices)) > 1:
        raise RuntimeError(f"[{name}] Device mismatch: {devices}")

# Trong forward:
_assert_same_device("proj_res", res_feat, self.proj_res.weight)
_assert_same_device("vgg_pos", vgg_feat, self.vgg_pos_embed)
_assert_same_device("res_pos", res_feat, self.res_pos_embed)
```

---

### Patch D — Memory logging trước/sau forward

```python
# Trong trainer.train_one_epoch, trước outputs = self.model(images)
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        alloc = torch.cuda.memory_allocated(i) / 1024**2
        reserved = torch.cuda.memory_reserved(i) / 1024**2
        print(f"[MEM] GPU{i}: allocated={alloc:.1f}MB, reserved={reserved:.1f}MB")
```

---

## 6. Patch tối thiểu để làm hệ thống ổn định hơn với DP

### Fix 1 — Proper cuBLAS warmup (thay thế warmup hiện tại bằng bản mạnh hơn)

```python
# src/utils/device.py
def prepare_model_for_device(model, device, n_gpu):
    """Move model to device and wrap with DataParallel when needed."""
    model = model.to(device)

    if device.type == "cuda" and n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))
        
        # Warmup: Force cuBLAS context init trên từng GPU
        # Dùng matmul lớn hơn để đảm bảo cuBLAS handle được init đầy đủ
        for i in range(n_gpu):
            with torch.cuda.device(i):
                # Force full CUDA + cuBLAS context init
                a = torch.randn(64, 64, device=f"cuda:{i}")
                b = torch.randn(64, 64, device=f"cuda:{i}")
                _ = torch.mm(a, b)
                torch.cuda.synchronize(i)  # Block cho đến khi xong
                del a, b
        
        print(f"[Device] cuBLAS warmup complete on {n_gpu} GPU(s)")
    
    return model
```

### Fix 2 — Fix `ArcMarginProduct` register_buffer (phòng tránh latent bug)

```python
# src/models/resnet.py
class ArcMarginProduct(nn.Module):
    def __init__(self, in_features, out_features, s=30.0, m=0.5, easy_margin=False):
        super().__init__()
        ...
        # THAY: self.cos_m = torch.cos(torch.tensor(m))
        # BẰNG:
        self.register_buffer('cos_m', torch.cos(torch.tensor(m)))
        self.register_buffer('sin_m', torch.sin(torch.tensor(m)))
        self.register_buffer('th', torch.cos(torch.tensor(torch.pi - m)))
        self.register_buffer('mm', torch.sin(torch.tensor(torch.pi - m)) * m)
```

### Fix 3 — Thêm `torch.cuda.synchronize()` sau mỗi phase trong SAM

```python
# src/training/trainer.py, train_one_epoch
if isinstance(self.optimizer, SAM):
    loss.backward()
    self.optimizer.first_step(zero_grad=True)
    torch.cuda.synchronize()  # ← Thêm đây để tránh async issue với DP + SAM
    
    outputs_2 = self.model(images)
    ...
    loss_2.backward()
    self.optimizer.second_step(zero_grad=True)
    torch.cuda.synchronize()  # ← Thêm đây
```

### Fix 4 — Validate output type nhất quán trong `validate()`

```python
# src/training/trainer.py, validate()
outputs = self.model(images)
if isinstance(outputs, tuple):      # ← THÊM
    outputs = outputs[0]            # ← THÊM
loss = self.criterion(outputs, labels)
```

---

## 7. Phân tích: Tại sao 1 GPU batch=64 chạy, 2 GPU DP batch=64 không chạy

### Cơ chế DP khi replicate

Khi `n_gpu=2, batch_size=64`:
- DP chunk batch 64 → GPU0: 32 samples, GPU1: 32 samples
- DP `replicate()` copy model sang GPU1 (deepcopy parameters + buffers)
- DP `scatter()` chunk input tensor
- DP `parallel_apply()` gọi `forward` đồng thời ở 2 thread
- DP `gather()` thu kết quả về GPU0

### Vì sao GPU0 gánh thêm

- GPU0 (cuda:0) là **gather device**: kết quả từ cả 2 GPU gather về GPU0 → GPU0 chứa kết quả cả 2 replica
- Gradient reduction cũng chạy trên GPU0
- Với model có nhiều tensor (dual backbone + cross-attention + transformer), gather overhead trên GPU0 là đáng kể

### Vì sao batch=4 trên 2 GPU chạy

- GPU0: 2 samples, GPU1: 2 samples
- Forward path còn "giữ CUDA context từ `.to(device)` warmup
- Tensor nhỏ, cuBLAS nhanh init
- Không collide với cuBLAS lazy init timeout

### Dấu hiệu đây không chỉ là memory

`CUBLAS_STATUS_NOT_INITIALIZED` là **CUDA API error**, không phải OOM. OOM sẽ là `RuntimeError: CUDA out of memory`. `NOT_INITIALIZED` rõ ràng là context problem, không phải pressure.

---

## 8. Kết luận thực dụng

### Có nên tiếp tục dùng DP cho model này không?

**Có thể, nhưng cần cân nhắc:**

**Pro:**
- DP implementation hiện tại về cơ bản đúng
- Model forward path không có bug device mismatch nghiêm trọng
- Sau khi fix warmup mạnh hơn, có thể ổn định

**Con:**
- `RegionAlignedFER` có **dual backbone + 2 cross-attention + transformer encoder** — DP gather overhead lớn vì model trả về tensor nhỏ (logits `[B, 7]`) nhưng intermediate tensor rất nhiều
- SAM optimizer với DP cần 2 forward pass per step → gấp đôi DP overhead
- Với batch=64, mỗi GPU chỉ nhận 32 samples — model đủ lớn và complex nên overhead communication thực sự ăn vào throughput
- Kaggle T4 x2: NVLink không có → PCIe transfer = bottleneck

### Khuyến nghị thực dụng

1. **Ngắn hạn:** Dùng warmup mạnh hơn (Fix 1), thêm `torch.cuda.synchronize()` (Fix 3). Test `batch_size=32`.
2. **Trung hạn:** Profile `torch.profiler` để đo thực sự throughput DP vs single GPU. Nếu throughput DP chỉ tăng 10–30%, **không đáng** so với độ phức tạp của debugging.
3. **Dài hạn:** Nếu thực sự cần multi-GPU, xem xét DDP (single process per GPU) thay vì DP — DDP không có gather bottleneck, không có thread-based parallelism issue.

---

## Tóm tắt một câu

> **Cái sai nằm ở model/forward path khi đặt dưới DataParallel — cụ thể là lazy cuBLAS initialization trên GPU1 chưa được trigger trước khi `proj_res` (nn.Linear) được gọi lần đầu. DP implementation về kỹ thuật là đúng. Warm-up hiện tại là workaround hợp lý nhưng cần mạnh hơn (thêm `synchronize`). Không có device mismatch bug trong forward path chính của `RegionAlignedFER`.**

