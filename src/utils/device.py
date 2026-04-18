import torch


def setup_device(config):
    """Resolve runtime device mode from config and current machine."""
    device_cfg = config.get("device", {})
    requested_n_gpu = int(device_cfg.get("n_gpu", 1))

    if requested_n_gpu <= 0:
        print("--> Device mode: CPU (config device.n_gpu <= 0)")
        return torch.device("cpu"), 0

    if not torch.cuda.is_available():
        print("--> CUDA is not available. Fallback to CPU.")
        return torch.device("cpu"), 0

    available_n_gpu = torch.cuda.device_count()
    actual_n_gpu = min(requested_n_gpu, available_n_gpu)

    if requested_n_gpu > available_n_gpu:
        print(
            f"--> Requested {requested_n_gpu} GPU(s), but only {available_n_gpu} available. "
            f"Use {actual_n_gpu} GPU(s) instead."
        )

    device = torch.device("cuda:0")

    if actual_n_gpu <= 1:
        print("--> Device mode: single GPU (cuda:0)")
    else:
        print(f"--> Device mode: DataParallel on {actual_n_gpu} GPU(s)")

    return device, actual_n_gpu


def prepare_model_for_device(model, device, n_gpu):
    """Move model to device and wrap with DataParallel when needed."""
    model = model.to(device)

    if device.type == "cuda" and n_gpu > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(n_gpu)))

    return model
