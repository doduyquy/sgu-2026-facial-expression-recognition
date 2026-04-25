import torch.optim as optim  
import torch.optim.lr_scheduler as lr_scheduler

def build_optimizer(model, config):
    train_cfg = config.get('training', {})
    opt_cfg = config.get('optimizer', {}) or {}
    opt_name = opt_cfg.get('name', train_cfg.get('optimizer', 'adam')).lower()
    lr = opt_cfg.get('lr', train_cfg.get('lr', 0.001))
    weight_decay = opt_cfg.get('weight_decay', train_cfg.get('weight_decay', 0.0001))

    params = model.parameters()

    if opt_name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        gamma = train_cfg.get('gamma', 0.9) 
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=gamma)
    # add another optimizer
    else:
        raise ValueError(f"Optimizer {opt_name} unsupported!")


def build_scheduler(optimizer, config):
    """Learning rate scheduler for model plateau | step | cosine"""
    train_cfg = config.get('training', {})
    sched_cfg = config.get('scheduler', {}) or {}
    scheduler_name = sched_cfg.get('name', train_cfg.get('scheduler', 'reduce_lr_on_plateau'))
    scheduler_name_norm = str(scheduler_name).lower()
    if scheduler_name_norm == 'none':
        return None

    elif scheduler_name_norm in {'reduce_lr_on_plateau', 'reducelronplateau'}:
        # reduce when val loss stopping reduce
        factor = sched_cfg.get('factor', train_cfg.get('lr_factor', 0.5))
        patience = sched_cfg.get('patience', train_cfg.get('lr_patience', 3))
        mode = sched_cfg.get('mode', 'min')
        min_lr = sched_cfg.get('min_lr', 0.0)
        monitor = sched_cfg.get('monitor', train_cfg.get('monitor', 'val_loss'))
        print(
            f"--> [Scheduler] ReduceLROnPlateau "
            f"(mode={mode}, monitor={monitor}, factor={factor}, patience={patience})"
        )
        
        scheduler = lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )
        scheduler.monitor_key = monitor
        scheduler.monitor_mode = mode
        return scheduler
    elif scheduler_name_norm == 'step':
        # decay(decrease) every n epochs
        step_size = config['training'].get('lr_step_size', 10)  
        gamma = config['training'].get('lr_gamma', 0.1)         # Decrease 1/10
        print(f"--> [Scheduler] StepLR (step_size={step_size}, gamma={gamma})")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name_norm == 'cosine':
        # decay with cosine
        T_max = config['training'].get('epochs', 101) 
        print(f"--> [Scheduler] CosineAnnealingLR (T_max={T_max})")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    else:
        raise ValueError(f"Not supported this {scheduler_name} scheduler!") 



if __name__ == "__main__":
    import torch.nn as nn
    # 1. Tạo một model giả lập
    dummy_model = nn.Linear(10, 2)

    # 2. Test trường hợp Adam (Hợp lệ)
    config_adam = {
        'training': {
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0005
        }
    }
    opt_adam = build_optimizer(dummy_model, config_adam)
    print(f"Test 1 - Adam: Success! Type: {type(opt_adam)}")

    # 3. Test trường hợp SGD với Gamma (Hợp lệ)
    config_sgd = {
        'training': {
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 0.0,
            'gamma': 0.95
        }
    }
    opt_sgd = build_optimizer(dummy_model, config_sgd)
    print(f"Test 2 - SGD: Success! Momentum: {opt_sgd.param_groups[0]['momentum']}")

    # 4. Test trường hợp lỗi (Unsupported)
    config_error = {'training': {'optimizer': 'rmsprop', 'lr': 0.01, 'weight_decay': 0}}
    try:
        build_optimizer(dummy_model, config_error)
    except ValueError as e:
        print(f"Test 3 - Error Handling: Success! Caught: {e}")
