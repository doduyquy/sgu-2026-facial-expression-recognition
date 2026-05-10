import torch.optim as optim  
import torch.optim.lr_scheduler as lr_scheduler

def build_optimizer(model, config):
    train_cfg = config.get('training', {})
    opt_name = train_cfg.get('optimizer', 'adam').lower()
    lr = float(train_cfg.get('lr', train_cfg.get('learning_rate', 0.001)))
    weight_decay = float(train_cfg.get('weight_decay', 0.0001))

    params = model.parameters()

    if opt_name == 'adam':
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        gamma = train_cfg.get('gamma', 0.9) 
        return optim.SGD(params, lr=lr, weight_decay=weight_decay, momentum=gamma)
    # add another optimizer
    else:
        raise ValueError(f"Optimizer {opt_name} unsupported!")


def build_scheduler(optimizer, config):
    """Learning rate scheduler for model plateau | step | cosine"""
    scheduler_name = config['training'].get('scheduler', 'reduce_lr_on_plateau')
    if scheduler_name == 'none':
        return None

    elif scheduler_name == 'reduce_lr_on_plateau':
        # reduce when val loss stopping reduce
        factor = float(config['training'].get('lr_factor', 0.5)) 
        patience = int(config['training'].get('lr_patience', 3)) 
        print(f"--> [Scheduler] ReduceLROnPlateau (factor={factor}, patience={patience})")
        
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=factor,
            patience=patience,
        )
    elif scheduler_name == 'step':
        # decay(decrease) every n epochs
        step_size = int(config['training'].get('lr_step_size', 10))  
        gamma = float(config['training'].get('lr_gamma', 0.1))         
        print(f"--> [Scheduler] StepLR (step_size={step_size}, gamma={gamma})")
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_name == 'cosine':
        # decay with cosine
        T_max = config['training'].get('epochs', 100) 
        print(f"--> [Scheduler] CosineAnnealingLR (T_max={T_max})")
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    elif scheduler_name == 'cosine_annealing_warmup':
        warmup_epochs = int(config['training'].get('warmup_epochs', 5))
        total_epochs = int(config['training'].get('epochs', 100))
        
        print(f"--> [Scheduler] CosineAnnealingWarmup (Warmup={warmup_epochs}, Total={total_epochs})")
        
        # Warmup from 0.1 * lr to lr
        warmup_sch = lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
        # Cosine from lr down
        cosine_sch = lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs - warmup_epochs)
        
        return lr_scheduler.SequentialLR(
            optimizer, 
            schedulers=[warmup_sch, cosine_sch], 
            milestones=[warmup_epochs]
        )

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