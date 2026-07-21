import torch
import torch.optim as optim  
import torch.optim.lr_scheduler as lr_scheduler

class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"

        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)

        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.defaults.update(self.base_optimizer.defaults)

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None: continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)
        self.first_step(zero_grad=True)
        closure()
        self.second_step()

    def _grad_norm(self):
        shared_device = None
        for group in self.param_groups:
            if len(group["params"]) > 0:
                shared_device = group["params"][0].device
                break
        if shared_device is None:
            return torch.tensor(0.0)
            
        grad_norms = [
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"]
            if p.grad is not None
        ]
        
        if len(grad_norms) == 0:
            return torch.tensor(0.0, device=shared_device)
            
        return torch.norm(torch.stack(grad_norms), p=2)

def build_optimizer(model, config):
    train_cfg = config.get('training', {})
    opt_name = train_cfg.get('optimizer', 'adam').lower()
    lr = float(train_cfg.get('lr', train_cfg.get('learning_rate', 0.001)))
    weight_decay = float(train_cfg.get('weight_decay', 0.0001))
    
    # SAM parameters
    sam_rho = float(train_cfg.get('sam_rho', 0.05))
    sam_adaptive = bool(train_cfg.get('sam_adaptive', False))

    # Differential learning rate for transfer learning
    backbone_params = []
    head_params = []
    for name, param in model.named_parameters():
        # Backbone ResNet layers get standard LR. New heads/reducers get 10x LR.
        if 'backbone' in name and 'dim_reducer' not in name and 'final_cbam' not in name:
            backbone_params.append(param)
        else:
            head_params.append(param)
            
    param_groups = [
        {'params': backbone_params, 'lr': lr},
        {'params': head_params, 'lr': lr * 2.0}
    ]

    if opt_name == 'adam':
        return optim.Adam(param_groups, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sgd':
        gamma = train_cfg.get('gamma', 0.9) 
        return optim.SGD(param_groups, lr=lr, weight_decay=weight_decay, momentum=gamma)
    elif opt_name == 'sam_adam':
        return SAM(param_groups, optim.Adam, rho=sam_rho, adaptive=sam_adaptive, lr=lr, weight_decay=weight_decay)
    elif opt_name == 'sam_sgd':
        gamma = train_cfg.get('gamma', 0.9) 
        return SAM(param_groups, optim.SGD, rho=sam_rho, adaptive=sam_adaptive, lr=lr, weight_decay=weight_decay, momentum=gamma)
    else:
        raise ValueError(f"Optimizer {opt_name} unsupported!")


def build_scheduler(optimizer, config):
    """Learning rate scheduler for model plateau | step | cosine"""
    scheduler_name = config['training'].get('scheduler', 'reduce_lr_on_plateau')
    if scheduler_name == 'none':
        return None

    elif scheduler_name == 'reduce_lr_on_plateau':
        factor = float(config['training'].get('lr_factor', 0.5))
        patience = int(config['training'].get('lr_patience', 5))
        # mode='max' because we track val_acc (higher is better)
        print(f"--> [Scheduler] ReduceLROnPlateau (mode=max, factor={factor}, patience={patience})")
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
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