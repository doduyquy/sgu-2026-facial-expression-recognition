import tensorflow as tf


def build_optimizer(model, config):
    """Build TF optimizer with differential learning rates.
    
    Args:
        model: tf.keras.Model
        config: dict with 'training' key
    
    Returns:
        tf.keras.optimizers.Optimizer
    
    Note: TF doesn't natively support per-parameter-group LR like PyTorch.
    Differential LR is handled in the training loop by applying LR multipliers
    when computing gradients. This function returns the base optimizer.
    """
    train_cfg = config.get('training', {})
    opt_name = train_cfg.get('optimizer', 'adam').lower()
    lr = float(train_cfg.get('lr', train_cfg.get('learning_rate', 0.001)))
    weight_decay = float(train_cfg.get('weight_decay', 0.0001))

    if opt_name == 'adam':
        optimizer = tf.keras.optimizers.Adam(
            learning_rate=lr,
            weight_decay=weight_decay if weight_decay > 0 else None,
        )
    elif opt_name == 'adamw':
        optimizer = tf.keras.optimizers.AdamW(
            learning_rate=lr,
            weight_decay=weight_decay,
        )
    elif opt_name == 'sgd':
        gamma = train_cfg.get('gamma', 0.9)
        optimizer = tf.keras.optimizers.SGD(
            learning_rate=lr,
            momentum=gamma,
            weight_decay=weight_decay if weight_decay > 0 else None,
        )
    else:
        raise ValueError(f"Optimizer {opt_name} unsupported!")

    return optimizer


def get_lr_multiplier(var_name, head_lr_mult=2.0):
    """Get learning rate multiplier for a variable based on its name.
    
    Backbone parameters get 1.0x LR, heads/reducers/cbam get head_lr_mult.
    Used in custom training loop to apply differential LR.
    """
    if 'backbone' in var_name and 'dim_reducer' not in var_name and 'final_cbam' not in var_name:
        return 1.0
    else:
        return head_lr_mult


def build_scheduler(optimizer, config):
    """Build learning rate schedule.
    
    Args:
        optimizer: tf.keras.optimizers.Optimizer
        config: dict with 'training' key
    
    Returns:
        tf.keras.optimizers.schedules.LearningRateSchedule or None
    
    Note: ReduceLROnPlateau in TF is a callback, not a schedule.
    For that case, returns a string identifier so the trainer can set up the callback.
    """
    scheduler_name = config['training'].get('scheduler', 'reduce_lr_on_plateau')
    
    if scheduler_name == 'none':
        return None

    elif scheduler_name == 'reduce_lr_on_plateau':
        factor = float(config['training'].get('lr_factor', 0.5))
        patience = int(config['training'].get('lr_patience', 5))
        print(f"--> [Scheduler] ReduceLROnPlateau (mode=max, factor={factor}, patience={patience})")
        # Return config dict — trainer will implement manual plateau detection
        return {
            'type': 'reduce_lr_on_plateau',
            'mode': 'max',
            'factor': factor,
            'patience': patience,
            'min_lr': float(config['training'].get('min_lr', 1e-6)),
        }

    elif scheduler_name == 'step':
        lr = float(config['training'].get('lr', config['training'].get('learning_rate', 0.001)))
        step_size = int(config['training'].get('lr_step_size', 10))
        gamma = float(config['training'].get('lr_gamma', 0.1))
        print(f"--> [Scheduler] StepLR (step_size={step_size}, gamma={gamma})")
        
        # Create PiecewiseConstantDecay
        epochs = int(config['training'].get('epochs', 100))
        boundaries = list(range(step_size, epochs, step_size))
        values = [lr * (gamma ** i) for i in range(len(boundaries) + 1)]
        return tf.keras.optimizers.schedules.PiecewiseConstantDecay(
            boundaries=boundaries,
            values=values,
        )

    elif scheduler_name == 'cosine':
        lr = float(config['training'].get('lr', config['training'].get('learning_rate', 0.001)))
        T_max = config['training'].get('epochs', 101)
        print(f"--> [Scheduler] CosineDecay (T_max={T_max})")
        return tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=lr,
            decay_steps=T_max,
        )

    else:
        raise ValueError(f"Not supported this {scheduler_name} scheduler!")


class ReduceLROnPlateau:
    """Manual ReduceLROnPlateau implementation for custom training loops.
    
    TF's built-in ReduceLROnPlateau is a callback for model.fit() only.
    This class can be used in custom training loops.
    """
    
    def __init__(self, optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.mode = mode
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best = None
        self.wait = 0
    
    def step(self, metric):
        """Call at the end of each epoch with the metric value."""
        if self.best is None:
            self.best = metric
            return
        
        improved = (metric > self.best) if self.mode == 'max' else (metric < self.best)
        
        if improved:
            self.best = metric
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                old_lr = float(self.optimizer.learning_rate.numpy())
                new_lr = max(old_lr * self.factor, self.min_lr)
                self.optimizer.learning_rate.assign(new_lr)
                self.wait = 0
                print(f"[Scheduler] ReduceLROnPlateau: LR {old_lr:.6f} -> {new_lr:.6f}")


if __name__ == "__main__":
    # Test block
    print("Testing TF optimizers...")
    
    # 1. Test Adam
    config_adam = {
        'training': {
            'optimizer': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0005
        }
    }
    
    # Create a dummy model for testing
    dummy_model = tf.keras.Sequential([tf.keras.layers.Dense(2, input_shape=(10,))])
    opt = build_optimizer(dummy_model, config_adam)
    print(f"Test 1 - Adam: Success! Type: {type(opt)}")

    # 2. Test SGD
    config_sgd = {
        'training': {
            'optimizer': 'sgd',
            'lr': 0.01,
            'weight_decay': 0.0,
            'gamma': 0.95
        }
    }
    opt = build_optimizer(dummy_model, config_sgd)
    print(f"Test 2 - SGD: Success! Type: {type(opt)}")

    # 3. Test error handling
    config_error = {'training': {'optimizer': 'rmsprop', 'lr': 0.01, 'weight_decay': 0}}
    try:
        build_optimizer(dummy_model, config_error)
    except ValueError as e:
        print(f"Test 3 - Error Handling: Success! Caught: {e}")