import tensorflow as tf
import numpy as np

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model, optimizer=None, extra_data=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, extra_data)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, optimizer, extra_data)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, optimizer=None, extra_data=None):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # Save TF checkpoint
        ckpt_kwargs = {'model': model}
        if optimizer is not None:
            ckpt_kwargs['optimizer'] = optimizer
        
        ckpt = tf.train.Checkpoint(**ckpt_kwargs)
        ckpt.write(self.path)
        
        # Save extra metadata as JSON if provided
        if extra_data is not None:
            import json
            meta_path = self.path + "_meta.json"
            serializable = {}
            for k, v in extra_data.items():
                if isinstance(v, (int, float, str, bool, list)):
                    serializable[k] = v
                elif isinstance(v, np.ndarray):
                    serializable[k] = v.tolist()
                elif isinstance(v, tf.Tensor):
                    serializable[k] = v.numpy().tolist()
            with open(meta_path, "w") as f:
                json.dump(serializable, f, indent=2)
            
        self.val_loss_min = val_loss