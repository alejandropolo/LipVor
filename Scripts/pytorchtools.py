# This code is based on the repository: https://github.com/Bjarten/early-stopping-pytorch/

import numpy as np
import torch

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If 2, prints a message for each validation loss improvement and early stopping count, else if 1, prints only early stopping count, else if 0, prints nothing.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print            
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        try:
            self.val_loss_min = np.Inf
        except:
            # For numpy>2.0.0
            self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, other_loss, model):
        score = -val_loss
        # Negate the validation loss to convert it into a score. The higher the score, the better.

        if self.best_score is None:
            # If this is the first iteration, there is no best score yet.
            self.best_score = score
            # So, set the current score as the best score.
            self.save_checkpoint(val_loss, other_loss, model)
            # And save the current model as a checkpoint.

        # elif score < self.best_score + self.delta or other_loss > 0:
        # Fix: Correct error_loss error
        elif score < self.best_score + self.delta:
            # If the current score is not significantly better than the best score (i.e., the improvement is less than delta)
            # or if the other loss is greater than 0,
            self.counter += 1
            # then increment the counter that keeps track of the number of iterations without significant improvement.

            if self.verbose>=1:
                # If verbose mode is enabled,
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
                # print the current state of the early stopping counter.

            if self.counter >= self.patience:
                # If the counter has reached the patience limit,
                self.early_stop = True
                # set the early stopping flag to True, indicating that training should be stopped.

        else:
            if other_loss==0:
                # If the current score is significantly better than the best score and the other loss is not greater than 0,
                self.best_score = score
                # update the best score to the current score,
                self.save_checkpoint(val_loss, other_loss, model)
                # save the current model as a checkpoint,
                self.counter = 0
                # and reset the counter.
    def save_checkpoint(self, val_loss, other_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose == 2:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss