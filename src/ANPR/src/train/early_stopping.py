import torch
import os 

class EarlyStopping():
    def __init__(self, model, eps, threshold, savePath="backupModel.pth"):
        """
        Initialize the EarlyStopping object.

        Args:
            model (torch.nn.Module): The model to be monitored.
            eps (float): Minimum improvement required to update the best loss.
            threshold (int): Number of epochs with no improvement before stopping.
            savePath (str): Path to save the best model's state_dict.
        """
        self.eps = eps
        self.threshold = threshold
        self.count = 0
        self.savePath = savePath
        self.model = model
        self.best = torch.inf

    def getNoImprovement(self):
        return self.count

    def __call__(self, loss):
        if loss < self.best - self.eps:
            # Save the model's state_dict if loss improves
            torch.save(self.model.state_dict(), self.savePath)
            self.count = 0
            self.best = loss
        else:
            # Increment count and check for threshold
            self.count += 1
            if self.count > self.threshold:
                return True #stop!
        
        return False