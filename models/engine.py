import random
from lightning import LightningModule
import numpy as np
from sklearn.metrics import classification_report, precision_recall_curve
from torch import nn
import os
import torch
import matplotlib.pyplot as plt
import wandb
import seaborn as sns
from lion_pytorch import Lion
from torch_ema import ExponentialMovingAverage
from utils.utils_model import pick_model
import constants as cst
from scipy.stats import mode


class Engine(LightningModule):
    def __init__(
        self,
        seq_size,
        horizon,
        max_epochs,
        model_type,
        is_wandb,
        experiment_type,
        lr,
        optimizer,
        dir_ckpt,
        num_features,
        dataset_type,
        num_layers=4,
        hidden_dim=256,
        num_heads=8,
        is_sin_emb=True,
        len_test_dataloader=None,
    ):
        super().__init__()
        self.seq_size = seq_size
        self.dataset_type = dataset_type
        self.horizon = horizon
        self.max_epochs = max_epochs
        self.model_type = model_type
        self.num_heads = num_heads
        self.is_wandb = is_wandb
        self.len_test_dataloader = len_test_dataloader
        self.lr = lr
        self.optimizer = optimizer
        self.dir_ckpt = dir_ckpt
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_features = num_features
        self.experiment_type = experiment_type
        self.model = pick_model(model_type, hidden_dim, num_layers, seq_size, num_features, num_heads, is_sin_emb, dataset_type) 
        self.ema = ExponentialMovingAverage(self.parameters(), decay=0.999)
        self.ema.to(cst.DEVICE)
        self.loss_function = nn.CrossEntropyLoss()
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []
        self.test_targets = []
        self.test_predictions = []
        self.test_proba = []
        self.val_targets = []
        self.val_loss = np.inf
        self.val_predictions = []
        self.min_loss = np.inf
        self.save_hyperparameters()
        self.last_path_ckpt = None
        self.first_test = True
        self.test_mid_prices = []
        
    def forward(self, x, batch_idx=None):
        output = self.model(x)
        return output
    
    def loss(self, y_hat, y):
        return self.loss_function(y_hat, y)
        
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.forward(x)
        batch_loss = self.loss(y_hat, y)
        batch_loss_mean = torch.mean(batch_loss)
        self.train_losses.append(batch_loss_mean.item())
        self.ema.update()
        if batch_idx % 1000 == 0:
            print(f'train loss: {sum(self.train_losses) / len(self.train_losses)}')
        return batch_loss_mean
    
    def on_train_epoch_start(self) -> None:
        print(f'learning rate: {self.optimizer.param_groups[0]["lr"]}')
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        # Validation: with EMA
        with self.ema.average_parameters():
            y_hat = self.forward(x)
            batch_loss = self.loss(y_hat, y)
            self.val_targets.append(y.cpu().numpy())
            self.val_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.val_losses.append(batch_loss_mean.item())
        return batch_loss_mean
        
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        mid_prices = ((x[:, 0, 0] + x[:, 0, 2]) // 2).cpu().numpy().flatten()
        self.test_mid_prices.append(mid_prices)
        # Test: with EMA
        if self.experiment_type == "TRAINING":
            with self.ema.average_parameters():
                y_hat = self.forward(x, batch_idx)
                batch_loss = self.loss(y_hat, y)
                self.test_targets.append(y.cpu().numpy())
                self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
                self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
                batch_loss_mean = torch.mean(batch_loss)
                self.test_losses.append(batch_loss_mean.item())
        else:
            y_hat = self.forward(x, batch_idx)
            batch_loss = self.loss(y_hat, y)
            self.test_targets.append(y.cpu().numpy())
            self.test_predictions.append(y_hat.argmax(dim=1).cpu().numpy())
            self.test_proba.append(torch.softmax(y_hat, dim=1)[:, 1].cpu().numpy())
            batch_loss_mean = torch.mean(batch_loss)
            self.test_losses.append(batch_loss_mean.item())
        return batch_loss_mean
    
    def on_validation_epoch_start(self) -> None:
        loss = sum(self.train_losses) / len(self.train_losses)
        self.train_losses = []
        # Store train loss for combined plotting
        self.current_train_loss = loss
        print(f'Train loss on epoch {self.current_epoch}: {loss}')
        
    def on_validation_epoch_end(self) -> None:
        self.val_loss = sum(self.val_losses) / len(self.val_losses)
        self.val_losses = []
        
        # model checkpointing
        if self.val_loss < self.min_loss:
            # if the improvement is less than 0.0002, we halve the learning rate
            if self.val_loss - self.min_loss > -0.002:
                self.optimizer.param_groups[0]["lr"] /= 2  
            self.min_loss = self.val_loss
            self.model_checkpointing(self.val_loss)
        else:
            self.optimizer.param_groups[0]["lr"] /= 2
        
        # Log losses to wandb (both individually and in the same plot)
        self.log_losses_to_wandb(self.current_train_loss, self.val_loss)
        
        # Continue with regular Lightning logging for compatibility
        self.log("val_loss", self.val_loss)
        print(f'Validation loss on epoch {self.current_epoch}: {self.val_loss}')
        targets = np.concatenate(self.val_targets)    
        predictions = np.concatenate(self.val_predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("val_f1_score", class_report["macro avg"]["f1-score"])
        self.log("val_accuracy", class_report["accuracy"])
        self.log("val_precision", class_report["macro avg"]["precision"])
        self.log("val_recall", class_report["macro avg"]["recall"])
        self.val_targets = []
        self.val_predictions = [] 
    
    def log_losses_to_wandb(self, train_loss, val_loss):
        """Log training and validation losses to wandb in the same plot."""
        if self.is_wandb:   
            # Log combined losses for a single plot
            wandb.log({
                "losses": {
                    "train": train_loss,
                    "validation": val_loss
                },
                "epoch": self.global_step
            })
    
    def on_test_epoch_end(self) -> None:
        targets = np.concatenate(self.test_targets)    
        predictions = np.concatenate(self.test_predictions)
        predictions_path = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "predictions")
        os.makedirs(os.path.dirname(predictions_path), exist_ok=True)
        np.save(predictions_path, predictions)
        class_report = classification_report(targets, predictions, digits=4, output_dict=True)
        print(classification_report(targets, predictions, digits=4))
        self.log("test_loss", sum(self.test_losses) / len(self.test_losses))
        self.log("f1_score", class_report["macro avg"]["f1-score"])
        self.log("accuracy", class_report["accuracy"])
        self.log("precision", class_report["macro avg"]["precision"])
        self.log("recall", class_report["macro avg"]["recall"])
        self.test_targets = []
        self.test_predictions = []
        self.test_losses = []  
        self.first_test = False
        test_proba = np.concatenate(self.test_proba)
        precision, recall, _ = precision_recall_curve(targets, test_proba, pos_label=1)
        self.plot_pr_curves(recall, precision, self.is_wandb) 
        
    def configure_optimizers(self):
        if self.model_type == "DEEPLOB":
            eps = 1
        else:
            eps = 1e-8
        if self.optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, eps=eps)
        elif self.optimizer == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=0.9)
        elif self.optimizer == 'Lion':
            self.optimizer = Lion(self.parameters(), lr=self.lr)
        return self.optimizer
    
    def _define_log_metrics(self):
        wandb.define_metric("val_loss", summary="min")

    def model_checkpointing(self, loss):        
        if self.last_path_ckpt is not None:
            os.remove(self.last_path_ckpt)
        filename_ckpt = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".pt"
                             )
        path_ckpt = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "pt", filename_ckpt)
        
        # Save PyTorch checkpoint
        with self.ema.average_parameters():
            self.trainer.save_checkpoint(path_ckpt)
            
            # Save ONNX model
            onnx_dir = os.path.join(cst.DIR_SAVED_MODEL, str(self.model_type), self.dir_ckpt, "onnx")
            os.makedirs(onnx_dir, exist_ok=True)
            
            onnx_filename = ("val_loss=" + str(round(loss, 3)) +
                             "_epoch=" + str(self.current_epoch) +
                             ".onnx"
                            )
            onnx_path = os.path.join(onnx_dir, onnx_filename)
            
            # Create dummy input with appropriate shape
            dummy_input = torch.randn(1, self.seq_size, self.num_features, device=self.device)
            
            # Export to ONNX
            try:
                torch.onnx.export(
                    self.model,                  # model being run
                    dummy_input,                 # model input (or a tuple for multiple inputs)
                    onnx_path,                   # where to save the model
                    export_params=True,          # store the trained parameter weights inside the model file
                    opset_version=12,            # the ONNX version to export the model to
                    do_constant_folding=True,    # whether to execute constant folding for optimization
                    input_names=['input'],       # the model's input names
                    output_names=['output'],     # the model's output names
                    dynamic_axes={               # variable length axes
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                    }
                )
            except Exception as e:
                print(f"Failed to export ONNX model: {e}")
        
        self.last_path_ckpt = path_ckpt  
        
    def plot_pr_curves(self, recall, precision, is_wandb):
        plt.figure(figsize=(20, 10), dpi=80)
        plt.plot(recall, precision, label='Precision-Recall', color='black')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve')
        if is_wandb:
            wandb.log({f"precision_recall_curve_{self.dataset_type}": wandb.Image(plt)})
        plt.savefig(cst.DIR_SAVED_MODEL + "/" + str(self.model_type) + "/" +f"precision_recall_curve_{self.dataset_type}.svg")
        #plt.show()
        plt.close()
        
def compute_most_attended(att_feature):
    ''' att_feature: list of tensors of shape (num_samples, num_layers, 2, num_heads, num_features) '''
    att_feature = np.stack(att_feature)
    att_feature = att_feature.transpose(1, 3, 0, 2, 4)  # Use transpose instead of permute
    ''' att_feature: shape (num_layers, num_heads, num_samples, 2, num_features) '''
    indices = att_feature[:, :, :, 1]
    values = att_feature[:, :, :, 0]
    most_frequent_indices = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]), dtype=int)
    average_values = np.zeros((indices.shape[0], indices.shape[1], indices.shape[3]))
    for layer in range(indices.shape[0]):
        for head in range(indices.shape[1]):
            for seq in range(indices.shape[3]):
                # Extract the indices for the current layer and sequence element
                current_indices = indices[layer, head, :, seq]
                current_values = values[layer, head, :, seq]
                # Find the most frequent index
                most_frequent_index = mode(current_indices, keepdims=False)[0]
                # Store the result
                most_frequent_indices[layer, head, seq] = most_frequent_index
                # Compute the average value for the most frequent index
                avg_value = np.mean(current_values[current_indices == most_frequent_index])
                # Store the average value
                average_values[layer, head, seq] = avg_value
    return most_frequent_indices, average_values



