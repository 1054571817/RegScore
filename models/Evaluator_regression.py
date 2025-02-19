from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl
import wandb


from models.TabAttention import TabAttention
from models.TabMixer import TabMixer
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT
from models.MultimodalModelMUL import MultimodalModelMUL
from models.MultimodalModelTransformer import MultimodalModelTransformer
from models.CHARMS import CHARMS
from models.Tip_utils.Tip_downstream_regscore import TIPBackboneRegScore


class Evaluator_Regression(pl.LightningModule):
  def __init__(self, hparams, logdir):
    super().__init__()
    self.save_hyperparameters(hparams)
    self.threshold = self.hparams.threshold
    self.alpha = self.hparams.alpha
    self.algorithm_name = self.hparams.algorithm_name
    if self.algorithm_name == "RegScore":
      self.scoring_strategy = self.hparams.scoring_strategy

    if self.hparams.eval_datatype == 'imaging':
      self.model = ImagingModel(self.hparams)
    elif self.hparams.eval_datatype == 'multimodal':
      if self.hparams.strategy == 'tip':
        if self.hparams.algorithm_name == 'RegScore':
          self.scoring_strategy = self.hparams.scoring_strategy
          print("________ Training RegScore model! _________")
          self.model = TIPBackboneRegScore(self.hparams)
        elif self.hparams.finetune_ensemble == True:
          self.model = TIPBackboneEnsemble(self.hparams)
        else:
          self.model = TIPBackbone(self.hparams)
      elif self.hparams.strategy == 'charms':
          self.model = CHARMS(self.hparams, logdir=logdir)
      else:
        raise ValueError("Should be tip or charms!")
    elif self.hparams.eval_datatype == 'tabular':
        self.model = TabularModel(self.hparams)
    elif self.hparams.eval_datatype == 'imaging_and_tabular':
      if self.hparams.algorithm_name == 'DAFT':
        self.model = DAFT(self.hparams)
      elif self.hparams.algorithm_name == 'TabMixer':
        self.model = TabMixer(self.hparams)
      elif self.hparams.algorithm_name == 'TabAttention':
        self.model = TabAttention(self.hparams)
      elif self.hparams.algorithm_name in set(['CONCAT','MAX']):
        if self.hparams.strategy == 'tip':
          # use TIP's tabular encoder
          self.model = MultimodalModelTransformer(self.hparams)
        else:
          # use MLP-based tabular encoder
          self.model = MultimodalModel(self.hparams)
      elif self.hparams.algorithm_name == 'MUL':
        self.model = MultimodalModelMUL(self.hparams)


    self.criterion = torch.nn.MSELoss()

    self.mae_train = torchmetrics.MeanAbsoluteError()
    self.mae_val = torchmetrics.MeanAbsoluteError()
    self.mae_test = torchmetrics.MeanAbsoluteError()

    self.pcc_train = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
    self.pcc_val = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)
    self.pcc_test = torchmetrics.PearsonCorrCoef(num_outputs=hparams.num_classes)

    task = 'binary'

    self.acc_train = torchmetrics.Accuracy(task=task, num_classes=2)
    self.acc_val = torchmetrics.Accuracy(task=task, num_classes=2)
    self.acc_test = torchmetrics.Accuracy(task=task, num_classes=2)

    self.precision_train = torchmetrics.Precision(task=task, num_classes=2)
    self.precision_val = torchmetrics.Precision(task=task, num_classes=2)
    self.precision_test = torchmetrics.Precision(task=task, num_classes=2)

    self.recall_train = torchmetrics.Recall(task=task, num_classes=2)
    self.recall_val = torchmetrics.Recall(task=task, num_classes=2)
    self.recall_test = torchmetrics.Recall(task=task, num_classes=2)

    self.f1_train = torchmetrics.F1Score(task=task, num_classes=2)
    self.f1_val = torchmetrics.F1Score(task=task, num_classes=2)
    self.f1_test = torchmetrics.F1Score(task=task, num_classes=2)

    self.best_val_score = 999999

    print(self.model)

  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    """
    Runs test step
    """
    x, y = batch
    y_hat = self.forward(x)
    y_hat = y_hat.detach()

    self.mae_test(y_hat, y)
    self.pcc_test(y_hat, y)

    y_hat[y_hat <= self.threshold] = 0
    y_hat[y_hat > self.threshold] = 1
    y[y <= self.threshold] = 0
    y[y > self.threshold] = 1

    self.acc_test(y_hat, y)
    self.precision_test(y_hat, y)
    self.recall_test(y_hat, y)
    self.f1_test(y_hat, y)


  def test_epoch_end(self, _) -> None:
    """
    Test epoch end
    """
    test_mae = self.mae_test.compute()
    test_pcc = self.pcc_test.compute()
    test_pcc_mean = torch.mean(test_pcc)

    self.log('test.mae', test_mae)
    self.log('test.pcc.mean', test_pcc_mean, metric_attribute=self.pcc_test)

    acc = self.acc_test.compute()
    precision = self.precision_test.compute()
    recall = self.recall_test.compute()
    f1 = self.f1_test.compute()

    self.log('test.acc', acc)
    self.log('test.precision', precision)
    self.log('test.recall', recall)
    self.log('test.f1', f1)

    self.acc_test.reset()
    self.precision_test.reset()
    self.recall_test.reset()
    self.f1_test.reset()

    if self.hparams.algorithm_name == 'RegScore':
      if self.scoring_strategy == "generalized_regscore" or self.scoring_strategy=="personalized_regscore":
        regscore = self.model.test_epoch_end()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Generates a prediction from a data point
    """
    y_hat =  self.model(x).squeeze()

    return y_hat

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Train and log.
    """
    x, y = batch

    if self.algorithm_name=="RegScore" and self.scoring_strategy=="generalized_regscore":
      x_bin = x[3] if not self.hparams.missing_tabular else x[4]
      batch_size = x_bin.shape[0]
      bias_column = torch.ones((batch_size, 1), device=x_bin.device)
      x_t_b = torch.cat([bias_column, x_bin], dim=1)
      y_hat, reg_params = self.model(x)
      y_hat = y_hat.squeeze()
      # pairwise_predictions = x_t_b @ reg_params.T
      # mask = ~torch.eye(batch_size, dtype=torch.bool)
      # y_hat_others = pairwise_predictions[mask].view(batch_size, -1)  # (batch_size, batch_size-1)
      # y_others = y.expand(batch_size, batch_size)[mask].view(batch_size, -1)
      # loss_general = self.criterion(y_hat_others, y_others)
      loss_personal = self.criterion(y_hat, y)
      loss = (1 - self.alpha) * loss_personal #+ self.alpha * loss_general
    else:
      y_hat = self.forward(x)
      loss = self.criterion(y_hat, y)

    y_hat = y_hat.detach()

    self.mae_train(y_hat, y)
    self.pcc_train(y_hat, y)

    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
    self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False)
    y_hat_d = y_hat.detach().clone()
    y_d = y.detach().clone()
    y_hat_d[y_hat_d<= self.threshold]= 0
    y_hat_d[y_hat_d> self.threshold]= 1
    y_d[y_d<= self.threshold]= 0
    y_d[y_d> self.threshold]= 1

    self.acc_train(y_hat_d, y_d)
    self.precision_train(y_hat_d, y_d)
    self.recall_train(y_hat_d, y_d)
    self.f1_train(y_hat_d, y_d)

    return loss

  def training_epoch_end(self, _) -> None:
    epoch_pcc_train = self.pcc_train.compute()
    epoch_pcc_train_mean = epoch_pcc_train.mean()
    self.log('eval.train.pcc.mean', epoch_pcc_train_mean, on_epoch=True, on_step=False, metric_attribute=self.pcc_train)
    self.pcc_train.reset()

    acc = self.acc_train.compute()
    precision = self.precision_train.compute()
    recall = self.recall_train.compute()
    f1 = self.f1_train.compute()

    self.log('eval.train.acc', acc)
    self.log('eval.train.precision', precision)
    self.log('eval.train.recall', recall)
    self.log('eval.train.f1', f1)

    self.acc_train.reset()
    self.precision_train.reset()
    self.recall_train.reset()
    self.f1_train.reset()

    if self.hparams.algorithm_name == 'RegScore':
      if self.scoring_strategy == "generalized_linreg":
        self.model.lin_reg_params = torch.cat(self.model.epoch_lin_reg_params, dim=0).mean(dim=0, keepdim=True)
        self.model.epoch_lin_reg_params = []
      elif self.scoring_strategy == "generalized_regscore":
        self.model.training_epoch_end()


  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    """
    Validate and log
    """
    x, y = batch

    y_hat = self.forward(x)
    loss = self.criterion(y_hat, y)
    y_hat = y_hat.detach()

    self.mae_val(y_hat, y)
    self.pcc_val(y_hat, y)

    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

    y_hat[y_hat <= self.threshold] = 0
    y_hat[y_hat > self.threshold] = 1
    y[y <= self.threshold] = 0
    y[y > self.threshold] = 1

    self.acc_val(y_hat, y)
    self.precision_val(y_hat, y)
    self.recall_val(y_hat, y)
    self.f1_val(y_hat, y)


  def validation_epoch_end(self, _) -> None:
    """
    Compute validation epoch metrics and check for new best values
    """
    if self.trainer.sanity_checking:
      return

    epoch_mae_val = self.mae_val.compute()
    epoch_pcc_val = self.pcc_val.compute()
    epoch_pcc_val_mean = torch.mean(epoch_pcc_val)

    self.log('eval.val.mae', epoch_mae_val, on_epoch=True, on_step=False, metric_attribute=self.mae_val)
    self.log('eval.val.pcc.mean', epoch_pcc_val_mean, on_epoch=True, on_step=False, metric_attribute=self.pcc_val)

    self.best_val_score = min(self.best_val_score, epoch_mae_val)

    self.mae_val.reset()
    self.pcc_val.reset()

    acc = self.acc_val.compute()
    precision = self.precision_val.compute()
    recall = self.recall_val.compute()
    f1 = self.f1_val.compute()

    self.log('eval.val.acc', acc)
    self.log('eval.val.precision', precision)
    self.log('eval.val.recall', recall)
    self.log('eval.val.f1', f1)

    self.acc_val.reset()
    self.precision_val.reset()
    self.recall_val.reset()
    self.f1_val.reset()


  def configure_optimizers(self):
    """
    Sets optimizer and scheduler.
    Must use strict equal to false because if check_val_n_epochs is > 1
    because val metrics not defined when scheduler is queried
    """
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    return optimizer

    return (
      {
        "optimizer": optimizer,
        "lr_scheduler": {
          "scheduler": scheduler,
          "monitor": 'eval.val.loss',
          "strict": False
        }
      }
    )