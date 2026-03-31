import os
import time
import torch
import torch.nn as nn


class Trainer():
  def __init__(self, obj_Args):
    self.obj_Args = obj_Args

  def configure_device(self):
    int_GpuId = getattr(self.obj_Args, 'gpu', 0)

    if not torch.cuda.is_available():
      self.obj_Device = torch.device('cpu')
      print("Use cpu.")
      return None

    int_GpuCount = torch.cuda.device_count()
    if int_GpuId < 0 or int_GpuId >= int_GpuCount:
      print(
        f"Requested gpu {int_GpuId} is invalid. "
        f"Available range: 0 ~ {int_GpuCount - 1}. Use gpu 0 instead."
      )
      int_GpuId = 0

    self.obj_Device = torch.device(f'cuda:{int_GpuId}')
    print(f"Use gpu {int_GpuId}: {torch.cuda.get_device_name(int_GpuId)}")

    return None

  def configure_loss_func(self):
    self.obj_Criterion = nn.CrossEntropyLoss()

    return None

  def configure_optimizers(self):
    dict_Optimizers = {
      'Adam': torch.optim.Adam(
        self.obj_Model.parameters(),
        lr=self.obj_Args.lr,
        weight_decay=self.obj_Args.weight_decay
      ),
      'AdamW': torch.optim.AdamW(
        self.obj_Model.parameters(),
        lr=self.obj_Args.lr,
        weight_decay=self.obj_Args.weight_decay
      )
    }
    self.obj_Optimizer = dict_Optimizers[self.obj_Args.optimizer]

    dict_LRSchedulers = {
      'step': torch.optim.lr_scheduler.StepLR(
        optimizer=self.obj_Optimizer,
        step_size=self.obj_Args.lr_decay_period,
        gamma=self.obj_Args.lr_decay_factor
      ),
      'cos': torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=self.obj_Optimizer,
        T_max=4,
        eta_min=1e-6,
        last_epoch=-1
      )
    }
    self.obj_LRScheduler = dict_LRSchedulers[self.obj_Args.lr_scheduler]

    return None

  def compute_loss(self, tensor_X, tensor_Y):
    tensor_Loss = self.obj_Criterion(tensor_X, tensor_Y)

    return tensor_Loss

  def compute_acc(self, tensor_X, tensor_Y):
    tensor_Acc = torch.sum(torch.argmax(tensor_X, dim=1) == tensor_Y)

    return tensor_Acc

  def get_fold_ckpts(self, int_FoldId):
    str_SavePath = self.obj_Args.save_path
    os.makedirs(str_SavePath, exist_ok=True)

    str_Prefix = f"fold={int_FoldId}-"
    list_Ckpts = []
    for str_Name in os.listdir(str_SavePath):
      str_FullPath = os.path.join(str_SavePath, str_Name)
      if not os.path.isfile(str_FullPath):
        continue
      if not str_Name.startswith(str_Prefix) or '-acc=' not in str_Name:
        continue

      str_AccText = os.path.splitext(str_Name.split('-acc=')[-1])[0]
      try:
        float_ParsedAcc = float(str_AccText)
      except ValueError:
        continue

      list_Ckpts.append((float_ParsedAcc, str_Name))

    return list_Ckpts

  def save_ckpt(self, int_Ep, float_Acc):
    int_KeepTopK = 10
    int_FoldId = self.obj_Args.valid_fold
    float_AccValue = float(float_Acc)
    str_SaveName = f"fold={int_FoldId}-ep={int_Ep:0>3}-acc={float_AccValue:.10f}.pt"

    list_ExistingCkpts = self.get_fold_ckpts(int_FoldId)

    if len(list_ExistingCkpts) < int_KeepTopK:
      self.obj_Model.save(str_SaveName)
      print(f"[Checkpoint] Saved: {str_SaveName}")
      return None

    float_WorstAcc, str_WorstName = min(list_ExistingCkpts, key=lambda tuple_Item: tuple_Item[0])
    if float_AccValue > float_WorstAcc:
      self.obj_Model.save(str_SaveName)
      str_WorstPath = os.path.join(self.obj_Args.save_path, str_WorstName)
      if os.path.exists(str_WorstPath):
        os.remove(str_WorstPath)
      print(f"[Checkpoint] Saved: {str_SaveName}; removed: {str_WorstName}")
    else:
      print(
        f"[Checkpoint] Skip save: {str_SaveName} "
        f"(acc={float_AccValue:.10f} <= worst_top10={float_WorstAcc:.10f})"
      )

    return None

  def training_step(self):
    tensor_TotalLoss, tensor_TotalAcc = 0, 0
    int_TotalCount = 0
    self.obj_Model.train()
    for dict_BatchData in self.obj_TrainLoader:
      self.obj_Optimizer.zero_grad()

      tensor_Images = dict_BatchData['image'].to(self.obj_Device)
      tensor_Labels = dict_BatchData['label'].to(self.obj_Device)
      tensor_Preds = self.obj_Model(tensor_Images)
      tensor_Loss = self.compute_loss(tensor_Preds, tensor_Labels)
      tensor_Acc = self.compute_acc(tensor_Preds, tensor_Labels)

      tensor_Loss.backward()
      self.obj_Optimizer.step()
      tensor_TotalLoss += tensor_Loss * tensor_Images.shape[0]
      tensor_TotalAcc += tensor_Acc
      int_TotalCount += tensor_Images.shape[0]

      del tensor_Images, tensor_Labels, tensor_Preds

    self.obj_LRScheduler.step()

    return {
      'loss': tensor_TotalLoss / int_TotalCount,
      'acc': tensor_TotalAcc / int_TotalCount
    }

  def validation_step(self):
    tensor_TotalLoss, tensor_TotalAcc = 0, 0
    int_TotalCount = 0
    self.obj_Model.eval()
    with torch.no_grad():
      for dict_BatchData in self.obj_ValidLoader:
        tensor_Images = dict_BatchData['image'].to(self.obj_Device)
        tensor_Labels = dict_BatchData['label'].to(self.obj_Device)
        tensor_Preds = self.obj_Model(tensor_Images)
        tensor_Loss = self.compute_loss(tensor_Preds, tensor_Labels)
        tensor_Acc = self.compute_acc(tensor_Preds, tensor_Labels)
        tensor_TotalLoss += tensor_Loss * tensor_Images.shape[0]
        tensor_TotalAcc += tensor_Acc
        int_TotalCount += tensor_Images.shape[0]

        del tensor_Images, tensor_Labels, tensor_Preds

    return {
      'loss': tensor_TotalLoss / int_TotalCount,
      'acc': tensor_TotalAcc / int_TotalCount
    }

  def fit(self, obj_Model, obj_Dataset):
    self.configure_device()
    self.obj_Model = obj_Model.to(self.obj_Device)
    self.obj_TrainLoader = obj_Dataset.train_dataloader()
    self.obj_ValidLoader = obj_Dataset.valid_dataloader()
    self.configure_loss_func()
    self.configure_optimizers()

    float_Tic = time.time()
    for int_Ep in range(1, self.obj_Args.epochs + 1):
      dict_TrainRecord = self.training_step()
      print(
        f"epoch: {int_Ep}/{self.obj_Args.epochs},",
        f"time: {int(time.time() - float_Tic)},",
        f"type: train,",
        f"loss: {dict_TrainRecord['loss']:.4f},",
        f"acc: {dict_TrainRecord['acc']:.4f},",
        f"lr: {self.obj_LRScheduler.get_last_lr()[0]:.8f}"
      )

      dict_ValidRecord = self.validation_step()
      print(
        f"epoch: {int_Ep}/{self.obj_Args.epochs},",
        f"time: {int(time.time() - float_Tic)},",
        f"type: valid,",
        f"loss: {dict_ValidRecord['loss']:.4f},",
        f"acc: {dict_ValidRecord['acc']:.4f},",
        f"eta: {(self.obj_Args.epochs - int_Ep) * (time.time() - float_Tic) / int_Ep / 3600:.2f} hours"
      )

      if dict_ValidRecord['acc'] > self.obj_Args.baseline:
        self.save_ckpt(int_Ep=int_Ep, float_Acc=dict_ValidRecord['acc'])

    return None

  def fliplr(self, tensor_Inputs):
    int_BatchSize = tensor_Inputs.shape[0]
    tensor_Fliplr = torch.stack([
      torch.fliplr(torch.flip(tensor_Inputs[int_Index, :, :, :], dims=[1, 2]))
      for int_Index in range(int_BatchSize)
    ], dim=0)

    return tensor_Fliplr

  def ensemble(self, list_ModelList, obj_Dataset):
    self.configure_device()
    self.list_ModelList = [obj_Model.to(self.obj_Device).eval() for obj_Model in list_ModelList]
    self.obj_TestLoader = obj_Dataset.test_dataloader()

    with open('prediction.csv', 'w') as obj_File:
      obj_File.write('image_name,pred_label\n')

      with torch.no_grad():
        for dict_BatchData in self.obj_TestLoader:
          tensor_Images = dict_BatchData['image'].to(self.obj_Device)
          tensor_FliplrImages = self.fliplr(tensor_Images)

          tensor_Preds = torch.sum(
            torch.stack([
              obj_Model(tensor_Images) + obj_Model(tensor_FliplrImages)
              for obj_Model in self.list_ModelList
            ]),
            dim=0
          ).argmax(dim=1)

          for str_Id, tensor_Pred in zip(dict_BatchData['id'], tensor_Preds):
            str_ImageName = os.path.splitext(str_Id)[0]
            int_PredLabel = int(tensor_Pred.item())
            obj_File.write(f"{str_ImageName},{int_PredLabel}\n")
            print(f"{str_ImageName},{int_PredLabel}")

          del tensor_Images, tensor_Preds, tensor_FliplrImages

    return None
