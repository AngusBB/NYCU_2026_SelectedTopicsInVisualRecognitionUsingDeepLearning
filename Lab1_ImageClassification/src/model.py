import os
import torch
import torch.nn as nn
from torchvision.models import (
  resnet50,
  resnet101,
  resnext50_32x4d,
  resnext101_32x8d,
  resnext101_64x4d,
  ResNet50_Weights,
  ResNet101_Weights,
  ResNeXt50_32X4D_Weights,
  ResNeXt101_32X8D_Weights,
  ResNeXt101_64X4D_Weights
)


class ResNet(nn.Module):
  def __init__(self, obj_Args, str_Ckpt=False):
    super(ResNet, self).__init__()
    self.obj_Args = obj_Args
    self.obj_Model = self.build_model(obj_Args.model)
    self.str_Ckpt = str_Ckpt

    self.modify_final_layer()
    self.use_data_parallel()

  def modify_final_layer(self):
    self.obj_Model.fc = nn.Linear(
      in_features=self.obj_Model.fc.in_features,
      out_features=100,
      bias=True
    )

    if self.str_Ckpt is not False:
      self.load(self.str_Ckpt)

    return None

  def build_model(self, str_ModelName):
    dict_Builders = {
      'resnet50': (resnet50, ResNet50_Weights.IMAGENET1K_V2),
      'resnet101': (resnet101, ResNet101_Weights.IMAGENET1K_V2),
      'resnext50': (resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V2),
      'resnext101': (resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V2),
      'resnext101_64x4d': (resnext101_64x4d, ResNeXt101_64X4D_Weights.IMAGENET1K_V1)
    }

    if str_ModelName not in dict_Builders:
      raise ValueError(f"Unknown model: {str_ModelName}")

    fn_Builder, obj_Weights = dict_Builders[str_ModelName]
    return fn_Builder(weights=obj_Weights)

  def use_data_parallel(self):
    int_GpuId = getattr(self.obj_Args, 'gpu', 0)

    if self.obj_Args.mode == 'training' and torch.cuda.is_available():
      self.obj_Model = nn.DataParallel(
        self.obj_Model,
        device_ids=[int_GpuId],
        output_device=int_GpuId
      )

    return None

  def load(self, str_Ckpt):
    print(f"load {str_Ckpt}.")
    dict_Ckpt = torch.load(
      os.path.join(self.obj_Args.save_path, str_Ckpt),
      map_location=torch.device('cpu')
    )
    self.obj_Model.load_state_dict(dict_Ckpt)

    return None

  def save(self, str_SaveName):
    os.makedirs(self.obj_Args.save_path, exist_ok=True)
    obj_ModelToSave = self.obj_Model.module if hasattr(self.obj_Model, 'module') else self.obj_Model
    torch.save(obj_ModelToSave.state_dict(), os.path.join(self.obj_Args.save_path, str_SaveName))

    return None

  def forward(self, tensor_Inputs):
    tensor_Outputs = self.obj_Model(tensor_Inputs)

    return tensor_Outputs
