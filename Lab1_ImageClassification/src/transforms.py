import random
from PIL import Image
import torch
from torchvision.transforms import (
  Compose,
  Resize,
  ToTensor,
  RandomHorizontalFlip,
  RandomRotation,
  RandomAffine,
  Normalize
  )


class LoadImg():
  def __init__(self, list_Keys):
    self.list_Keys = list_Keys

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        dict_Data[str_Key] = Image.open(dict_Data[str_Key]).convert("RGB")
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data


class ResizeImg():
  def __init__(self, list_Keys, tuple_Size=(224, 224)):
    self.list_Keys = list_Keys
    self.tuple_Size = tuple_Size

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        dict_Data[str_Key] = Resize(size=self.tuple_Size)(dict_Data[str_Key])
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data


class RandomTrans():
  def __init__(self, list_Keys):
    self.list_Keys = list_Keys

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        dict_Data[str_Key] = RandomHorizontalFlip()(dict_Data[str_Key])
        dict_Data[str_Key] = RandomRotation(15)(dict_Data[str_Key])
        dict_Data[str_Key] = RandomAffine(
          0,
          shear=10,
          scale=(0.8, 1.2)
        )(dict_Data[str_Key])
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data


class ImgToTensor():
  def __init__(self, list_Keys):
    self.list_Keys = list_Keys

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        dict_Data[str_Key] = ToTensor()(dict_Data[str_Key])
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data


class ImgNormalize():
  def __init__(self, list_Keys):
    self.list_Keys = list_Keys

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        dict_Data[str_Key] = Normalize(
          (0.485, 0.456, 0.406),
          (0.229, 0.224, 0.225)
        )(dict_Data[str_Key])
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data


class RandomNoise():
  def __init__(self, list_Keys, float_P=0.1, float_Sigma=0.01):
    self.list_Keys = list_Keys
    self.float_Prob = float_P
    self.float_Sigma = float_Sigma

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        if random.random() <= self.float_Prob:
          dict_Data[str_Key] += self.float_Sigma * torch.randn(dict_Data[str_Key].shape)
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data


class GridMask():
  def __init__(self, list_Keys, float_P=0.25, int_Dmin=60, int_Dmax=160, float_Ratio=0.6):
    self.list_Keys = list_Keys
    self.float_Prob = float_P
    self.int_Dmin = int_Dmin
    self.int_Dmax = int_Dmax
    self.float_Ratio = float_Ratio

  def generate_grid_mask(self, tensor_Image):
    int_D = random.randint(self.int_Dmin, self.int_Dmax)
    int_Dx = random.randint(0, int_D - 1)
    int_Dy = random.randint(0, int_D - 1)
    int_Sl = int(int_D * (1 - self.float_Ratio))
    for int_RowStart in range(int_Dx, tensor_Image.shape[0], int_D):
      for int_ColStart in range(int_Dy, tensor_Image.shape[1], int_D):
        int_RowEnd = min(int_RowStart + int_Sl, tensor_Image.shape[0])
        int_ColEnd = min(int_ColStart + int_Sl, tensor_Image.shape[1])
        tensor_Image[:, int_RowStart:int_RowEnd, int_ColStart:int_ColEnd] = 0

    return tensor_Image

  def __call__(self, dict_Data):
    for str_Key in self.list_Keys:
      if str_Key in dict_Data:
        if random.random() <= self.float_Prob:
          dict_Data[str_Key] = self.generate_grid_mask(dict_Data[str_Key])
      else:
        raise KeyError(f"{str_Key} is not a key of data.")

    return dict_Data
