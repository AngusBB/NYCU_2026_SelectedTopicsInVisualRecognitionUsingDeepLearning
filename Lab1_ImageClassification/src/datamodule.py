import os
from src.transforms import *
from torch.utils.data import Dataset, DataLoader


class Lab1Dataset(Dataset):
  def __init__(self, obj_Args, str_Type, list_FileList, transform=None):
    self.obj_Args = obj_Args
    self.str_Type = str_Type
    list_DataFiles = getattr(self.obj_Args, f"{str_Type}_file", None)
    if list_DataFiles is None:
      raise AttributeError(f"Missing argument: {str_Type}_file")
    self.list_FileList = [
      (str_File, item_Data)
      for item_Data in list_FileList
      for str_File in list_DataFiles
    ]
    self.obj_Transform = transform

  def __len__(self):
    return len(self.list_FileList)

  def __getitem__(self, int_Index):
    if self.str_Type == 'test':
      str_File, str_Image = self.list_FileList[int_Index]
      int_Label = -1
    else:
      str_File, tuple_Data = self.list_FileList[int_Index]
      str_Image, str_LabelText = tuple_Data[0], tuple_Data[1][:3]
      int_Label = int(str_LabelText) - 1

    dict_Data = {
      'id': str_Image,
      'image': os.path.join(self.obj_Args.data_root, str_File, str_Image),
      'label': int_Label
    }

    if self.obj_Transform is not None:
      dict_Data = self.obj_Transform(dict_Data)

    return dict_Data


class DataModule():
  def __init__(self, obj_Args):
    self.obj_Args = obj_Args

  def read_file(self, str_FilePath):
    with open(str_FilePath, 'r') as obj_File:
      list_Content = obj_File.readlines()

    return list_Content

  def train_dataloader(self):
    if self.obj_Args.mode == 'training':
      list_FileLines = []
      for int_FoldId in range(1, self.obj_Args.num_folds + 1):
        if int_FoldId != self.obj_Args.valid_fold:
          list_FileLines += self.read_file(
            os.path.join(self.obj_Args.data_root, 'fold', f'fold{int_FoldId}.txt')
          )

      list_DataList = [tuple(str_Word.strip().split()) for str_Word in list_FileLines]

      obj_TrainTrans = Compose([
        LoadImg(list_Keys=["image"]),
        RandomTrans(list_Keys=["image"]),
        ResizeImg(list_Keys=["image"], tuple_Size=self.obj_Args.image_size),
        ImgToTensor(list_Keys=["image"]),
        RandomNoise(
          list_Keys=["image"],
          float_P=self.obj_Args.rand_noise_prob,
          float_Sigma=self.obj_Args.rand_noise_sigma
        ),
        ImgNormalize(list_Keys=["image"]),
        GridMask(
          list_Keys=["image"],
          int_Dmin=90,
          int_Dmax=300,
          float_Ratio=0.7,
          float_P=0.3
        )
      ])

      obj_TrainSet = Lab1Dataset(
        self.obj_Args,
        'train',
        list_DataList,
        transform=obj_TrainTrans
      )

      obj_TrainLoader = DataLoader(
        obj_TrainSet,
        batch_size=self.obj_Args.batch_size,
        num_workers=4,
        shuffle=True
      )
      print(f"train num: {len(obj_TrainSet)}.")
      return obj_TrainLoader

    return None

  def valid_dataloader(self):
    if self.obj_Args.mode == 'training':
      list_FileLines = self.read_file(
        os.path.join(self.obj_Args.data_root, 'fold', f'fold{self.obj_Args.valid_fold}.txt')
      )

      list_DataList = [tuple(str_Word.strip().split()) for str_Word in list_FileLines]

      obj_ValidTrans = Compose([
        LoadImg(list_Keys=["image"]),
        ResizeImg(list_Keys=["image"], tuple_Size=self.obj_Args.image_size),
        ImgToTensor(list_Keys=["image"]),
        ImgNormalize(list_Keys=["image"])
      ])

      obj_ValidSet = Lab1Dataset(
        self.obj_Args,
        'valid',
        list_DataList,
        transform=obj_ValidTrans
      )

      obj_ValidLoader = DataLoader(
        obj_ValidSet,
        batch_size=self.obj_Args.batch_size,
        num_workers=4,
        shuffle=False
      )
      print(f"valid num: {len(obj_ValidSet)}.")
      return obj_ValidLoader

    return None

  def test_dataloader(self):
    if self.obj_Args.mode == 'inference':
      list_FileLines = self.read_file(
        os.path.join(self.obj_Args.data_root, 'test_images.txt')
      )

      list_DataList = [str_Word.strip() for str_Word in list_FileLines]

      obj_TestTrans = Compose([
        LoadImg(list_Keys=["image"]),
        ResizeImg(list_Keys=["image"], tuple_Size=self.obj_Args.image_size),
        ImgToTensor(list_Keys=["image"]),
        ImgNormalize(list_Keys=["image"])
      ])

      obj_TestSet = Lab1Dataset(
        self.obj_Args,
        'test',
        list_DataList,
        transform=obj_TestTrans
      )

      obj_TestLoader = DataLoader(
        obj_TestSet,
        batch_size=self.obj_Args.batch_size,
        num_workers=4,
        shuffle=False
      )
      print(f"test num: {len(obj_TestSet)}.")
      return obj_TestLoader

    return None
