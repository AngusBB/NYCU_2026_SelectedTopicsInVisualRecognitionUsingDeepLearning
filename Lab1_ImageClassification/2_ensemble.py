import os
import re
import argparse
from collections import defaultdict
from src.datamodule import DataModule
from src.model import ResNet
from src.trainer import Trainer


def parse_args():
  obj_Parser = argparse.ArgumentParser()
  obj_Parser.add_argument('--mode', type=str, default='inference')
  obj_Parser.add_argument('--model', type=str, default='resnext101_64x4d')
  obj_Parser.add_argument('--batch_size', type=int, default=24)
  obj_Parser.add_argument('--image_size', type=int, nargs=2, default=[400, 400], metavar=('H', 'W'))

  obj_Parser.add_argument('--test_file', nargs='+', default=['test'])

  obj_Parser.add_argument('--file_root', type=str, default='data')
  obj_Parser.add_argument('--data_root', type=str, default='data_preprocessed')
  obj_Parser.add_argument('--save_path', type=str, default='ckpt_resnext101_64_is600_bs45')

  obj_Parser.add_argument('--top_k', type=int, default=3)
  obj_Parser.add_argument('--gpu', type=int, default=0)

  obj_Args = obj_Parser.parse_args()
  obj_Args.image_size = tuple(obj_Args.image_size)
  return obj_Args


def find_top_k_ckpt_per_fold(str_SavePath, int_TopK=3):
  if not os.path.isdir(str_SavePath):
    raise FileNotFoundError(f"Checkpoint directory not found: {str_SavePath}")

  obj_Pattern = re.compile(r'^fold=(\d+)-ep=(\d+)-acc=([0-9]*\.?[0-9]+)(?:\.pt)?$')
  dict_FoldCkpts = defaultdict(list)

  for str_Name in os.listdir(str_SavePath):
    obj_Match = obj_Pattern.match(str_Name)
    if obj_Match is None:
      continue

  int_FoldId = int(obj_Match.group(1))
  int_Epoch = int(obj_Match.group(2))
  float_Acc = float(obj_Match.group(3))
  dict_FoldCkpts[int_FoldId].append((float_Acc, int_Epoch, str_Name))

  if len(dict_FoldCkpts) == 0:
    raise ValueError(f"No valid checkpoints found in {str_SavePath}")

  list_Selected = []
  for int_FoldId in sorted(dict_FoldCkpts.keys()):
    list_Ranked = sorted(
      dict_FoldCkpts[int_FoldId],
      key=lambda tuple_Item: (tuple_Item[0], tuple_Item[1]),
      reverse=True
    )
    list_TopItems = list_Ranked[:int_TopK]
    print(f"Fold {int_FoldId} top-{len(list_TopItems)} checkpoints:")
    for float_Acc, int_Epoch, str_Name in list_TopItems:
      print(f"  ep={int_Epoch:03d}, acc={float_Acc:.4f}, ckpt={str_Name}")
      list_Selected.append(str_Name)

  return list_Selected


if __name__ == '__main__':
  obj_Args = parse_args()
  obj_Dataset = DataModule(obj_Args)
  list_SelectedCkpts = find_top_k_ckpt_per_fold(obj_Args.save_path, int_TopK=obj_Args.top_k)
  list_ModelList = [ResNet(obj_Args, str_Ckpt=str_CkptName) for str_CkptName in list_SelectedCkpts]
  obj_Trainer = Trainer(obj_Args)
  obj_Trainer.ensemble(list_ModelList, obj_Dataset)
