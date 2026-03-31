import random
import numpy as np
import torch
import argparse
from src.datamodule import DataModule
from src.model import ResNet
from src.trainer import Trainer


def set_seed(int_Seed: int):
  random.seed(int_Seed)
  np.random.seed(int_Seed)
  torch.manual_seed(int_Seed)
  torch.cuda.manual_seed_all(int_Seed)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False


def parse_args():
  obj_Parser = argparse.ArgumentParser()
  obj_Parser.add_argument('--seed', type=int, default=42)
  obj_Parser.add_argument('--mode', type=str, default='training')
  obj_Parser.add_argument('--model', type=str, default='resnext101_64x4d')

  obj_Parser.add_argument('--epochs', type=int, default=100)
  obj_Parser.add_argument('--batch_size', type=int, default=48)
  obj_Parser.add_argument('--optimizer', type=str, default='AdamW')
  obj_Parser.add_argument('--lr', type=float, default=1e-4)
  obj_Parser.add_argument('--lr_scheduler', type=str, default='step')
  obj_Parser.add_argument('--lr_decay_period', type=int, default=3)
  obj_Parser.add_argument('--lr_decay_factor', type=float, default=0.8)
  obj_Parser.add_argument('--weight_decay', type=float, default=1e-4)

  obj_Parser.add_argument('--image_size', type=int, nargs=2, default=[600, 600], metavar=('H', 'W'))
  obj_Parser.add_argument('--rand_noise_sigma', type=float, default=0.02)
  obj_Parser.add_argument('--rand_noise_prob', type=float, default=0.1)
  obj_Parser.add_argument('--baseline', type=float, default=0.8)

  obj_Parser.add_argument('--num_folds', type=int, default=5)

  obj_Parser.add_argument('--train_file', nargs='+', default=['train'])
  obj_Parser.add_argument('--valid_file', nargs='+', default=['train'])
  obj_Parser.add_argument('--data_root', type=str, default='data_preprocessed')
  obj_Parser.add_argument('--save_path', type=str, default='ckpt_resnext101_64_is600_bs45')

  obj_Parser.add_argument('--gpu', type=int, default=0)

  obj_Args = obj_Parser.parse_args()
  obj_Args.image_size = tuple(obj_Args.image_size)
  return obj_Args


if __name__ == '__main__':
  obj_Args = parse_args()
  set_seed(obj_Args.seed)

  for int_FoldId in range(1, obj_Args.num_folds + 1):
    print(f"Fold {int_FoldId}:")
    obj_Args.valid_fold = int_FoldId

    obj_Dataset = DataModule(obj_Args)
    obj_Model = ResNet(obj_Args)
    obj_Trainer = Trainer(obj_Args)
    obj_Trainer.fit(obj_Model, obj_Dataset)

    del obj_Dataset, obj_Model, obj_Trainer
