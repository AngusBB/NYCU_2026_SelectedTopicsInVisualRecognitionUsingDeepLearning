import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
  obj_Parser = argparse.ArgumentParser(
    description=(
      "Create fold files."
    )
  )
  obj_Parser.add_argument(
    "--source-root",
    type=Path,
    default=Path("data"),
    help="Source root containing train/, val/, test/.",
  )
  obj_Parser.add_argument(
    "--output-root",
    type=Path,
    default=Path("data_preprocessed"),
    help="Output root for train/, test/, fold/, and metadata txt files.",
  )
  obj_Parser.add_argument(
    "--num-folds",
    type=int,
    default=5,
    help="Number of folds to generate.",
  )
  obj_Parser.add_argument(
    "--seed",
    type=int,
    default=42,
    help="Random seed used for fold splitting.",
  )
  obj_Parser.add_argument(
    "--transfer",
    choices=["copy", "hardlink", "symlink"],
    default="copy",
    help="How to place images into output folders.",
  )
  obj_Parser.add_argument(
    "--clear-output",
    action="store_true",
    help="Remove output directory before writing new content.",
  )
  obj_Parser.add_argument(
    "--train-subdirs",
    nargs="+",
    default=["train", "val"],
    help="Source subdirectories to be merged as training data.",
  )
  return obj_Parser.parse_args()


def is_image(path_Image: Path) -> bool:
  return path_Image.is_file() and path_Image.suffix.lower() in IMAGE_EXTENSIONS


def ensure_dirs(path_OutputRoot: Path) -> Dict[str, Path]:
  path_TrainDir = path_OutputRoot / "train"
  path_TestDir = path_OutputRoot / "test"
  path_FoldDir = path_OutputRoot / "fold"
  path_TrainDir.mkdir(parents=True, exist_ok=True)
  path_TestDir.mkdir(parents=True, exist_ok=True)
  path_FoldDir.mkdir(parents=True, exist_ok=True)
  return {
    "train": path_TrainDir,
    "test": path_TestDir,
    "fold": path_FoldDir,
  }


def transfer_file(path_Src: Path, path_Dst: Path, str_Mode: str) -> None:
  if path_Dst.exists():
    path_Dst.unlink()
  if str_Mode == "copy":
    shutil.copy2(path_Src, path_Dst)
  elif str_Mode == "hardlink":
    try:
      path_Dst.hardlink_to(path_Src)
    except OSError:
      shutil.copy2(path_Src, path_Dst)
  elif str_Mode == "symlink":
    path_Dst.symlink_to(path_Src.resolve())
  else:
    raise ValueError(f"Unsupported transfer mode: {str_Mode}")


def collect_labeled_samples(path_SourceRoot: Path, list_TrainSubdirs: List[str]) -> List[Tuple[Path, int, str]]:
  list_Samples: List[Tuple[Path, int, str]] = []
  for str_Subdir in list_TrainSubdirs:
    path_SplitDir = path_SourceRoot / str_Subdir
    if not path_SplitDir.exists():
      raise FileNotFoundError(f"Missing source directory: {path_SplitDir}")

    list_ClassDirs = sorted(path_Item for path_Item in path_SplitDir.iterdir() if path_Item.is_dir())
    for path_ClassDir in list_ClassDirs:
      try:
        int_ClassIndex = int(path_ClassDir.name)
      except ValueError as obj_Exc:
        raise ValueError(
          f"Class directory should be an integer but got: {path_ClassDir.name}"
        ) from obj_Exc

      int_LabelId = int_ClassIndex + 1
      str_LabelText = f"{int_LabelId:03d}"
      for path_ImagePath in sorted(path_ClassDir.iterdir()):
        if is_image(path_ImagePath):
          list_Samples.append((path_ImagePath, int_LabelId, str_LabelText))

  if not list_Samples:
    raise RuntimeError("No labeled training images found.")
  return list_Samples


def collect_test_samples(path_SourceRoot: Path) -> List[Path]:
  path_TestDir = path_SourceRoot / "test"
  if not path_TestDir.exists():
    raise FileNotFoundError(f"Missing source directory: {path_TestDir}")

  list_TestFiles = [path_Item for path_Item in sorted(path_TestDir.iterdir()) if is_image(path_Item)]
  if not list_TestFiles:
    raise RuntimeError("No test images found.")
  return list_TestFiles


def copy_training_data(
  list_Samples: List[Tuple[Path, int, str]],
  path_OutputTrainDir: Path,
  str_Transfer: str,
) -> List[Tuple[str, str]]:
  dict_UsedNames: Dict[str, int] = {}
  list_Rows: List[Tuple[str, str]] = []

  for path_Src, _int_LabelId, str_LabelText in list_Samples:
    str_BaseName = path_Src.name
    if str_BaseName in dict_UsedNames:
      dict_UsedNames[str_BaseName] += 1
      str_Stem = path_Src.stem
      str_Suffix = path_Src.suffix
      str_FinalName = f"{str_Stem}_{dict_UsedNames[str_BaseName]}{str_Suffix}"
    else:
      dict_UsedNames[str_BaseName] = 0
      str_FinalName = str_BaseName

    path_Dst = path_OutputTrainDir / str_FinalName
    transfer_file(path_Src, path_Dst, str_Mode=str_Transfer)
    list_Rows.append((str_FinalName, str_LabelText))

  list_Rows.sort(key=lambda tuple_Item: tuple_Item[0])
  return list_Rows


def copy_test_data(list_TestFiles: List[Path], path_OutputTestDir: Path, str_Transfer: str) -> List[str]:
  list_Names: List[str] = []
  dict_UsedNames: Dict[str, int] = {}

  for path_Src in list_TestFiles:
    str_BaseName = path_Src.name
    if str_BaseName in dict_UsedNames:
      dict_UsedNames[str_BaseName] += 1
      str_Stem = path_Src.stem
      str_Suffix = path_Src.suffix
      str_FinalName = f"{str_Stem}_{dict_UsedNames[str_BaseName]}{str_Suffix}"
    else:
      dict_UsedNames[str_BaseName] = 0
      str_FinalName = str_BaseName

    path_Dst = path_OutputTestDir / str_FinalName
    transfer_file(path_Src, path_Dst, str_Mode=str_Transfer)
    list_Names.append(str_FinalName)

  list_Names.sort()
  return list_Names


def write_lines(path_OutputFile: Path, list_Lines: List[str]) -> None:
  with path_OutputFile.open("w", encoding="utf-8") as obj_File:
    for str_Line in list_Lines:
      obj_File.write(f"{str_Line}\n")


def build_folds(
  list_TrainRows: List[Tuple[str, str]],
  int_NumFolds: int,
  int_Seed: int,
) -> Dict[int, List[Tuple[str, str]]]:
  if int_NumFolds < 2:
    raise ValueError("num_folds must be >= 2")

  dict_Grouped: Dict[str, List[str]] = {}
  for str_ImageName, str_LabelText in list_TrainRows:
    dict_Grouped.setdefault(str_LabelText, []).append(str_ImageName)

  obj_Rng = random.Random(int_Seed)
  dict_Folds: Dict[int, List[Tuple[str, str]]] = {
    int_FoldId: [] for int_FoldId in range(1, int_NumFolds + 1)
  }
  for str_LabelText, list_ImageNames in dict_Grouped.items():
    list_Shuffled = list_ImageNames[:]
    obj_Rng.shuffle(list_Shuffled)
    for int_Index, str_ImageName in enumerate(list_Shuffled):
      int_FoldId = (int_Index % int_NumFolds) + 1
      dict_Folds[int_FoldId].append((str_ImageName, str_LabelText))

  for int_FoldId in dict_Folds:
    dict_Folds[int_FoldId].sort(key=lambda tuple_Row: tuple_Row[0])
  return dict_Folds


def write_metadata(
  path_OutputRoot: Path,
  list_TrainRows: List[Tuple[str, str]],
  list_TestNames: List[str],
  dict_Folds: Dict[int, List[Tuple[str, str]]],
) -> None:
  _ = list_TrainRows
  write_lines(path_OutputRoot / "test_images.txt", list_TestNames)

  path_FoldDir = path_OutputRoot / "fold"
  for int_FoldId, list_FoldRows in dict_Folds.items():
    list_FoldLines = [
      f"{str_ImageName} {str_LabelText}" for str_ImageName, str_LabelText in list_FoldRows
    ]
    write_lines(path_FoldDir / f"fold{int_FoldId}.txt", list_FoldLines)


def main() -> None:
  obj_Args = parse_args()

  path_SourceRoot = obj_Args.source_root.resolve()
  path_OutputRoot = obj_Args.output_root.resolve()

  if obj_Args.clear_output and path_OutputRoot.exists():
    shutil.rmtree(path_OutputRoot)

  dict_Paths = ensure_dirs(path_OutputRoot)

  list_TrainSamples = collect_labeled_samples(path_SourceRoot, obj_Args.train_subdirs)
  list_TestSamples = collect_test_samples(path_SourceRoot)

  list_TrainRows = copy_training_data(list_TrainSamples, dict_Paths["train"], obj_Args.transfer)
  list_TestNames = copy_test_data(list_TestSamples, dict_Paths["test"], obj_Args.transfer)
  dict_Folds = build_folds(list_TrainRows, obj_Args.num_folds, obj_Args.seed)

  write_metadata(path_OutputRoot, list_TrainRows, list_TestNames, dict_Folds)

  print("Done.")
  print(f"Source: {path_SourceRoot}")
  print(f"Output: {path_OutputRoot}")
  print(f"Training images: {len(list_TrainRows)}")
  print(f"Testing images: {len(list_TestNames)}")
  print(f"Folds: {obj_Args.num_folds}")


if __name__ == "__main__":
  main()