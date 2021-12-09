import copy
import itertools
import numpy as np
import cv2
from tabulate import tabulate
from termcolor import colored
import logging
class_names=[]
def print_class_histogram(roidbs):
  """
    Args:
        roidbs (list[dict]): the same format as the output of `training_roidbs`.
  """
  #class_names = DatasetRegistry.get_metadata(cfg.DATA.TRAIN[0], "class_names")
  # labels are in [1, NUM_CATEGORY], hence +2 for bins
  hist_bins = np.arange(len(class_names) + 2)

  # Histogram of ground-truth objects
  gt_hist = np.zeros((len(class_names) + 1,), dtype=np.int)
  for entry in roidbs:
    # filter crowd?
    gt_inds = np.where((entry["class"] > 0) & (entry["is_crowd"] == 0))[0]
    gt_classes = entry["class"][gt_inds]
    if len(gt_classes):
      assert gt_classes.max() <= len(class_names) - 1, gt_classes.max()
    gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
  data = list(
      itertools.chain(*[[class_names[i + 1], v]
                        for i, v in enumerate(gt_hist[1:])]))
  COL = min(6, len(data))
  total_instances = sum(data[1::2])
  data.extend([None] * ((COL - len(data) % COL) % COL))
  data.extend(["total", total_instances])
  data = itertools.zip_longest(*[data[i::COL] for i in range(COL)])
  # the first line is BG
  table = tabulate(
      data,
      headers=["class", "#box"] * (COL // 2),
      tablefmt="pipe",
      stralign="center",
      numalign="left")
  logging.info("Ground-Truth category distribution:\n" + colored(table, "cyan"))