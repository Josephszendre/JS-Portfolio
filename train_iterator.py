from glob import glob
import pickle
import torch
from torch.utils.data.dataset import Dataset

import numpy as np


class LarryKingDataset(Dataset):
    """Dataset wrapping images and target labels for Kaggle - Planet Amazon from Space competition.

    Arguments:
        A CSV file path
        Path to image folder
        Extension of images
        PIL transforms
    """

    def __init__(self, path, corpus):
        self.path = path
        glob_path = path + "/*.txt"
        self.files = glob(glob_path)
        self.corpus = corpus



    def __getitem__(self, index):
        rand_idx = np.random.randint(0, len(self.files))
        lines = list(open(self.files[rand_idx], "r"))[3:]
        indexed = list(map(lambda s: torch.LongTensor(
                        self.corpus.tokenize_string(s) + [self.corpus.eos]
                        ), lines ))

        return indexed, lines

    def __len__(self):
        return len(self.files)
