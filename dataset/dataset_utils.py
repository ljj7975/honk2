from enum import Enum

import random


class DatasetType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"

def shuffle_in_groups(a, b):
    assert len(a) == len(b)
    zipped = list(zip(a, b))
    random.shuffle(zipped)
    return zip(*zipped)
