from enum import Enum


LABEL_SILENCE = "__silence__"
LABEL_UNKNOWN = "__unknown__"

class DatasetType(Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
