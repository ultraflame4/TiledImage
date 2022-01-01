import dataclasses
import glob
from typing import Literal

import cv2
import numpy as np
from tqdm import tqdm

c_option = Literal["python","cpu","cuda"]
@dataclasses.dataclass()
class OptionFlags:
    REF_PATH: str
    TILES_DIR: str
    OUT_PATH: str
    TIMEIT: bool = False
    MAX_MEMORY: int = 2048
    COMPUTE_OPTION: c_option = "python"




@dataclasses.dataclass()
class TiledImageGenerate:
    flags: OptionFlags = None
    reference: np.ndarray = None
    tiles: np.ndarray = None  # Array of images
    canvas: np.ndarray = None # big, full image
    average_tile_vals: np.ndarray = None
    wc_size: tuple[int,int] = None # (max)size of working canvas. Actual size may differ due to differing reference slice sizes. (if slice is on edge, it may not be perfectly square)

    def set_wc(self,size:int):
        self.wc_size = size,size
