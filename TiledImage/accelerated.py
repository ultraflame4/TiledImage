"""
This module contains stuff that tiles and generates the tiled image faster
"""
import glob
from typing import Literal

import cv2
import numba
import numpy as np
import numpy.typing
import tqdm
import numba as nb

from TiledImage import others


def loadTiles(path: str) -> np.ndarray:
    others.printLoadingTiles()
    return np.array([cv2.imread(f) for f in tqdm.tqdm(glob.iglob(path))])


@numba.jit(nopython=True, parallel=True)
def computeTileAverageValues(tiles: np.ndarray):
    print("Computing Average color for each tile")
    avgToTile: np.ndarray = np.zeros((tiles.shape[0], 3))
    for i in nb.prange(tiles.shape[0]):
        tile: np.ndarray = tiles[i]
        meanColor: np.ndarray = np.array([
            np.mean(tile[:, :, 0]),
            np.mean(tile[:, :, 1]),
            np.mean(tile[:, :, 2])
            ])

        avgToTile[i] = meanColor
    return avgToTile, tiles


@numba.jit(nopython=True, parallel=True,nogil=True)
def roundColorToNearestAvailable(color: np.ndarray, availableColors: np.ndarray) -> int:
    distances = np.zeros(availableColors.shape[0])
    for i in nb.prange(availableColors.shape[0]):
        distances[i] = np.linalg.norm(availableColors[i] - color)

    return np.where(distances == min(distances))[0]


# @numba.jit(nopython=True, parallel=True)
def CPU_compute(refImage: np.ndarray, outImage: np.ndarray, tileAvgVals: np.ndarray, tiles: np.ndarray,
                tileShape: tuple):
    for y, x in nb.pndindex(refImage.shape[:2]):
        refPixel = refImage[y, x]
        nx, ny = x * tileShape[1], y * tileShape[0]

        tileIndex = roundColorToNearestAvailable(refPixel, tileAvgVals)
        outImage[ny:ny + tileShape[0], nx:nx + tileShape[1]] = tiles[tileIndex][0]


def generate(refPath, savePath, tilesDir, compute: Literal["cpu", "gpu"] = "cpu", downscale=True):
    """

    :param refPath: Image reference path
    :param savePath: Path to sace output to.
    :param tilesDir: images to use for tiling
    :param compute: compute option. Whether to use gpu or cpu for computation
    :param downscale: Whether to downscale reference image so that output image is about the same size
    :return:
    """

    rawtiles = loadTiles(tilesDir)
    avgVals, tiles = computeTileAverageValues(rawtiles)
    t = rawtiles[0]
    refImage: np.ndarray = cv2.imread(refPath)
    if downscale:
        refImage = cv2.resize(refImage, (round(refImage.shape[1] / t.shape[1]),
                                         round(refImage.shape[0] / t.shape[0]))
                              )
    outImage = np.zeros((refImage.shape[0] * t.shape[0], refImage.shape[1] * t.shape[1], 3))
    others.printImageOutputDetails(savePath,outImage.shape[1],outImage.shape[0])
    if compute == "cpu":
        print("Beginning tiling...")
        CPU_compute(refImage, outImage, avgVals, tiles, t.shape)
        print("Finished.")

    elif compute == "gpu":
        print("Warning: Only CUDA GPUs are supported. The CUDA toolkit must also be installed."
              " Get CUDA toolkit from: https://developer.nvidia.com/cuda-toolkit")

        print("\n There currently isnt any gpu support. Please do not use this mode.")
    else:
        print("Error. Invalid compute option", compute)
        exit(-10)

    cv2.imwrite(savePath, outImage)
    print("Saved output to", savePath)
