from pathlib import Path

import cv2

import TilImg.video
import TilImg

print(TilImg.__version__)
tiles,tile_shape = TilImg.utils.load_imageset(Path(), "./assets/tiles/*.png")
video = TilImg.video.Video(Path("./assets/ref_vid.mp4"))
TilImg.video.generate_tiledimage_video(video, tiles, tile_shape, useCuda=True)