from pathlib import Path

import cv2

import TiledImage.video
import TiledImage

print(TiledImage.__version__)
tiles,tile_shape = TiledImage.utils.load_imageset(Path(), "./assets/tiles/*.png")
video = TiledImage.video.Video(Path("./assets/ref_vid.mp4"))
TiledImage.video.generate_tiledimage_video(video, tiles, tile_shape)