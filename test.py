import os
from pathlib import Path

import numpy as np
from PIL import Image
import cv2

import TiledImage as tm

os.makedirs("./build/",exist_ok=True)
tiles,tile_shape = tm.load_imageset(Path("./assets/tiles"))

# atlas = tm.create_tiles_atlas(tiles,tile_shape)
# Image.fromarray(atlas).save("./build/atlas.png")

referenceImage = tm.load_image(Path("./assets/blackhole1.jpg"),resize=1/max(tile_shape),silent=False)
referenceImage = tm.load_image(Path("./assets/blackhole1.jpg"),silent=False)


print(referenceImage.shape,tiles.shape)

# Image.fromarray(referenceImage).save("./build/ref.png")


# r = tiles[0].reshape((tile_shape[0]*tile_shape[1],tile_shape[2]))
# print(r.mean(axis=0))

# image = tm.generate_tiledimage(referenceImage, tiles, tile_shape)
# Image.fromarray(image).save("./build/out1.png")

image = tm.generate_tiledimage_gu(referenceImage, tiles, tile_shape,useCuda=True)
Image.fromarray(image).save("./build/out2.png")
