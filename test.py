import os
from pathlib import Path

from PIL import Image

import TiledImage as tm

tiles,tile_shape = tm.load_imageset(Path("./assets/tiles"))
atlas = tm.create_tiles_atlas(tiles,tile_shape)
os.makedirs("./build/",exist_ok=True)
# Image.fromarray(atlas).save("./build/atlas.png")

referenceImage = tm.load_image(Path("./assets/blackhole1.jpg"),resize=1/max(tile_shape),silent=False)
# Image.fromarray(referenceImage).save("./build/ref.png")
tm.tile_withreference(referenceImage,atlas,tile_shape)
