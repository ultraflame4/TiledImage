import os
from pathlib import Path
from PIL import Image
import TilImg as tm
import TilImg.utils

os.makedirs("./build/",exist_ok=True)
tiles,tile_shape = TilImg.utils.load_imageset(Path(), "./assets/tiles/*.png")

# atlas = tm.create_tiles_atlas(tiles,tile_shape)
# Image.fromarray(atlas).save("./build/atlas.png")

referenceImage = TilImg.utils.load_image(Path("./assets/blackhole1.jpg"), resize=1 / max(tile_shape), silent=False)
# referenceImage = tm.load_image(Path("./assets/blackhole1.jpg"),silent=False)


print(referenceImage.shape,tiles.shape)

# Image.fromarray(referenceImage).save("./build/ref.png")


# r = tiles[0].reshape((tile_shape[0]*tile_shape[1],tile_shape[2]))
# print(r.mean(axis=0))

# image = tm.generate_tiledimage(referenceImage, tiles, tile_shape)
# Image.fromarray(image).save("./build/out1.png")

image = tm.generate_tiledimage_gu(referenceImage, tiles, tile_shape)
Image.fromarray(image).save("./build/out2.png")
