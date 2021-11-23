from typing import Literal

import fire
from PIL import Image
import TiledImage as TM
from TiledImage import accelerated


def commandLine_generate(refPath,saveDir,tilesDir,downsize=True,keepRatio=True,quads=4,scale=1,compute_mode:Literal["old","numba-cpu","numba-gpu"]="old"):
    print(f"Tiled Image:\n Using compute mode: {compute_mode}. \nDownsize Set to {downsize}")
    if not downsize:
        print("Warning: Not downsizing the image will cause the image to be very big!!")



    if compute_mode!="old":
        accelerated.generate(refPath, saveDir, tilesDir,downscale=downsize)
        return

    im:Image.Image=Image.open(refPath)
    if scale !=1:
        im=im.resize((round(im.width*scale),round(im.height*scale)))

    t = TM.TiledImageMaker(TM.loadImagesFromFolder(tilesDir),im)
    t.downsample=downsize
    t.keepRatio=keepRatio
    t.generate(quads,saveDir)

if __name__ == "__main__":

    fire.Fire(commandLine_generate)