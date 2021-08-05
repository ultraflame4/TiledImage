import fire
from PIL import Image
import TiledImage as TM




def commandLine_generate(refPath,saveDir,tilesDir,downsize=True,keepRatio=True,quads=4,scale=1):
    im:Image.Image=Image.open(refPath)
    if scale !=1:
        im=im.resize((round(im.width*scale),round(im.height*scale)))

    t = TM.TiledImageMaker(TM.loadImagesFromFolder(tilesDir),im)
    t.downsample=downsize
    t.keepRatio=keepRatio
    t.generate(quads,saveDir)

if __name__ == "__main__":

    fire.Fire(commandLine_generate)