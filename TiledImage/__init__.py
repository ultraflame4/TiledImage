import math
import glob
import threading

from PIL import Image
import tqdm


def resizeImage(image: Image.Image, w, h, keepRatio=True):
    if keepRatio:
        ratio = image.width / image.height
        if w > h:
            return image.resize((int(ratio * h), h))
        else:
            return image.resize((w, int(ratio / w)))

    return image.resize((w, h))


def loadImagesFromFolder(path:str)->list[Image.Image]:
    """
    :param path: path of folder. make sure to add "/*"
    :return: lsit of pillow images
    """

    return [Image.open(f) for f in tqdm.tqdm(glob.iglob(path),desc="Loading image tiles...")]


class ImageTiles:
    def __init__(self,imageTiles:list[Image.Image]):
        self.imTiles = imageTiles
        self.tiles: dict[(int,int,int),Image] = {}

    def averageColor(self,tile:Image.Image):
        return tile.convert("RGB").resize((1,1)).getpixel((0,0))[:3]

    def prepTiles(self):
        self.tiles={self.averageColor(im):im for im in tqdm.tqdm(self.imTiles,desc="Averaging image colors...")}

    def getNearest(self,r,g,b):

        distances = {math.sqrt( (r - r2)**2 + (g - g2)**2 + (b - b2)**2 ):(r2,g2,b2) for r2,g2,b2 in self.tiles.keys()}
        rgb = distances[min(*distances.keys())]
        return self.tiles[rgb]

class CanvasQuad:
    def __init__(self, x, y, canvasQuadSize, canvas:Image.Image):
        self.x=x
        self.y=y
        self.canvasQuadSize=canvasQuadSize


        self.worldX = self.canvasQuadSize[0] * x
        self.worldY = self.canvasQuadSize[1] * y

        self.canvas = canvas

    def fill(self,color="#000000"):
        self.canvas.paste(Image.new("RGB", self.canvasQuadSize, color), (self.worldX, self.worldY))

    def setTile(self,im:Image.Image,x,y):
        self.canvas.paste(im,(self.worldX+x,self.worldY+y))

class ImageQuadrant:
    def __init__(self, x: int, y: int, imageTiles: ImageTiles, refQuad: Image.Image,quadCanvas:CanvasQuad,tileSize):
        self.x = x
        self.y = y
        self.refQuad = refQuad
        self.imageTiles = imageTiles
        self.quadCanvas = quadCanvas
        self.tileSize=tileSize

    def run(self,pbar:tqdm.tqdm):

        for x in range(self.refQuad.width):
            for y in range(self.refQuad.height):
                pix = self.refQuad.getpixel((x,y))
                tile = self.imageTiles.getNearest(pix[0],pix[1],pix[2])
                self.quadCanvas.setTile(tile,self.tileSize[0]*x,self.tileSize[1]*y)
                pbar.update(1)

class TiledImageMaker:
    def __init__(self, imageTiles: list[Image.Image], referenceImage: Image.Image):
        """
        :param imageTiles: Tiles to use for the tiled image. They have to be of the same width and height
        :param referenceImage: The image reference.
        """
        self.quads: list[ImageQuadrant] = []
        self.tiles = ImageTiles(imageTiles)
        self.refImage = referenceImage.copy()
        self.tile_w = imageTiles[0].width
        self.tile_h = imageTiles[0].height

        # Resize the referenceImage to be smaller, resulting in smaller image output. Improves performance
        self.downsample = True
        self.keepRatio = True

    def getCanvas(self):
        print((self.refImage.width * self.tile_w, self.refImage.height * self.tile_h))
        return Image.new("RGB", (self.refImage.width * self.tile_w, self.refImage.height * self.tile_h))

    def _prep_reference_image(self):
        if self.downsample:
            self.refImage = resizeImage(
                self.refImage,
                int(self.refImage.width / self.tile_w), int(self.refImage.height / self.tile_h),
                self.keepRatio)

    def generate(self, quadNo=2,save_dir="./out.png"):
        self._prep_reference_image()
        canvas = self.getCanvas()
        originalSize = canvas.size
        self.tiles.prepTiles()

        quadRefSize = (math.ceil(self.refImage.width / quadNo), math.ceil(self.refImage.height / quadNo))
        quadCanvasSize = (quadRefSize[0]*self.tile_w,quadRefSize[1]*self.tile_h)

        print(f"Quad Size | Canvas:{quadCanvasSize} Reference {quadRefSize}")

        for y in range(quadNo):
            for x in range(quadNo):
                xPos = x * quadRefSize[0]
                yPos = y * quadRefSize[1]
                quadIm = self.refImage.crop((xPos, yPos, quadRefSize[0]+xPos,quadRefSize[1]+yPos))
                quad = ImageQuadrant(x, y, self.tiles,
                                     quadIm,
                                     CanvasQuad(x,y,quadCanvasSize,canvas),(self.tile_w,self.tile_h))
                self.quads.append(quad)

        with tqdm.tqdm(total=self.refImage.width*self.refImage.height,desc=f"Progress [size:{self.refImage.size}]") as pbar:
            threads = [threading.Thread(target=i.run,args=(pbar,)) for i in self.quads]
            [i.start() for i in threads]
            [i.join() for i in threads]

        print(canvas.size)
        canvas.save(save_dir)
