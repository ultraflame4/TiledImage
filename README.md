# TiledImage V3

This program composites many images into one big image using a set of images and a reference image

### Limitations
- All images in the set must be of the same resolution and dimensions
- The final product may have black bars on the side or have clipped textures because<br/>
  the dimensions of the reference image do not match dimensions of the images in the image set.<br/>
  The reference image dimensions has to be a multiple of the image set's dimensions to avoid this.<br/>
  