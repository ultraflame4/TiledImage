
from distutils.core import setup

setup(name='TiledImage',
      version='1.0',
      description='Make image, from images',
      author='ultraflame42',
      author_email='ultraflame4@gmail.com',
      url='https://github.com/ultraflame4/TiledImage',
      packages=['TiledImage'],
      install_requires=[
          "tqdm>=4.62.0",
          "Pillow>=8.3.1"
          ]
     )