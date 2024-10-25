''' utils.__init__.py
>>> 💥 This is a package for utility functions.
'''

#--- Define the __version__ variable ---#
__version__ = "3.0.0"

#--- Import the necessary modules ---#
import os
from .logger import Logger, color
from .parser import ParseKwargs

#--- Define global variables ---#
FISHSCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'fishscale_data.yaml')

#--- Define the __all__ variable to include when the module is imported ---#
__all__ = ['Logger', 'color', 'ParseKwargs', 'FISHSCALE_DIR']