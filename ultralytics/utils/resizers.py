''' 💥 src.utils.resizers.py 
>>> This script is used to define utility functions to analyze frames using the image detector and the VLM model.
'''

#------------------------------------------------------------------------------------------------------------#
# 1. ⚙️ IMPORTS LIBRARIES AND MODULES
#------------------------------------------------------------------------------------------------------------#
import numpy as np

from dataclasses import dataclass
from functools import partial
from enum import Enum, auto
from typing import Callable, Optional

from ultralytics.engine.results import Boxes
from ultralytics.utils import LOGGER
from ultralytics.utils.frame_utils import adjust_box_to_image_dimensions, pad_up_to_square, square_up_box

#------------------------------------------------------------------------------------------------------------#
# 2. 🔢 ENUMS
#------------------------------------------------------------------------------------------------------------#
class ResizerType(Enum):
    ''' Enumeration to represent the type of resizer. '''
    IDENTITY = auto()
    EXACT = auto()
    AROUND = auto()
    SQUARE_AROUND_BOX = auto()
    PAD_TO_SQUARE = auto()
    
    @classmethod
    def _from_str(cls, s : str) -> 'ResizerType':
        ''' Factory method to create a ResizerType from a string.
        
        Parameters
        ----------
        s : str
            The string representation of the ResizerType.
            
        Returns
        -------
        ResizerType
            The ResizerType.
        '''
        try:
            return cls[s.upper()]
        except KeyError:
            LOGGER.error(f"Invalid resizer type: {s}")

#------------------------------------------------------------------------------------------------------------#
# 3. 💫 DATACLASSES
#------------------------------------------------------------------------------------------------------------#
@dataclass
class ResizerConfig:
    ''' Dataclass to represent the configuration of a resizer. '''
    type: ResizerType
    scale_factor: Optional[float] = None
    min_proportion: Optional[float] = None
    
    def __post_init__(self):
        ''' Post initialization method to validate the configuration. '''
        #--- Check the configuration ---#
        if self.scale_factor : assert self.scale_factor > 0, "The scale factor must be greater than 0."
        if self.min_proportion : assert 0 <= self.min_proportion <= 1, "The minimum proportion must be between 0 and 1."
        
        #--- Convert the resizer type to an enumeration ---#        
        self.type = ResizerType._from_str(self.type)

#------------------------------------------------------------------------------------------------------------#
# 4. 🚀 FUNCTIONS
#------------------------------------------------------------------------------------------------------------#

#--- Identity Resizer ---#
def identity(frame: np.ndarray, _box: Boxes) -> np.ndarray:
    ''' Resizer that returns the frame as is.
    
    Parameters
    ----------
    frame : np.ndarray
        The frame to resize.
    box : Boxes
        The box to resize the frame to. (Not used)
        
    Returns
    -------
    np.ndarray
        The resized frame.
    '''
    #--- Return the frame as is ---#
    return frame


#--- Exact Resizer ---#
def exact(frame: np.ndarray, box: Boxes) -> np.ndarray:
    ''' Resizer that returns a rectangular subframe exactly corresponding to the *box*.
    
    Parameters
    ----------
    frame : np.ndarray
        The frame to resize.
    box : Boxes
        The box to resize the frame to.
        
    Returns
    -------
    np.ndarray
        The resized frame.
    '''
    #--- Extract the bounding box coordinates ---#
    x1, y1, x2, y2 = x1y1x2y2(box)
    
    #--- Return the cropped image ---#
    return frame[y1:y2, x1:x2]

#--- Around Resizer ---#
def around(frame: np.ndarray, box: Boxes, scale_factor: float, min_proportion: float) -> np.ndarray:
    ''' Resizer that returns a rectangular subframe centered on the *box* whose dimensions are adjusted to the image dimensions.
    The rectangular region corresponds to an area centered on the *box* but scaled up/down by *scale_factor*. 
    This region is increased if it would be smaller than the *frame*'s dimensions times the *min_proportion* 
    (set `min_proportion=0` if you do not want this behaviour).
    
    Parameters
    ----------
    frame : np.ndarray
        The frame to resize.
    box : Boxes
        The box to resize the frame to.
    scale_factor : int | float
        The factor by which to multiply the box dimensions.
    min_proportion : int | float
        The minimum proportion of the size of the image that the box will be adjusted to.   
    '''
    #--- Adjust the box to the image dimensions ---#
    x1, y1, x2, y2 = adjust_box_to_image_dimensions(frame, x1y1x2y2(box), scale_factor, min_proportion)

    #--- Return the cropped image ---#
    return frame[y1:y2, x1:x2]

#--- Square Around Resizer ---#
def square_around_box(frame: np.ndarray, box: Boxes, scale_factor: float) -> np.ndarray:
    '''Resizer that returns a square subframe centered on the *box* whose side length is ``min(length,width)*scale_factor``.
    
    Parameters
    ----------
    frame : np.ndarray
        The frame to resize.
    box : Boxes
        The box to resize the frame to.
    scale_factor : int | float
        The factor by which to multiply the box dimensions.

    Returns
    -------
    np.ndarray
        The resized frame.
    '''
    #--- Square up the box ---#
    box = square_up_box(frame, x1y1x2y2(box))
    
    #--- Adjust the box to the image dimensions ---#
    x1, y1, x2, y2 = adjust_box_to_image_dimensions(frame, box, scale_factor, 0)

    #--- Return the cropped image ---#
    return frame[y1:y2, x1:x2]

#--- Pad Frame to Square Resizer ---#
def pad_to_square(frame: np.ndarray, _box: Boxes) -> np.ndarray:
    ''' Resizer that pads the frame with grey to make it square.
    The shade of grey is (114, 114, 114) in RGB.
    
    Parameters
    ----------
    frame : np.ndarray
        The frame to resize.
    _box : Boxes
        The box to resize the frame to. (Not used)
        
    Returns
    -------
    np.ndarray
    '''
    #--- Return the padded frame ---#
    return pad_up_to_square(frame)

#------------------------------------------------------------------------------------------------------------#
# 5. 🏭 FACTORY
#------------------------------------------------------------------------------------------------------------#
class ResizerFactory:
    ''' Factory class to create resizer functions with a given configuration. '''
    
    #--- Map the resizer type to the resizer function ---#
    _RESIZER_MAP = {
        ResizerType.IDENTITY: identity,
        ResizerType.EXACT: exact,
        ResizerType.AROUND: around,
        ResizerType.SQUARE_AROUND_BOX: square_around_box,
        ResizerType.PAD_TO_SQUARE: pad_to_square
    }

    @classmethod
    def create(cls, config : ResizerConfig) -> Callable[[np.ndarray, Boxes], np.ndarray]:
        ''' Create a resizer function with the given configuration. 
        
        Parameters
        ----------
        config : ResizerConfig
            The configuration of the resizer.
            
        Returns
        -------
        Callable[[np.ndarray, Boxes], np.ndarray]
            The resizer function with the given configuration.
        '''
        #--- Check if the resizer type is valid ---#
        assert config.type in cls._RESIZER_MAP, f"Invalid resizer type: {config.type}"
        
        #--- Get the resizer function ---#
        resizer = cls._RESIZER_MAP[config.type]

        #--- Return the resizer function with the configuration ---#
        if config.type in [ResizerType.IDENTITY, ResizerType.PAD_TO_SQUARE]:
            return resizer
        elif config.type == ResizerType.SQUARE_AROUND_BOX:
            config_dict = {k: v for k, v in vars(config).items() if k not in ['type', 'min_proportion']}
            return partial(resizer, **config_dict)
        else:
            config_dict = {k: v for k, v in vars(config).items() if k != 'type'}
            return partial(resizer, **config_dict)
        
#------------------------------------------------------------------------------------------------------------#
# 6. 🛠️ UTILITY FUNCTIONS 
#------------------------------------------------------------------------------------------------------------#

#--- Utility function to extract the x1, y1, x2 and y2 coordinates from a ``Boxes`` instance ---#
def x1y1x2y2(box: Boxes) -> tuple:
    ''' Utility function that extracts the x1, y1, x2 and y2 coordinates from a ``Boxes`` instance.
    
    Parameters
    ----------
    box : Boxes
        The box to extract the coordinates from.
        
    Returns
    -------
    tuple
        A tuple containing the x1, y1, x2 and y2 coordinates. The coordinates are integers.
    '''
    #--- Extract the bounding box coordinates ---#
    x1, y1, x2, y2 = box.xyxy[0]
    
    #--- Return the coordinates as integers ---#
    return int(x1), int(y1), int(x2), int(y2)