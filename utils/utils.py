''' utils.utils.py
>>> 💥 This file contains utility functions that are used in the project.
'''

#------------------------------------------------------------------------------#
# CLASSES AND FUNCTIONS                                                        #
#------------------------------------------------------------------------------#

#--- Class: ANSI ---#
class ANSI:
    ''' This class contains the ANSI escape codes for colors and styles. '''
    
    def __init__(self) -> None:
        ''' The constructor for the ANSI class. '''
        #--- Set the ANSI escape codes for colors and styles ---#
        self.ansi = {
            #--- Classic colors ---#
            "black": "\033[30m",
            "red": "\033[31m",
            "green": "\033[32m",
            "yellow": "\033[33m",
            "blue": "\033[34m",
            "magenta": "\033[35m",
            "cyan": "\033[36m",
            "white": "\033[37m",
            #--- Classic Background colors ---#
            "bg_black": "\033[40m",
            "bg_red": "\033[41m",
            "bg_green": "\033[42m",
            "bg_yellow": "\033[43m",
            "bg_blue": "\033[44m",
            "bg_magenta": "\033[45m",
            "bg_cyan": "\033[46m",
            "bg_white": "\033[47m",
            #--- Bright colors ---#
            "bright_black": "\033[90m",
            "bright_red": "\033[91m",
            "bright_green": "\033[92m",
            "bright_yellow": "\033[93m",
            "bright_blue": "\033[94m",
            "bright_magenta": "\033[95m",
            "bright_cyan": "\033[96m",
            "bright_white": "\033[97m",
            #--- Bright Background colors ---#
            "bg_bright_black": "\033[100m",
            "bg_bright_red": "\033[101m",
            "bg_bright_green": "\033[102m",
            "bg_bright_yellow": "\033[103m",
            "bg_bright_blue": "\033[104m",
            "bg_bright_magenta": "\033[105m",
            "bg_bright_cyan": "\033[106m",
            "bg_bright_white": "\033[107m",   
            #--- Styles ---#
            "end": "\033[0m",
            "bold": "\033[1m",
            "italic": "\033[3m",
            "underline": "\033[4m",
        }

    def __call__(self, *args) -> str:
        ''' This function returns the ANSI escape codes for the specified color and style.
        
        Parameters
        ----------
        args : tuple
            The input arguments. The first two arguments are the color and style, respectively. The last argument is the string to color.
        string : str, optional
            The input string. The default is "".
        
        Returns
        -------
        str
            The input string wrapped with ANSI escape codes for the specified color and style.
        '''
        #--- Return the input string if there is only one argument ---#
        if len(args) == 1: return args[0]
        
        #--- Get the color and style ---#
        *styles, string = args
        
        #--- Return the ANSI escape codes for the specified color and style ---#
        return "".join(self.ansi[x] for x in styles) + f"{string}" + self.ansi["end"]

#--- Function: Color and Style ---#
def color(*args) -> str:
    ''' This function returns the ANSI escape codes for the specified color and style.
        
    Parameters
    ----------
    args : tuple
        The input arguments. The first two arguments are the color and style, respectively. The last argument is the string to color.
    string : str, optional
        The input string. The default is "".
    
    Returns
    -------
    str
        The input string wrapped with ANSI escape codes for the specified color and style.
    '''
    #--- Return the ANSI escape codes for the specified color and style ---#
    return ANSI()(*args)