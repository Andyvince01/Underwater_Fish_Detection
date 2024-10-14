''' utils.logger.py
>>> 💥 This module contains the Logger class which is used to set up a logger.
'''

import logging, os, platform
from .utils import color

#--- Environment Booleans ---#
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

#------------------------------------------------------------------------------#
# CLASSES AND FUNCTIONS                                                        #
#------------------------------------------------------------------------------#

#--- Class: Logger ---#
class Logger:
    ''' This class is used to set up a logger. '''

    def __init__(self, logger_name : str, level : int = logging.INFO, on_file : str = 'log', on_screen : bool = False):
        ''' This class initializes a logger.
        
        Parameters
        ----------
        logger_name : str
            The name of the logger.
        level : int, optional
            The logging level. The default is logging.INFO.
        log_filename : str, optional
            The filename of the log. The default is None.
        screen : bool, optional
            Whether to log to the screen. The default is False.
        '''
        #--- Set the logger attributes ---#
        self.logger_name = logger_name
        self.level : int = self._set_level(level)
        self.on_file = 'log' if on_file is True else on_file
        self.on_screen = on_screen
                
        #--- Initialize the logger ---#
        self.logger = logging.getLogger(self.logger_name)      
        if self.logger.handlers: return                         # If the logger is already initialized, return
        
        #--- Set the logger formatter ---#
        self.formatter = logging.Formatter(f'\033[1m[%(asctime)s]\033[0m (%(name)s) - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        #--- Set up the logger to log to a file and/or the screen ---#
        if self.on_file: self._setup_log_file()
        if self.on_screen: self._setup_screen()

        #--- Set the logging level of the logger ---#
        self.logger.setLevel(self.level)
    
    #--- Public Functions ---#
    def debug(self, message : str) -> None:
        ''' This function logs a debug message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to debug and color the message of green ---#
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(color('green', self._emojis(message)))
                   
    def error(self, message : str) -> None:
        ''' This function logs an error message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to error and color the message of red ---#
        self.logger.setLevel(logging.ERROR)
        self.logger.error(color('red', 'underline', 'bold', self._emojis(message)))
    
    def info(self, message : str) -> None:
        ''' This function logs an info message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to info and color the message of black ---#
        self.logger.setLevel(logging.INFO)
        self.logger.info(self._emojis(message))
    
    def warning(self, message : str) -> None:
        ''' This function logs a warning message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        #--- Set the logging level to warning and color the message of yellow ---#
        self.logger.setLevel(logging.WARNING)
        self.logger.warning(color('yellow', 'underline', self._emojis(message)))
                 
    #--- Private Functions ---#
    def _emojis(self, string : str = '') -> str:
        ''' This function a platform-dependent emoji-safe version of the input string.
        
        Parameters
        ----------
        string : str, optional
            The input string. The default is "".

        Returns
        -------
        str
            The string with emojis. 
        '''
        #--- Return the string with emojis if the platform is not Windows ---#
        return string.encode().decode("ascii", "ignore") if WINDOWS else string
   
    def _set_level(self, level : str = 'info') -> int:
        ''' This function sets the level of the logger.
        
        Parameters
        ----------
        level : str, optional
            The logging level. The default is 'info'.
        
        Returns
        -------
        int
            The logging level. If the level is not found, it returns logging.NOTSET (0).
        '''
        #--- Set the logging level ---#
        try:
            return getattr(logging, level.upper())
        except AttributeError:
            return logging.NOTSET                       # Return the default level - All messages are processed!
         
    def _setup_log_file(self) -> None:
        ''' This function sets up the log file. '''
        #--- Set the filename of the log file and create the directory if it does not exist ---#
        file_path = os.path.join(os.path.abspath(os.path.dirname(__file__)), self.on_file)
        filename = os.path.join(file_path, f'{self.logger_name}.log')
        os.makedirs(file_path, exist_ok=True)

        #--- Set the file handler of the logger ---#
        fh = logging.FileHandler(filename=filename)
        fh.setLevel(self.level); fh.setFormatter(self.formatter)
        
        #--- Add the file handler to the logger ---#
        self.logger.addHandler(fh)

    def _setup_screen(self) -> None:
        ''' This function sets up the screen. '''
        #--- Set the stream handler of the logger ---#
        ch = logging.StreamHandler()
        ch.setLevel(self.level); ch.setFormatter(self.formatter)
        
        #--- Add the stream handler to the logger ---#
        self.logger.addHandler(ch)    
    
    #--- Helper Functions ---#
    def __str__(self) -> str:
        return f"Logger: {self.logger_name}{' at' + self.on_file if self.on_file else ''} with level {self.level} and screen={self.on_screen}."

if __name__ == '__main__':
    #--- Test the Logger Class ---#
    logger = Logger(logger_name='base', level='Invalid', on_file=True, on_screen=True)
    #--- Test the Logger Class ---#
    logger.info('Initial Dataset Finished 🎉.')
    logger.debug('Initial Model Finished 🎉.')
    logger.error('Begin Model Inference ❌.')