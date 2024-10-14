import logging, os, platform

# Define the environment booleans
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans

BOLD_START = "\033[1m"
BOLD_END = "\033[0m"

class Logger:
    ''' This class is used to set up a logger. '''

    def __init__(self, logger_name : str, level : int = logging.INFO, log_filename = None, screen : bool = False):
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
        # Initialize the class variables
        self.logger_name = logger_name
        self.level = level
        self.log_filename = log_filename
        self.screen = screen
        
        # Initialize the logger
        self.logger = logging.getLogger(self.logger_name)      
        # Prevent duplicate handlers
        if self.logger.handlers: return
        
        # Set the format of the logger to include the timestamp
        self.formatter = logging.Formatter(f'{BOLD_START}[%(asctime)s]{BOLD_END} (%(name)s) - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

        # Setup the logger handlers
        if self.log_filename is not None:
            self._setup_log_file()
        if self.screen:
            self._setup_screen()

        self.logger.setLevel(self.level)  # Set logger level
                        
    def info(self, message : str) -> None:
        ''' This function logs an info message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        self.logger.setLevel(logging.INFO)
        self.logger.info(self._emojis(message))
    
    def error(self, message : str) -> None:
        ''' This function logs an error message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        self.logger.setLevel(logging.ERROR)
        self.logger.error(self._emojis(message))
    
    def debug(self, message : str) -> None:
        ''' This function logs a debug message.
        
        Parameters
        ----------
        message : str
            The input message.
        '''
        self.logger.setLevel(logging.DEBUG)
        self.logger.debug(self._emojis(message))
    
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
        return string.encode().decode("ascii", "ignore") if WINDOWS else string
    
    def _setup_log_file(self) -> None:
        ''' This function sets up the log file. '''
        # Create the directory if it does not exist
        self.filename = os.path.join(self.log_filename, f'{self.logger_name}.log')
        os.makedirs(self.log_filename, exist_ok=True)
        # Set the file handler of the logger
        fh = logging.FileHandler(filename=self.filename)
        fh.setLevel(self.level)
        fh.setFormatter(self.formatter)
        self.logger.addHandler(fh)

    def _setup_screen(self) -> None:
        ''' This function sets up the screen. '''
        # Set the stream handler of the logger
        ch = logging.StreamHandler()
        ch.setLevel(self.level)
        ch.setFormatter(self.formatter)
        self.logger.addHandler(ch)

    @property
    def logger_name(self) -> str:
        ''' This function returns the name of the logger. '''
        return self._logger_name
    
    @logger_name.setter
    def logger_name(self, value : str) -> None:
        ''' This function sets the name of the logger. '''
        self._logger_name = value
        
    def __str__(self) -> str:
        return f"Logger: {self.logger_name}{' at' + self.log_filename if self.log_filename else ''} with level {self.level} and screen={self.screen}."

# if __name__ == '__main__':
#     # Set up the logger
#     logger = Logger(logger_name='base', root='UIE-DM/logs', level=logging.INFO, screen=True)
#     logger.setup_logger()
#     # Log an info message
#     logger.info('Initial Dataset Finished 🎉.')
#     logger.info('Initial Model Finished 🎉.')
#     logger.info('Begin Model Inference ❌.')