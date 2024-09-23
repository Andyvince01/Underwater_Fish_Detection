''' utils.py
> This file contains utility functions for the project.
'''

import argparse, os

# Define the path to the fishscale data directory
FISHSCALE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'fishscale_data.yaml')

# Class to parse keyword arguments
class ParseKwargs(argparse.Action):

    def __call__(
        self,
        parser : argparse.ArgumentParser, 
        namespace : argparse.Namespace,
        values : dict,
        option_string : str = None
    ):
        '''Parse the keyword arguments and add them to the namespace.

        Parameters
        ----------
        parser : argparse.ArgumentParser
            The parser object.
        namespace : argparse.Namespace
            The namespace object.
        values : dict
            The keyword arguments to parse.
        '''
        # Set the attribute in the namespace as a dictionary
        setattr(namespace, self.dest, {})
        # Get the keyword arguments
        for value in values:
            # Split the key-value pair by the colon (':')
            key, value = value.split('=')
            # Set the attribute in the namespace
            getattr(namespace, self.dest)[key] = self._convert_value(value)

    # Private Method
    def _convert_value(self, value : str) -> any:
        '''Convert the value to the appropriate type.

        Parameters
        ----------
        value : str
            The value to convert.

        Returns
        -------
        any
            The converted value.
        '''
        # Convert the value to 'boolean' if it is 'True' or 'False'
        if value.lower() in ['true', 'false']:
            return value.lower() == 'true'
        
        # Convert the value to 'int' if it is an integer
        try:
            return int(value)
        except ValueError:
            pass
        
        # Convert the value to 'float' if it is a float
        try:
            return float(value)
        except ValueError:
            pass

        # Return the value as a string            
        return value

if __name__ == '__main__':
    # Define the command line arguments
    parser = argparse.ArgumentParser(description='Parse keyword arguments.')
    # Add the arguments to the parser
    parser.add_argument(
        '--kwargs',
        nargs='*',
        action=ParseKwargs,
        help='The keyword arguments to parse.'
    )
    # Parse the arguments
    args = parser.parse_args()
    
    # Print the parsed keyword arguments
    print(args.kwargs)