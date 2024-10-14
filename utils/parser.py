''' utils.parser.py
>>> 💥 This file contains utility functions for the project.
'''

#------------------------------------------------------------------------------#
# IMPORT LIBRARIES AND/OR MODULES                                              #
#------------------------------------------------------------------------------#
import argparse

#--- Class: ParseKwargs (argparse.Action) ---#
class ParseKwargs(argparse.Action):

    def __call__(
        self,
        parser : argparse.ArgumentParser,           # The parser object responsible for parsing the command line arguments (e.g., parser)
        namespace : argparse.Namespace,             # The namespace object that contains the parsed arguments (e.g., args)
        values : dict,                              # The keyword arguments to parse (e.g., ['key1=value1', 'key2=value2'])
        option_string : str = None                  # The option string that triggered the action (e.g., '--kwargs')
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
        ''' This function converts the value to the appropriate data type.

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
    #--- Parse the keyword arguments ---#
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
    
    # Print the keyword arguments
    print(args.kwargs)