'''
A general outline for the plotting classes, and some commonly used
functions
'''
class DrawData():
    def __init__(self):
        self.xlimit = [0, 0]
        self.ylimit = [0, 0]
        self.zlimit = [0, 0]
        self.projection = None

    def check_plot_limits(self, x, y, z=None):
        '''
        Accepts lists for x and y, or x,y and z data, and returns the min and
        max values. This is used for setting plotting limits

        PARAMETERS
        ----------
        x: list of data
        y: list of data
        z: list of data, Optional (Default: None)
            can be left as None if checking data for a 2d plot
        '''
        if x.ndim > 1:
            self.xlimit[0] = min(min(x.min(axis=1)), self.xlimit[0])
            self.xlimit[1] = max(max(x.max(axis=1)), self.xlimit[1])
        else:
            self.xlimit[0] = min(min(x), self.xlimit[0])
            self.xlimit[1] = max(max(x), self.xlimit[1])

        if y.ndim > 1:
            self.ylimit[0] = min(min(y.min(axis=1)), self.ylimit[0])
            self.ylimit[1] = max(max(y.max(axis=1)), self.ylimit[1])
        else:
            self.ylimit[0] = min(min(y), self.ylimit[0])
            self.ylimit[1] = max(max(y), self.ylimit[1])

        if z is not None:
            if z.ndim > 1:
                self.zlimit[0] = min(min(z.min(axis=1)), self.zlimit[0])
                self.zlimit[1] = max(max(z.max(axis=1)), self.zlimit[1])
            else:
                self.zlimit[0] = min(min(z), self.zlimit[0])
                self.zlimit[1] = max(max(z), self.zlimit[1])

    def make_list(self, parameter):
        '''
        Returns the parameter passed in to a list if it is not already one

        PARAMETERS
        ----------
        parameter: any data type to be converted to a list
        '''
        if not isinstance(parameter, list):
            parameter = [parameter]
        return parameter

    def plot(self):
        raise Exception(
            'ERROR: The instantiated subclass is missing a plot function')
