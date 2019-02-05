class DrawData():
    '''

    '''
    def __init__(self):
        '''

        '''
        self.xlimit = [0,0]
        self.ylimit = [0,0]
        self.zlimit = [0,0]

    def check_plot_limits(self, x, y, z=None):
        '''

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

    def plot(self):
        '''

        '''
        raise Exception ('ERROR: The instantiated subclass is missing a plot function')
