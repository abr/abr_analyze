import numpy as np
from terminaltables import AsciiTable

# Function to print out parameters into an ASCII Table
def print_params(data, title=None, invert=False):
    '''
    data must be in the following format for N tests
    dict = {
            'test1': {'common_param1': data,
                      'common_param2': data}
            'test2': {'common_param1': data,
                      'common_param2': data}
            }
    '''

    # HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    ENDC = '\033[0m'
    # BOLD = '\033[1m'
    # UNDERLINE = '\033[4m'

    tests = list(data.keys())
    test_labels = tests[:]

    # set the table header colour
    for mm, header in enumerate(test_labels):
        test_labels[mm] = '\033[95m' + header + '\033[0m'

    formatted_data = []
    column_labels = ['test_name']
    for nn, test in enumerate(tests):
        temp_data = []
        keys = list(data[test].keys())
        for key in keys:
            if key not in column_labels:
                column_labels.append(key)
            loaded = data[test][key]
            if loaded is True:
                col = GREEN
            elif loaded is False:
                col = RED
            else:
                col = YELLOW
            loaded = '%s%s%s'%(col, loaded, ENDC)
            temp_data.append(loaded)
        formatted_data.append([test_labels[nn]] + temp_data)

    table_data = []
    table_data.append(column_labels)
    for row in formatted_data:
        table_data.append(row)
    if invert:
        table_data = list(np.array(table_data).T)
    table = AsciiTable(table_data)
    if title is not None:
        print(BLUE + '----------' + title + '----------' + ENDC)
    print(table.table)
