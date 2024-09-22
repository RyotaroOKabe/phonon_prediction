

import inspect
def make_dict(variables):
    frame = inspect.currentframe().f_back
    dictionary = {name: value for name, value in frame.f_locals.items() if id(value) in [id(var) for var in variables]}
    return dictionary