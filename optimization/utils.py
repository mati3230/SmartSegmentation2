import numpy as np
import os


def normalize(x, eps=1e-5, axis=0):
    """Normalize via standardisation.

    Parameters
    ----------
    x : np.ndarray
        Array that should be normalized.
    eps : float
        Constant that is used in case of a 0 standard deviation.
    axis : int
        Normalization direction.

    Returns
    -------
    np.ndarray
        Description of returned object.

    """
    return (x - np.mean(x, axis=axis)) / (np.std(x, axis=axis) + eps)


def mkdir(directory):
    """Method to create a new directory.

    Parameters
    ----------
    directory : str
        Relative or absolute path.
    """
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    """Check if a file exists.

    Parameters
    ----------
    filepath : str
        Relative or absolute path to a file.

    Returns
    -------
    boolean
        True if the file exists.

    """
    return os.path.isfile(filepath)


def save_config(log_dir, config):
    """Save a custom configuration such as learning rate.

    Parameters
    ----------
    log_dir : str
        Directory where the configuration should be placed.
    config : str
        String with the configuration.
    """
    text_file = open(log_dir + "/config.txt", "w")
    text_file.write(config)
    text_file.close()


def importer(name, root_package=False, relative_globals=None, level=0):
    """Imports a python module.

    Parameters
    ----------
    name : str
        Name of the python module.
    root_package : boolean
        See https://docs.python.org/3/library/functions.html#__import__.
    relative_globals : type
        See https://docs.python.org/3/library/functions.html#__import__.
    level : int
        See https://docs.python.org/3/library/functions.html#__import__.

    Returns
    -------
    type
        Python module. See
        https://docs.python.org/3/library/functions.html#__import__.

    """
    return __import__(name, locals=None, # locals has no use
                      globals=relative_globals,
                      fromlist=[] if root_package else [None],
                      level=level)


def get_type(path_str, type_str):
    """Load a specific class type.

    Parameters
    ----------
    path_str : str
        Path to the python file of the desired class.
    type_str : str
        String of the class name.

    Returns
    -------
    type
        Requested class type.

    """
    module = importer(path_str)
    mtype = getattr(module, type_str)
    return mtype


def parse_float_value(usr_cmds, i, type):
    """Parse a float value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where float values are stored.
    i : int
        The i-th value should be the name of the float value. The (i+1)-th
        value should be the float value itself.
    type : str
        Should be whether:
            real pos float: > 0
            real neg float: < 0
            pos float: >= 0
            neg float: <= 0

    Returns
    -------
    tuple(float, str)
        Returns the float value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    try:
        value = float(value)
    except ValueError:
        return ("error", "Value '" + str(value) + "' cannot be converted to " + str(type))
    if type == "real pos float" and value <= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "real neg float" and value >= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "pos float" and value < 0:
        return ("error", "Value has to be greater than or equal 0")
    elif type == "neg float" and value > 0:
        return ("error", "Value has to be greater than or equal 0")
    return (value, "ok")


def _parse_int_value(value, type):
    """Parse a int value.

    Parameters
    ----------
    value : int
        An int value.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    type
        Returns the int value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = int(value)
    except ValueError:
        return ("error", "Value '" + str(value) + "' cannot be converted to " + str(type))
    if type == "real pos int" and value <= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "real neg int" and value >= 0:
        return ("error", "Value has to be greater than 0")
    elif type == "pos int" and value < 0:
        return ("error", "Value has to be greater than or equal 0")
    elif type == "neg int" and value > 0:
        return ("error", "Value has to be greater than or equal 0")
    return (value, "ok")


def parse_int_value(usr_cmds, i, type):
    """Parse a float value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(int, str)
        Returns the int value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_int_value(value, type)


def parse_bool_value(usr_cmds, i):
    """Parse a boolean value from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the int value. The (i+1)-th
        value should be the int value itself.

    Returns
    -------
    tuple(boolean, str)
        Returns the boolean value and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value - expected True or False")
    if value != "True" and value != "False":
        return ("error", "Invalid value - expected True or False")
    if value == "True":
        return (True, "ok")
    return (False, "ok")


def _parse_tuple_int_value(value, type):
    """Parse an int tuple.

    Parameters
    ----------
    value : tuple(int)
        Tuple that should be parsed.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(tuple(int), str)
        Returns the tuple(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    if len(value) < 4:
        return ("error", "Tuple must be at least '(x, )'")
    if value[0] != "(" or value[-1] != ")":
        return ("error", "Tuple must be at least '(x, )'")
    t_values = value[1:-1]
    t_values = t_values.split(",")
    if len(t_values) < 2:
        return ("error", "Tuple must be at least '(x, )'")
    result = []
    for i in range(len(t_values)):
        t_val = t_values[i]
        if t_val == "" or t_val == " ":
            continue
        int_type = type[6:]
        value, msg = _parse_int_value(t_val, int_type)
        if value == "error":
            return (value, msg)
        result.append(value)
    result = tuple(result)
    return (result, "ok")


def parse_tuple_int_value(usr_cmds, i, type):
    """Parse a tuple with int values from a user command list.

    Parameters
    ----------
    usr_cmds : list(str)
        List where int values are stored.
    i : int
        The i-th value should be the name of the tuple. The (i+1)-th
        value should be the tuple itself.
    type : str
        real pos int: > 0
        real neg int: < 0
        pos int: >= 0
        neg int: <= 0

    Returns
    -------
    tuple(tuple(int), str)
        Returns the tuple(int) and a string with 'ok' if no error occurs. In
        case of an error a tuple('error', error message) is returned.

    """
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_tuple_int_value(value, type)
