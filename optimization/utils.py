import numpy as np
import os


def normalize(x, eps=1e-5, axis=0):
    return (x - np.mean(x, axis=axis)) / (np.std(x, axis=axis) + eps)


def mkdir(directory):
    if not os.path.isdir(directory):
        os.makedirs(directory)


def file_exists(filepath):
    return os.path.isfile(filepath)


def save_config(log_dir, config):
    text_file = open(log_dir + "/config.txt", "w")
    text_file.write(config)
    text_file.close()


def importer(name, root_package=False, relative_globals=None, level=0):
    return __import__(name, locals=None, # locals has no use
                      globals=relative_globals,
                      fromlist=[] if root_package else [None],
                      level=level)


def get_type(path_str, type_str):
    module = importer(path_str)
    mtype = getattr(module, type_str)
    return mtype


def parse_float_value(usr_cmds, i, type):
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
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_int_value(value, type)


def parse_bool_value(usr_cmds, i):
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
    try:
        value = usr_cmds[i + 1]
    except IndexError:
        return ("error", "No value")
    return _parse_tuple_int_value(value, type)
