import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import datetime
from multiprocessing import Process, Value
import json
from .utils import\
    parse_float_value,\
    parse_int_value,\
    parse_bool_value,\
    parse_tuple_int_value,\
    _parse_tuple_int_value,\
    get_type


class BaseRemoteTrainer(ABC):
    def __init__(self, args_file, types_file):
        self.train = Value("i", True)
        with open(args_file) as f:
            self.params = json.load(f)
        with open(types_file) as f:
            self.params_types = json.load(f)
        value, msg = _parse_tuple_int_value(
            self.params["state_size"], self.params_types["state_size"])
        if value == "error":
            raise Exception(msg)
        print("state value:", value)
        self.params["state_size"] = value
        self.train_process = None

    def start(self, shared_value, params):
        print("start")
        mem = params["main_gpu_mem"]
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=mem)])
                logical_gpus = tf.config.experimental.list_logical_devices('GPU')
                print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
            except RuntimeError as e:
                # Virtual devices must be set before GPUs have been initialized
                print(e)

        env_name = params["env_name"]
        n_actions = params["n_actions"]
        n_ft_outpt = params["n_ft_outpt"]
        test_interval = params["test_interval"]
        # write_tensorboard = False

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "./logs/" + env_name + "/train/" + current_time

        model_name = params["model_name"]
        model_dir = "./models/" + env_name + "/" + model_name

        policy_type = get_type(params["policy_path"], params["policy_type"])
        env_type = get_type(params["env_path"], params["env_type"])

        env_args = {}
        if params["data_provider_path"]:
            if params["data_provider_path"] != "":
                data_prov_type = get_type(
                    params["data_provider_path"], "DataProvider")
                env_args = {
                    "data_prov_type": data_prov_type,
                    "max_scenes": params["max_scenes"],
                    "train_p": params["train_p"],
                    "train_mode": params["train_mode"]
                }
        self.train_method(
            shared_value,
            params,
            env_type,
            env_args,
            policy_type,
            n_ft_outpt,
            n_actions,
            train_log_dir,
            model_dir,
            model_name,
            test_interval)

    @abstractmethod
    def train_method(
            self,
            shared_value,
            params,
            env_type,
            env_args,
            policy_type,
            n_ft_outpt,
            n_actions,
            train_log_dir,
            model_dir,
            model_name,
            test_interval):
        pass

    def execute_command(self, usr_cmds):
        usr_cmds = usr_cmds.split()
        for i in range(len(usr_cmds)):
            usr_cmd = usr_cmds[i]
            if usr_cmd == "start":
                self.train.value = True
                self.train_process =\
                    Process(target=self.start, args=(self.train, self.params))
                self.train_process.start()
                return str(self.params.items()) + "\nok"
            elif usr_cmd == "stop":
                if self.train_process:
                    self.train.value = False
                self.train_process.terminate()
                self.train_process.join()
                return "training stopped"
            elif usr_cmd in self.params.keys():
                type = self.params_types[usr_cmd]
                if type[-5:] == "float":
                    value, msg = parse_float_value(usr_cmds, i, type)
                elif type[-3:] == "int":
                    value, msg = parse_int_value(usr_cmds, i, type)
                elif type == "str":
                    value = usr_cmds[i + 1]
                    msg = "ok"
                elif type[:5] == "tuple":
                    if type[-3:] == "int":
                        value, msg =\
                            parse_tuple_int_value(usr_cmds, i, type)
                elif type == "bool":
                    value, msg = parse_bool_value(usr_cmds, i)
                if value != "error":
                    self.params[usr_cmd] = value
                i += 1
                return msg
            elif usr_cmd == "help":
                return "start,\nstop\n" + str(self.params.keys()) + ",\nhelp,\nparams"
            elif usr_cmd == "params":
                return str(self.params.items())
        return "unknown command"
