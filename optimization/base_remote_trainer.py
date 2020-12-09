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
    """This class executes the training. Moreover, parameters
    of the training can be changed with user commands. The user commands
    should be represented as str list.

    Parameters
    ----------
    args_file : str
        Path (relative or absolute) to a json file where the parameters are
        specified.
    types_file : str
        Path (relative or absolute) to a json file where the types of the
        parameters are specified.

    Attributes
    ----------
    train : multiprocessing.Value
        Shared value of the multiprocessing library to stop the training
        process.
    params : dict
        The parameters of the _args.json file.
    params_types : dict
        The types of the parameters that are specified in the _types.json file.
    train_process : multiprocessing.Process
        Master Process of the training.

    """
    def __init__(self, args_file, types_file):
        """Constructor.

        Parameters
        ----------
        args_file : str
            Path (relative or absolute) to a json file where the parameters are
            specified.
        types_file : str
            Path (relative or absolute) to a json file where the types of the
            parameters are specified.
        """
        # multiprocessing value for stopping the training across different
        # processes
        self.train = Value("i", True)
        # load the parameters and their types
        with open(args_file) as f:
            self.params = json.load(f)
        with open(types_file) as f:
            self.params_types = json.load(f)
        # check the state size
        value, msg = _parse_tuple_int_value(
            self.params["state_size"], self.params_types["state_size"])
        if value == "error":
            raise Exception(msg)
        print("state value:", value)
        self.params["state_size"] = value
        self.train_process = None

    def start(self, shared_value, params):
        """Preparation for the training and start of the training.

        Parameters
        ----------
        shared_value : type
            multiprocessing value for stopping the training across different
            processes
        params : dictionary
            Dictionary where parameters such as the path of the environment
            class or data provider class are specified. It should also store
            parameters for the training such as the batch size.

        """
        print("start")
        # gpu memory that should be used by the main process
        mem = params["main_gpu_mem"]
        # force tf to use gpus
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                # setup memory usage for the main process
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

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = "./logs/" + env_name + "/train/" + current_time

        model_name = params["model_name"]
        model_dir = "./models/" + env_name + "/" + model_name

        policy_type = get_type(params["policy_path"], params["policy_type"])
        env_type = get_type(params["env_path"], params["env_type"])

        env_args = {}
        if "data_provider_path" in params:
            if params["data_provider_path"] != "":
                data_prov_type = get_type(
                    params["data_provider_path"], "DataProvider")
                env_args = {
                    "data_prov_type": data_prov_type,
                    "max_scenes": params["max_scenes"],
                    "train_p": params["train_p"],
                    "train_mode": params["train_mode"]
                }
        # start the training
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
        """This methods starts the training procedure.

        Parameters
        ----------
        shared_value : multiprocessing.Value
            Shared value of the multiprocessing library to stop the training
            process over multi processes.
        params : dictionary
            Dictionary where parameters such as the path of the environment
            class or data provider class are specified. It should also store
            parameters for the training such as the batch size. The parameter
            types can be found in, e.g., ppo2_types.json.
        env_type : type
            Type of the environment.
        env_args : dictionary
            Parameters of the environment.
        policy_type : type
            Type of the policy such as a specific actor critic neural net.
        n_ft_outpt : int
            Number of features the will a specific net should output before a
            action and a state value mlp are calculated. In other words, the
            number of neurons that is used by the action and value function.
            Note that we split a neural net in a general feature approximator
            and, e.g., an action approximator
        n_actions : int
            Number of actions that are available in the environment.
        train_log_dir : str
            Directory where the logs will be saved.
        model_dir : str
            Directory where the weights of the nets will be saved.
        model_name : str
            Name of the neural net.
        test_interval : int
            Number that specifies after how many training updates a test is
            calculated. Note that we use a train test split.
        """
        pass

    def execute_command(self, usr_cmds):
        """Execute a user command that could originate from a, e.g., messaging
        platform such as telegram.

        Parameters
        ----------
        usr_cmds : list
            List that contains user commands as strings. Commands are, for
            example, 'start' or 'stop'. The user can also set arguments
            according to the available types of the algorithm (see
            ppo2_types.json).

        Returns
        -------
        str
            Answer to the command that specifies if the commands could be
            executed.

        """
        usr_cmds = usr_cmds.split()
        for i in range(len(usr_cmds)):
            usr_cmd = usr_cmds[i]
            # start the training
            if usr_cmd == "start":
                self.train.value = True
                self.train_process =\
                    Process(target=self.start, args=(self.train, self.params))
                self.train_process.start()
                return str(self.params.items()) + "\nok"
            # stop the training
            elif usr_cmd == "stop":
                # stop other processes
                if self.train_process:
                    self.train.value = False
                self.train_process.terminate()
                self.train_process.join()
                return "training stopped"
            # user will change a parameter of the training
            elif usr_cmd in self.params.keys():
                # detect the type of the parameter
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
                # change the parameter
                if value != "error":
                    self.params[usr_cmd] = value
                i += 1
                return msg
            elif usr_cmd == "help":
                return "start,\nstop\n" + str(self.params.keys()) + ",\nhelp,\nparams"
            elif usr_cmd == "params":
                return str(self.params.items())
        return "unknown command"
