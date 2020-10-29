from abc import ABC, abstractmethod
from .utils import mkdir, file_exists
import numpy as np


class BasePolicy(ABC):

    def __init__(
            self,
            name,
            n_ft_outpt,
            n_actions,
            seed=None,
            stddev=0.3,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full"):
        super().__init__()
        self.use_lstm = False
        self.name = name
        self.seed = seed
        self.check_numerics = check_numerics
        self.n_actions = n_actions
        self.trainable = trainable
        self.mode = mode
        if mode == "pre":
            return
        self.init_net(
            name=name,
            n_ft_outpt=n_ft_outpt,
            seed=seed,
            stddev=stddev,
            trainable=trainable,
            check_numerics=check_numerics,
            initializer=initializer,
            mode=mode)
        self.init_variables(
            name=name,
            n_ft_outpt=n_ft_outpt,
            n_actions=n_actions,
            stddev=stddev,
            trainable=trainable,
            seed=seed,
            initializer=initializer,
            mode=mode)
        # self.reset()

    @abstractmethod
    def action(self, state):
        pass

    @abstractmethod
    def init_variables(
            self,
            name,
            n_ft_outpt,
            n_actions,
            stddev=0.3,
            trainable=True,
            seed=None,
            initializer="glorot_uniform",
            mode="full"):
        pass

    @abstractmethod
    def init_net(
            self,
            name,
            n_ft_outpt,
            seed=None,
            stddev=0.3,
            trainable=True,
            check_numerics=False,
            initializer="glorot_uniform",
            mode="full"):
        pass

    @abstractmethod
    def get_vars(self):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def preprocess(self, state):
        pass

    def preprocess_action(self, action):
        return action

    def save(self, directory, filename):
        mkdir(directory)
        vars_ = self.get_vars()
        var_dict = {}
        for var_ in vars_:
            var_dict[str(var_.name)] = np.array(var_.value())
        np.savez(directory + "/" + filename + ".npz", **var_dict)

    def load(self, directory, filename):
        # print("load", directory, filename, "...")
        filepath = directory + "/" + filename + ".npz"
        if not file_exists(filepath):
            raise Exception("File path '" + filepath + "' does not exist")
        model_data = np.load(filepath, allow_pickle=True)
        vars_ = self.get_vars()
        if len(vars_) != len(model_data):
            keys = list(model_data.keys())
            for i in range(min(len(vars_), len(model_data))):
                print(vars_[i].name, "\t", keys[i])
            raise Exception("data mismatch")
        i = 0
        for key, value in model_data.items():
            varname = str(vars_[i].name)
            if np.isnan(value).any():
                raise Exception("loaded value is NaN")
            if key != varname:
                raise Exception(
                    "Variable names mismatch: " + key + ", " + varname)
            vars_[i].assign(value)
            i += 1
