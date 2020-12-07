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
        """Contructor. Uses by default no lstm.

        Parameters
        ----------
        name : str
            Name of the policy.
        n_ft_outpt : int
            Number of features that should be calculated by the feature
            detector.
        n_actions : int
            Number of available actions.
        seed : int
            Random seed that should be used by the agent processes.
        stddev : float
            (Deprecated) Standard deviation of a gaussian with zero mean to
            initialize the neurons.
        trainable : boolean
            If True the value of the neurons can be changed.
        check_numerics : boolean
            If True numeric values will be checked in tensorflow calculation to
            detect, e.g., NaN values.
        initializer : str
            Keras initializer that will be used (e.g. orthogonal).
        mode : str
            Full or Half. If Half, then only the action without the value will
            be calculated.
        """
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
    def action(self, obs):
        """Action selection of the actor critic model.

        Parameters
        ----------
        obs : np.ndarray
            Observation.

        Returns
        -------
        int
            The chosen action

        """
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
        """Initialize the variables/layers that process the output of the
        feature extractor.

        Parameters
        ----------
        name : str
            Name of the policy.
        n_ft_outpt : int
            Number of features that should be calculated by the feature
            detector.
        n_actions : int
            Number of available actions.
        seed : int
            Random seed that should be used by the agent processes.
        stddev : float
            (Deprecated) Standard deviation of a gaussian with zero mean to
            initialize the neurons.
        trainable : boolean
            If True the value of the neurons can be changed.
        check_numerics : boolean
            If True numeric values will be checked in tensorflow calculation to
            detect, e.g., NaN values.
        initializer : str
            Keras initializer that will be used (e.g. orthogonal).
        mode : str
            Full or Half. If Half, then only the action without the value will
            be calculated.
        """
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
        """Initialize the feature extractor.

        Parameters
        ----------
        name : str
            Name of the policy.
        n_ft_outpt : int
            Number of features that should be calculated by the feature
            detector.
        n_actions : int
            Number of available actions.
        seed : int
            Random seed that should be used by the agent processes.
        stddev : float
            (Deprecated) Standard deviation of a gaussian with zero mean to
            initialize the neurons.
        trainable : boolean
            If True the value of the neurons can be changed.
        check_numerics : boolean
            If True numeric values will be checked in tensorflow calculation to
            detect, e.g., NaN values.
        initializer : str
            Keras initializer that will be used (e.g. orthogonal).
        mode : str
            Full or Half. If Half, then only the action without the value will
            be calculated.
        """
        pass

    @abstractmethod
    def get_vars(self):
        """Returns the weights of the neural net.

        Returns
        -------
        list(tf.Tensor)
            List of the weight tensors.

        """
        pass

    @abstractmethod
    def reset(self):
        """Reset the neural net."""
        pass

    @abstractmethod
    def preprocess(self, obs):
        """Optional preprocessing of the observation.

        Parameters
        ----------
        obs : np.ndarray
            The observation.

        Returns
        -------
        np.ndarray
            Observation that can be input in the neural net.

        """
        pass

    def preprocess_action(self, action):
        """Optional preprocessing of the action. This method can be overwritten.
        For example, the method is useful if the action is returned as
        tf.Tensor.

        Parameters
        ----------
        action : int
            The action.

        Returns
        -------
        int
            The action as int.

        """
        return action

    def save(self, directory, filename):
        """Save the weights of the neural net as npz file.

        Parameters
        ----------
        directory : str
            Directory where the weights should be saved.
        filename : str
            Filename - e.g. name of the neural net.
        """
        mkdir(directory)
        vars_ = self.get_vars()
        var_dict = {}
        for var_ in vars_:
            var_dict[str(var_.name)] = np.array(var_.value())
        np.savez(directory + "/" + filename + ".npz", **var_dict)

    def load(self, directory, filename):
        """Load the weights of the neural net from a npz file.

        Parameters
        ----------
        directory : str
            Directory where the weights should be saved.
        filename : str
            Filename - e.g. name of the neural net.
        """
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
