from abc import ABC, abstractmethod


class BaseNet(ABC):
    """Short summary.

    Parameters
    ----------
    name : str
        Name of the neural net.
    outpt : int
        Number of features that should be calculated by the feature
        detector.
    trainable : boolean
        If True the value of the neurons can be changed.
    seed : int
        Random seed that should be used.
    check_numerics : boolean
        If True numeric values will be checked in tensorflow calculation to
        detect, e.g., NaN values.

    Attributes
    ----------
    name : str
        Name of the neural net.
    seed : int
        Random seed that should be used.
    trainable : boolean
        If True the value of the neurons can be changed.
    outpt : int
        Number of features that should be calculated by the feature
        detector.
    check_numerics : boolean
        If True numeric values will be checked in tensorflow calculation to
        detect, e.g., NaN values.

    """
    def __init__(
            self,
            name,
            outpt,
            trainable=True,
            seed=None,
            check_numerics=False):
        """Short summary.

        Parameters
        ----------
        name : str
            Name of the neural net.
        outpt : int
            Number of features that should be calculated by the feature
            detector.
        trainable : boolean
            If True the value of the neurons can be changed.
        seed : int
            Random seed that should be used.
        check_numerics : boolean
            If True numeric values will be checked in tensorflow calculation to
            detect, e.g., NaN values.

        """
        super().__init__()
        self.name = name
        self.seed = seed
        self.trainable = trainable
        self.outpt = outpt
        self.check_numerics = check_numerics

    @abstractmethod
    def get_vars(self):
        """This method should return the neurons of the neural net.

        Returns
        -------
        tf.Tensor
            Neurons as variable.
        """
        pass

    @abstractmethod
    def compute(self, obs):
        """Compute a feature vector from the observation.

        Parameters
        ----------
        obs : tf.Tensor
            Observation from which the network will calculate features as
            vector.

        Returns
        -------
        tf.Tensor
            Feature vector.

        """
        pass
