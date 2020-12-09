from abc import ABC, abstractmethod


class BaseDataProvider(ABC):
    """Abstract class for an environment that uses data from a storage.
    For example, a realization of this class is used to load and preprocess
    point cloud data (see '../scannet_provider.py')"""
    def __init__(self):
        super().__init__()

    @abstractmethod
    def get_cloud_and_segments(self):
        """The method should return a point cloud and the corresponding
        segments.

        Returns
        -------
        np.ndarray
            Point Cloud
        np.ndarray
            Vector with segment numbers for each point
        """
        pass
