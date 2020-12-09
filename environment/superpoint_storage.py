import numpy as np
import json


class SuperpointStorage:
    """This class stores the superpoints with their point idxs and neighbours.

    Parameters
    ----------
    pns_orig: list
        will be a tuple where the first element is a list of
        points idxs of the superpoints and the second element is list of
        neighbour superpoints of the superpoints (pns is shorthand for
        'p'oints indices and 'n'eighbour's' of a superpoint)
    sort_mode: boolean
        if True, an ascending sortation according to the sizes of the
        superpoints will be calculated.

    Attributes
    ----------
    _sort_mode : boolean
        if True, an ascending sortation according to the sizes of the
        superpoints will be calculated.
    _pns_orig : list
        will be a tuple where the first element is a list of
        points idxs of the superpoints and the second element is list of
        neighbour superpoints of the superpoints (pns is shorthand for
        'p'oints indices and 'n'eighbour's' of a superpoint)
    _blacklist : list(list(int))
        Enter a superpoint index and get blacklisted superpoint indices.
    _lengths : np.ndarray
        The array contains the cardinalities of the superpoints.
    _orig_lengths : np.ndarray
        The array contains the initial cardinalities of the superpoints. Will
        not be changed.
    _sorted_idxs : np.ndarray
        Sorted superpoint indices according to their cardinalities. In this
        array, the sorted array position are stored. They are calculated by a
        argsort operation.
    _orig_sorted_idxs : np.ndarray
        Ground truth sorted superpoint indices according to their cardinalities.
        In this array, the sorted array position are stored. They are
        calculated by a argsort operation.
    """
    def __init__(self, pns_orig=None, sort_mode=False):
        """
        Constructor.

        Parameters
        ----------
        pns_orig: list
            will be a tuple where the first element is a list of
            points idxs of the superpoints and the second element is list of
            neighbour superpoints of the superpoints (pns is shorthand for
            'p'oints indices and 'n'eighbour's' of a superpoint)
        sort_mode: boolean
            if True, an ascending sortation according to the sizes of the
            superpoints will be calculated.
        """
        self._sort_mode = sort_mode
        if pns_orig:
            self._pns_orig = [list(), list()]
            self._pns_orig[0] = pns_orig[0].copy()
            self._pns_orig[1] = pns_orig[1].copy()
            self.init()

    def init(self):
        """
        Reset the blacklist and _pns which will be changed with unions of
        superpoints.
        """
        self._blacklist = []
        self._pns = []
        if self._sort_mode:
            self._sort()

    def save(self, id):
        """
        Save the points indices and the neighbours of the superpoints according
        to a point cloud with a certain id.

        Parameters
        ----------
        id : int
            Index of a scene or point cloud.
        """
        tmp_pns = [list(), list()]
        for i in range(len(self._pns_orig[0])):
            tmp_pns[0].append(self._pns_orig[0][i].tolist())
            tmp_pns[1].append(self._pns_orig[1][i].tolist())

        with open("./cache/"+id+"/pns_orig.json", "w") as outfile:
            json.dump(tmp_pns, outfile, indent=4)

    def load(self, id):
        """
        Load the points indices and the neighbours of the superpoints according
        to a point cloud with a certain id.

        Parameters
        ----------
        id : int
            Index of a scene or point cloud.
        """
        self._pns_orig = [list(), list()]
        with open("./cache/"+id+"/pns_orig.json") as infile:
            tmp_pns = json.load(infile)
            for i in range(len(tmp_pns[0])):
                self._pns_orig[0].append(np.array(tmp_pns[0][i], np.int32))
                self._pns_orig[1].append(np.array(tmp_pns[1][i], np.int32))
        self.init()

    def reset(self):
        """
        Reassign values from orig arrays. This is helpful if the storage is
        used multiple times.
        """
        # copy every object of the orig list
        self._pns = [list(), list()]
        for i in range(len(self._pns_orig[0])):
            self._pns[0].append(
                np.array(self._pns_orig[0][i], copy=True))
            self._pns[1].append(
                np.array(self._pns_orig[1][i], copy=True))
        if self._sort_mode:
            self._lengths = np.array(self._orig_lengths, copy=True)
            self._sorted_idxs = np.array(self._orig_sorted_idxs, copy=True)

        # delete the blacklists and recreate them
        del self._blacklist[:]
        n_superpoints = len(self._pns_orig[0])
        for i in range(n_superpoints):
            self._blacklist.append([i])

    def get_sorted_idx(self, idx):
        """Get the sorted idx of a superpoint. See method _sort for more
        information.

        Parameters
        ----------
        idx : int
            Index of a superpoint or the i-th superpoint.

        Returns
        -------
        int
             Sorted idx of a superpoint.
        """
        return self._sorted_idxs[idx]

    def get_sp_point_idxs(self, idx):
        """
        Get the point cloud idxs of a superpoint with index idx.

        Parameters
        ----------
        idx : int
            Index of a superpoint or the i-th superpoint.

        Returns
        -------
        np.ndarray
            Point indices of a superpoint.
        """
        if idx >= len(self._pns[0]):
            return np.zeros((0, ), np.int32)
        P_idxs = self._pns[0][idx]
        return P_idxs

    def get_neighbours_of_superpoint(self, idx):
        """
        Get the neighbours (e.g. superpoints with original idxs such
        as 150, 12, ...) of the idx-th superpoint. The neighbour values are
        transformed to their actual sorted idx position of self._sorted_idxs
        in sort_mode. Additionally, the blacklisted connections are filtered.

        Parameters
        ----------
        idx : int
            Index of a superpoint or the i-th superpoint.

        Returns
        -------
        np.ndarray
            Neighbours of a superpoint where the blacklisted connections are
            filtered.
        """
        if idx >= len(self._pns[1]):
            return np.zeros((0,), np.int32)
        neighbours = self._pns[1][idx]

        # copy neighbours
        neighbours_cp = np.array(neighbours, copy=True)
        # filter blacklisted connections
        idxs_to_del = []
        for number in self._blacklist[idx]:
            idxs_to_del.append(np.where(neighbours_cp == number)[0])
        idxs_to_del = np.concatenate(idxs_to_del)
        idxs_to_del = np.unique(idxs_to_del)
        neighbours_cp = np.delete(neighbours_cp, idxs_to_del)

        return neighbours_cp

    def get_n_superpoints(self):
        """
        Returns the current number of superpoints.

        Returns
        -------
        int
            Current number of superpoints.
        """
        return len(self._pns[0])

    def _replace(self, old_idx, new_idx):
        """
        Replaces the superpoint with old_idx and keep all other idxs
        up to date by replacing with new_idx where the old_idx was.
        Higher idxs in comparison to old_idx will be decremented as
        one element is deleted.

        Parameters
        ----------
        old_idx : int
            Index of a superpoint that the will be replaced by a superpoint
            with index new_idx.
        new_idx : int
            Index of a superpoint that replaces a superpoint with the index
            old_idx.
        """
        if self._sort_mode:
            if old_idx >= self._lengths.shape[0]:
                return
            self._lengths = np.delete(self._lengths, old_idx)
            self._sorted_idxs = np.delete(self._sorted_idxs, old_idx)
            del self._pns[0][old_idx]
            del self._pns[1][old_idx]
            del self._blacklist[old_idx]
            if new_idx > old_idx:
                new_idx -= 1
        # update all neighbours with the union
        n_superpoints = self.get_n_superpoints()
        for i in range(n_superpoints):
            n_idxs = self._pns[1][i]
            if self._sort_mode:
                idxs_to_change = np.where(n_idxs > old_idx)[0]
                n_idxs[idxs_to_change] -= 1
            # replace old idx with new idx
            idxs_to_change = np.where(n_idxs == old_idx)[0]
            n_idxs[idxs_to_change] = new_idx
            # filter multiple values of a superpoint idx
            n_idxs = np.unique(n_idxs)
            self._pns[1][i] = n_idxs

            # change blacklist of each superpoint
            for j in range(len(self._blacklist[i])):
                superpoint = self._blacklist[i][j]
                # replace old idx with new idx
                if self._sort_mode:
                    if superpoint > old_idx:
                        self._blacklist[i][j] -= 1
                if superpoint == old_idx:
                    self._blacklist[i][j] = new_idx

    def unify(self, superpoint_idx_a, superpoint_idx_b):
        """
        The superpoints with idxs superpoint_idx_a and superpoint_idx_b will be
        unified. The resulting superpoint has the idx superpoint_idx_a.

        Parameters
        ----------
        superpoint_idx_a : int
            Index of superpoint a.
        superpoint_idx_b : int
            Index of superpoint b.
        """
        # unify neighbours and point idxs
        neighbours_b = self.get_neighbours_of_superpoint(superpoint_idx_b)
        neighbours_a = self.get_neighbours_of_superpoint(superpoint_idx_a)

        P_idxs_a = self.get_sp_point_idxs(superpoint_idx_a)
        P_idxs_b = self.get_sp_point_idxs(superpoint_idx_b)

        P_idxs_a = np.vstack(
            (P_idxs_a[:, None], P_idxs_b[:, None]))
        P_idxs_a = P_idxs_a.reshape(-1)

        neighbours_a =\
            np.vstack((neighbours_a[:, None], neighbours_b[:, None]))
        neighbours_a = np.unique(neighbours_a)

        # unify blacklists to not consider already blocked connections
        self._blacklist[superpoint_idx_a].extend(self._blacklist[superpoint_idx_b])

        # apply changes in _pns_orig
        self._pns[0][superpoint_idx_a] = P_idxs_a
        self._pns[1][superpoint_idx_a] = neighbours_a
        if self._sort_mode:
            self._lengths[superpoint_idx_a] = P_idxs_a.shape[0]
        self._replace(old_idx=superpoint_idx_b, new_idx=superpoint_idx_a)
        if self._sort_mode:
            self._sort_superpoints()

    def break_connection(self, superpoint_idx_a, superpoint_idx_b):
        """
        Breaks the neighbourhood connection between superpoint_a and
        superpoint_b.

        Parameters
        ----------
        superpoint_idx_a : int
            Index of superpoint a.
        superpoint_idx_b : int
            Index of superpoint b.
        """
        len_b = len(self._blacklist)
        if superpoint_idx_b < len_b:
            self._blacklist[superpoint_idx_b].append(superpoint_idx_a)
        if superpoint_idx_a < len_b:
            self._blacklist[superpoint_idx_a].append(superpoint_idx_b)

    def break_all_connections(self, superpoint_idx):
        """
        Breaks every neighbourhood connection to a superpoint. This method is
        usually called if a certain segment should is deleted, e.g., when two
        superpoints are unified.

        Parameters
        ----------
        superpoint_idx : int
            Description of parameter `superpoint_idx`.
        """
        n_superpoints = len(self._pns_orig[0])
        for i in range(n_superpoints):
            self._blacklist[i].append(superpoint_idx)

    def _sort_superpoints(self):
        """
        Get the sorted pns_orig indices according to the size of a superpoint.
        This method will be called if superpoints will be unified.
        """
        # self._lengths is a copy of the
        self._sorted_idxs = np.argsort(self._lengths)

    def _sort(self):
        """
        Get the sorted pns_orig indices according to the size of a superpoint.
        Can be helpful to query points in an ascending order. The neighbours
        will also be sorted in ascending order according to the size of a
        neighbour superpoint.
        This is method will be called once in the initialization process of
        this class.
        """
        n_superpoints = len(self._pns_orig[0])
        self._orig_lengths = np.zeros((n_superpoints, ), dtype=np.int32)
        for i in range(n_superpoints):
            self._orig_lengths[i] = self._pns_orig[0][i].shape[0]
        # store the order of the sorted superpoints
        self._orig_sorted_idxs = np.argsort(self._orig_lengths)
        # sort the neighbours
        for i in range(n_superpoints):
            neighbours = self._pns_orig[1][i]
            lengths = self._orig_lengths[neighbours]
            sortation = np.argsort(lengths)
            neighbours = neighbours[sortation]
            self._pns_orig[1][i] = neighbours
