import numpy as np
from .base_environment import BaseEnvironment
from .segmentation_ext import PCLCloud, vccs, get_plane_segment
from collections import deque
from .utils import\
    generate_colors,\
    render_point_cloud,\
    render_point_cloud4,\
    get_interval
from .scene import Scene


class SuperpointGrowingEnv(BaseEnvironment):

    def __init__(
            self,
            data_prov_type,
            max_scenes,
            train_p=0.8,
            voxel_r=0.1,
            seed_r=1,
            color_i=0.75,
            normal_i=0.75,
            spatial_i=0.0,
            min_superpoint_size=20,
            object_punishment_factor=0,
            object_punishment_exp=1,
            psi_scale=1,
            batch_id=-1,
            n_cpus=None,
            train_mode=True):
        """Constructor.

        Parameters
        ----------
        data_prov_type : BaseDataProvider
            Type of class BaseDataProvider (see base_data_provider.py for
            more information).
        max_scenes : int
            Number of scenes/point clouds that should be used.
        train_p : float
            Specify, how many percent of the scenes/point clouds should be used
            for training of an agent.
        voxel_r : float
            Voxel resolution of the VCCS algorithm.
        seed_r : float
            Seed resolution of the VCCS algorithm.
        color_i : float
            Color importance of the VCCS algorithm.
        normal_i : float
            Normal importance of the VCCS algorithm.
        spatial_i : float
            Spatial importance of the VCCS algorithm.
        min_superpoint_size : int
            Minimum size of a superpoint. Smaller superpoints want be stored
            in the scene.
        object_punishment_factor : float
            If this factor is >= 0, then the reward will reduced. The reduction
            takes the mismatch of segmented objects into account.
        object_punishment_exp : float
            Value for the exponent to in- or decrease the object punishment.
        psi_scale : float
            Scale factor for the degree of imitation. The values 0 or 1 will be
            returned from this environment with a factor of 1. This could be
            changed to the values of 0 and .5 with a scale factor of 0.5.
        batch_id : int
            Only use a certain batch with batch_id. The batch size is equal to
            the number of cpus.
        n_cpus : int
            Number of cpus that will be used for the training.
        train_mode : boolean
            If True, scenes/point clouds will be splitted into training and
            test scenes.
        """
        super().__init__()
        if id != -1:
            self._data_prov = data_prov_type(
                max_scenes=max_scenes,
                train_mode=train_mode,
                train_p=train_p,
                batch_id=batch_id,
                n_cpus=n_cpus)
        else:
            self._data_prov = data_prov_type(
                max_scenes=max_scenes, train_mode=train_mode, train_p=train_p)
        self.voxel_r = voxel_r
        self.seed_r = seed_r
        self.color_i = color_i
        self.normal_i = normal_i
        self.spatial_i = spatial_i
        self._unsegmented = 0
        self._min_superpoint_size = min_superpoint_size
        self._object_punishment_factor = object_punishment_factor
        self._object_punishment_exp = object_punishment_exp
        self._psi_scale = psi_scale
        self._id_to_scene = {}
        print(locals())

    def _update_segment(self, superpoint_idx):
        """
        Update the assigned segments. Concretely, assign the segment number of
        a superpoint to the points of that superpoint.

        Parameters
        ----------
        superpoint_idx : int
            Index of a specific superpoint.
        """
        # get the point idxs of the superpoint
        P_idxs = self._scene.superpoint_storage.get_sp_point_idxs(
            superpoint_idx)
        if P_idxs.shape[0] == 0:
            return
        # assign segment value to assigned segments
        segment = self._superpoint_segments[superpoint_idx]
        self._assigned_segments[P_idxs] = segment

    def _compute_punishment(self):
        """
        Compute a triangle where the top has the value of 1 at the nr of
        original segments (ground line). Will be zero at
        min_assignable_segments and max_assignable_segments (lower vertices of
        the triangle). The upper edges of the triangle are bended with the
        object_punishment_exp variable.

        Returns
        -------
        float
            Description of returned object.

        """
        n_orig_segments = self._scene.orig_segment_nrs.shape[0]
        max_assignable_segments = self._scene.superpoint_coverage.shape[1]
        min_assignable_segments = 2
        x = self._n_assigned_segments
        # compute triangle edges
        if self._n_assigned_segments > n_orig_segments:
            m = 1 / (n_orig_segments - max_assignable_segments)
            b = -m * max_assignable_segments
        else:
            m = 1 / (n_orig_segments - min_assignable_segments)
            b = -m * min_assignable_segments
        # compute position on triangle
        y = x * m + b
        # bend edges
        y = pow(y, self._object_punishment_exp)
        return y

    def _neighbourhood_assignment(
            self,
            superpoint_idx,
            orig_segment,
            orig_segment_length,
            visited_neighbours):
        """
        Apply depth breath search to approximate the potential assignment
        of the unsegmented neighbourhood of a superpoint.

        Parameters
        ----------
        superpoint_idx : int
            Idx of an unsegmented superpoint.
        orig_segment : int
            Value of original segment.
        orig_segment_length : int
            Length (nr of points) of the orig_segment.
        visited_neighbours : np.ndarray
            Idxs of superpoints that already have been seen.

        Returns
        -------
        float
            The possible relative assignment.

        """
        rel_assignment = 0
        if superpoint_idx in visited_neighbours:
            return rel_assignment
        # prevent to consider superpoint_idx multiple times
        visited_neighbours.append(superpoint_idx)
        # get the neighbours
        neighbours = self._scene.superpoint_storage.get_neighbours_of_superpoint(
            superpoint_idx)
        # if there is no neighbour
        if neighbours.shape[0] == 0:
            return rel_assignment
        # consider neighbours
        for neighbour_idx in neighbours:
            # neighbour should be unsegmented
            neighbour_superpoint = self._superpoint_segments[neighbour_idx]
            if neighbour_superpoint != self._unsegmented:
                continue
            # and not considered so far
            if neighbour_idx in visited_neighbours:
                continue
            visited_neighbours.append(neighbour_idx)
            # which orig segments are crossed by the neighbour_idx
            orig_segments = self._superpoint_to_segment_nrs[neighbour_idx]
            # if neighbour has nothing to do with orig_segment
            if orig_segment not in orig_segments:
                continue
            # determine the relative assignment
            k = self._scene.orig_segment_to_idx[orig_segment]
            rel_assignment += (self._scene.superpoint_coverage[(
                k, neighbour_idx)] / orig_segment_length)
            # consider neighbours of neighbour
            rel_assignment += self._neighbourhood_assignment(
                neighbour_idx,
                orig_segment,
                orig_segment_length,
                visited_neighbours)
        return rel_assignment

    def _rel_unsegmented_assignment(self, orig_segment, orig_segment_length):
        """
        Approximate the maximum possible assignment of the unsegmented
        superpoints within an orig_segment.

        Parameters
        ----------
        orig_segment : int
            Number of the original segment.
        orig_segment_length : int
            Number of points of the original segment.

        Returns
        -------
        float
            The maximum possible relative assignment.

        """
        # potential maximum assignment
        max_rel_assignment = 0
        # idx of the original segment
        k = self._scene.orig_segment_to_idx[orig_segment]
        # superpoints within orig_segment
        superpoint_list = self._segment_to_superpoint[orig_segment]
        for superpoint_idx in superpoint_list:
            # temporary assignment of that superpoint
            rel_assignment = 0
            if superpoint_idx in self._unified_idxs:
                continue
            # value of the superpoint with idx superpoint_idx
            superpoint = self._superpoint_segments[superpoint_idx]
            # value should be unsegmented
            if superpoint != self._unsegmented:
                continue
            # abs coverage / orig segment length
            rel_assignment = (self._scene.superpoint_coverage[(
                k, superpoint_idx)] / orig_segment_length)
            """
            structure to prevent double consideration in case of
            bidirectional connections
            """
            visited_neighbours = []
            # determine the assignment of the neighbour/s of the superpoint
            n_assignment = self._neighbourhood_assignment(
                superpoint_idx,
                orig_segment,
                orig_segment_length,
                visited_neighbours)
            # assert n_assignment is not None
            rel_assignment += n_assignment
            if rel_assignment > max_rel_assignment:
                max_rel_assignment = rel_assignment

        return max_rel_assignment

    def _get_n_main(self, orig_segment, assigned_segment_idx):
        """
        Returns the number of points that are assigned with the
        assigned_segment within the interval of the orig_segment.

        Parameters
        ----------
        orig_segment : int
            Number of an original segment.
        assigned_segment_idx : int
            Index of an assigned segment.

        Returns
        -------
        int
            Number of points that are in the assigned segment.
        int
            Length of the original segment.

        """
        idx = self._scene.orig_segment_to_idx[orig_segment]
        # start = self._scene.orig_indices[idx]
        # length = self._scene.orig_segment_counts[idx]
        # stop = start + length
        _, _, length = get_interval(
            idx,
            self._scene.orig_indices,
            self._scene.orig_segment_counts)

        # _assigned_segments does also include unsegmented points
        n_main = self._current_cov[idx, assigned_segment_idx]
        return n_main, length

    def _compute_reward(self, assigned_segment_idx):
        """
        The assigned segment with idx assigned_segment_idx could lie in
        multiple true orig segments. The value of the assigned_segment_idx
        can only be assigned to one orig_segment.

        Parameters
        ----------
        assigned_segment_idx : int
            Index of the superpoint of the assigned segment.

        Returns
        -------
        float
            Reward of the assignment.
        boolean
            Flag that indicates if the reward could be computed.
        """
        reward = 0
        # if superpoint was unified to another subseg
        if assigned_segment_idx in self._unified_idxs:
            return reward, True
        # get the assigned segment value
        assigned_segment = self._superpoint_segments[assigned_segment_idx]
        if assigned_segment in self._assignment.values():
            return reward, True
        # update the assigned segments vector
        self._update_segment(assigned_segment_idx)
        # orig values of superpoint
        segment_vals = self._superpoint_to_segment_nrs[assigned_segment_idx]
        # assert assigned_segment != self._unsegmented

        """
        key: assigned segment, value: reward
        orig_segments of the structure are unassigned and assigned segments
        of that structure are not rewarded - potential rewards are calculated
        """
        segment_to_reward = {}
        for orig_segment in segment_vals:
            # if orig segment is already assigned
            if orig_segment in self._assignment:
                continue
            """
            how many points are segmented with the assigned segment in
            context of the original segment
            """
            n_main, length =\
                self._get_n_main(orig_segment, assigned_segment_idx)
            # length of the orig segment
            rel_assignment = n_main / length

            """
            determine the assignments of the other assigned segment values
            filter unsegmented and assigned_segment from the assigned
            segment values
            """
            subsegs = self._segment_to_superpoint[orig_segment]
            only_assigned_idxs = []
            abs_counts = []
            for seg_idx in subsegs:
                if seg_idx in self._unified_idxs:
                    continue
                if seg_idx == assigned_segment_idx:
                    continue
                subseg_val = self._superpoint_segments[seg_idx]
                if subseg_val == self._unsegmented:
                    continue
                if subseg_val in self._assignment.values():
                    continue
                only_assigned_idxs.append(seg_idx)
                k = self._scene.orig_segment_to_idx[orig_segment]
                abs_counts.append(self._current_cov[k, seg_idx])
            # if there are other segment values to consider
            if len(only_assigned_idxs) > 0:
                # rel assignment of each other assigned segment value
                abs_counts = np.array(abs_counts)
                rel_counts = abs_counts / length
                """
                is a assignment of the other assigned segment values greater
                than the considered assigned segment value?
                """
                rel_idxs = np.where(rel_counts > rel_assignment)[0]
                if rel_idxs.shape[0] > 0:
                    continue
                    # return reward, False
            # determine the potential assignment of the unsegmented subsegs
            max_rel_unsegmented =\
                self._rel_unsegmented_assignment(orig_segment, length)
            # assert max_rel_unsegmented <= 1

            # if considered assigned segment max?
            pot_reward = n_main / self._scene.n_P
            if rel_assignment < max_rel_unsegmented:
                """
                no - check if there is another assignment with higher
                pot reward
                """
                better_assignment_available = False
                for or_seg, pr_l in segment_to_reward.items():
                    pr, le = pr_l[0], pr_l[1]
                    """ uncomment for a reward based assignment
                    if pr >= pot_reward:
                        better_assignment_available = True
                        break
                    """
                    if le >= length:
                        better_assignment_available = True
                        break
                if not better_assignment_available:
                    return reward, False
                # better assignment available
                continue
            # yes
            # orig segment could be potentially assigned with assigned segment
            segment_to_reward[orig_segment] = (pot_reward, length)
        """
        all items of segment_to_reward have the relative majority - assign
        the orig_segment which gives the most reward ->> realy?
        In Tiator2020 largest orig segment gets the assignment regardless
        of the reward
        TODO
        """
        max_pot_reward = -1
        max_or_seg = -1
        max_length = -1
        for or_seg, pr_l in segment_to_reward.items():
            pr = pr_l[0]
            length = pr_l[1]
            """ uncomment for reward based assignment
            if pr > max_pot_reward:
                max_pot_reward = pr
                max_or_seg = or_seg
                max_length = length
            """
            # what if equality? - order matters!
            if length > max_length:
                max_length = length
                max_pot_reward = pr
                max_or_seg = or_seg
        """
        in case of other segments are all larger than assigned segment
        """
        if max_pot_reward == -1:
            return reward, False
        reward = max_pot_reward
        self._assignment[max_or_seg] = assigned_segment
        # print("assign:", max_or_seg, assigned_segment, reward)
        # tag assigned segment as rewarded
        return reward, True

    def _next_neighbour(self):
        """ Sets the next neighbour as union candidate. """
        self._neighbour_segment_idx = self._neighbours_to_visit.pop()

    def _next_superpoint(self):
        """Select next the main superpoint and its neighbours."""
        self._neighbours_to_visit = deque()
        # no more work - done
        if len(self._to_do) == 0:
            self._setup(action=0, last_one=True)
            return

        # get main segment and the neighbours
        self._main_segment_idx = self._to_do.popleft()
        # print(self._main_segment_idx)
        self._to_evaluate.append(self._main_segment_idx)
        self._current_object += 1
        self._n_assigned_segments += 1
        self._superpoint_segments[self._main_segment_idx] = self._current_object
        neighbour_idxs = self._scene.superpoint_storage.get_neighbours_of_superpoint(
            self._main_segment_idx)
        for idx in neighbour_idxs:
            self._neighbours_to_visit.append(idx)

        # if no neighbours are available
        if len(self._neighbours_to_visit) == 0:
            last_one = len(self._to_do) == 0
            self._setup(action=0, last_one=last_one)
            return
        # set next merging candidate
        self._next_neighbour()

    def _evaluate(self):
        """
        Try to compute reward for all assigned segments in _to_evaluate.
        TODO is it better to consider the superpoints in a FIFO or LIFO manner?
        """
        # for i in reversed(range(len(self._to_evaluate))):
        idx_to_del = []
        for i in range(len(self._to_evaluate)):
            segment_idx = self._to_evaluate[i]
            reward, success = self._compute_reward(segment_idx)
            if success:
                self._reward += reward
                # del self._to_evaluate[i]
                idx_to_del.append(i)
        for idx in reversed(idx_to_del):
            del self._to_evaluate[idx]

    def _setup(self, action, last_one=False):
        """Prepare to segment the next object.

        Parameters
        ----------
        action : int
            Should the superpoints be unified?
        last_one : boolean
            Last action in the environment.
        """
        if last_one:
            self._done = True
        else:
            if len(self._neighbours_to_visit) == 0:
                self._evaluate()
                self._next_superpoint()
            else:
                self._next_neighbour()

    def _unify_superpoints(self, action):
        """Union of neighbour with main segment.

        Parameters
        ----------
        action : int
            Should the superpoints be unified?
        """
        if action == 1:
            self._scene.superpoint_storage.unify(
                self._main_segment_idx,
                self._neighbour_segment_idx)

            # add neighbour to list of already considered neighbours
            # if self._neighbour_segment_idx not in self._unified_idxs:
            self._unified_idxs.append(self._neighbour_segment_idx)

            # remove that neighbour from the todo list
            self._to_do.remove(self._neighbour_segment_idx)

            """
            update the _neighbours_to_visit list
            already considered neighbours will not be in this array
            """
            neighbours = self._scene.superpoint_storage.get_neighbours_of_superpoint(
                self._main_segment_idx)
            self._neighbours_to_visit.clear()
            for idx in neighbours:
                self._neighbours_to_visit.append(idx)

            """
            change superpoint orig_segment relationship according to the
            unify - main superpoint is now in orig segments of neighbour
            """
            main_orig_segments =\
                self._superpoint_to_segment_nrs[self._main_segment_idx]
            n_orig_segments =\
                self._superpoint_to_segment_nrs[self._neighbour_segment_idx]
            # concat original segments values
            main_orig_segments = np.concatenate(
                (main_orig_segments, n_orig_segments))
            # filter multiple values
            main_orig_segments = np.unique(main_orig_segments)
            # reassign the relationship
            self._superpoint_to_segment_nrs[self._main_segment_idx] =\
                main_orig_segments

            # self._update_segment(self._main_segment_idx)
            for i in range(len(self._to_evaluate)):
                segment_idx = self._to_evaluate[i]
                if segment_idx == self._neighbour_segment_idx:
                    self._to_evaluate[i] = self._main_segment_idx

            """
            update the information from orig segment values to superpoints
            update superpoint coverage to prevent np.where's in reward
            calculation
            """
            for orig_segment in main_orig_segments:
                k = self._scene.orig_segment_to_idx[orig_segment]
                self._current_cov[k, self._main_segment_idx] +=\
                    self._scene.superpoint_coverage[k, self._neighbour_segment_idx]
                subsegs = self._segment_to_superpoint[orig_segment]
                """
                prevent same value  multiple times - check if the main
                superpoint is available in the superpoints
                """
                main_available = self._main_segment_idx in subsegs
                for i in reversed(range(len(subsegs))):
                    if subsegs[i] == self._neighbour_segment_idx:
                        if main_available:
                            # subsegs.remove(self._neighbour_segment_idx)
                            del subsegs[i]
                        else:
                            subsegs[i] = self._main_segment_idx
        else:
            self._scene.superpoint_storage.break_connection(
                self._main_segment_idx, self._neighbour_segment_idx)

    def _psi(self, action):
        """ Expert function.

        Parameters
        ----------
        action : int
            Should the superpoints be unified?
        """
        """ Returns the psi value """
        beta = np.argmax(self._current_cov[:, self._main_segment_idx])
        alpha = np.argmax(
            self._scene.superpoint_coverage[:, self._neighbour_segment_idx])
        # print("alpha:", alpha, "beta:", beta, "action", action)
        """
        print(
            "alpha:",
            self._current_cov[:, self._main_segment_idx].shape,
            "beta:",
            self._scene.superpoint_coverage[:, self._neighbour_segment_idx].shape)
        """
        if action == 1:
            if beta == alpha:
                return 1
            return -1
        else:
            if beta == alpha:
                return -1
            return 1

    def step(self, action):
        """Step in the environment.

        Parameters
        ----------
        action : int
            Should the superpoints be unified?

        Returns
        -------
        tuple
            State of the environment.
        float
            Degree of the expert imitation in terms of the psi function.
        boolean
            Is the episode done?
        dictionary
            Auxiliary information about the environment (neighbours to visit,
            reward).
        """
        self._reward = 0
        psi = self._psi(action) * self._psi_scale
        self._unify_superpoints(action)
        # reward will be calculated here
        self._setup(action)
        if self._done:
            self._evaluate()
            # punish for incorrect number of segments
            object_punishment = self._compute_punishment()
            object_punishment = 1 - object_punishment
            object_punishment *= self._object_punishment_factor
            self._reward -= object_punishment
            state = (
                self._scene.P,
                np.zeros((0, 1), dtype=(np.int32)),
                np.zeros((0, 1), dtype=(np.int32)),
                self._assigned_segments)
        else:
            state = self.next_state()
        info = {
            "neighbours_to_visit": self._neighbours_to_visit,
            "reward": self._reward}
        return state, psi, self._done, info

    def render(self, r_segments=False, animate=False):
        """Rendering of the point cloud.

        Parameters
        ----------
        r_segments : boolean
            If True, the segments of the environment will be rendered.
        """
        P = self._scene.P
        if r_segments:
            uni_segments = np.unique(self._superpoint_segments)
            n_segments = uni_segments.shape[0]
            print("n segments:", n_segments)
            colors = generate_colors(max_colors=(n_segments + 10))
            render_point_cloud(
                P=(P[:, :3]),
                segments=(self._assigned_segments),
                colors=colors,
                animate=animate)
        else:
            render_point_cloud4(
                P=P,
                P_idxs=(self._P_idxs),
                neighbour_idxs=(self._neighbour_idxs))

    def _reset(self):
        """
        Reset all containers that will be change with steps in the environment.
        """
        """
        assignment from superpoints to objects
        (e.g. 3 sub segments can be 1 object)
        the idx of each object can be associated with a sub segment
        """
        self._scene.superpoint_storage.reset()
        n_superpoints = self._scene.superpoint_coverage.shape[1]
        self._superpoint_segments =\
            self._unsegmented * np.ones(n_superpoints, np.int32)
        # coverage of the unified superpoints (isles)
        self._current_cov = np.array(self._scene.superpoint_coverage, copy=True)
        self._done = False
        self._assigned_segments = np.array(
            self._scene.assigned_segments, copy=True)
        self._assigned_segments[:] = self._unsegmented

        # 1 for the plane
        self._n_assigned_segments = 1
        self._to_do = deque()
        # copy orig todo list
        self._to_do.extend(self._scene.orig_to_do)
        self._neighbours_to_visit = deque()
        self._P_idxs = None
        self._neighbour_idxs = None
        self._main_segment_idx = 0
        self._current_object = self._unsegmented + 2
        self._neighbour_segment_idx = 0
        self._to_evaluate = []
        self._assignment = {}
        # idxs of neighbour superpoints that are unified to a main superpoints
        self._unified_idxs = []
        # assigned segment values that are already rewarded
        # copy structures to change them during runtime
        self._superpoint_to_segment_nrs = []
        for i in range(len(self._scene.orig_superpoint_to_segment_nr)):
            arr = self._scene.orig_superpoint_to_segment_nr[i]
            # copy array
            self._superpoint_to_segment_nrs.append(np.array(arr, copy=True))
        self._segment_to_superpoint = {}
        for k, v in self._scene.orig_segment_nr_to_superpoint.items():
            # copy list
            self._segment_to_superpoint[k] = v.copy()

    def reset(self, train=True):
        """Reset the whole environment.

        Parameters
        ----------
        train : boolean
            Will the agent train with the environment?

        Returns
        -------
        tuple
            State of the environment.

        """
        id = self._data_prov.select_id(train=train)
        if id in self._id_to_scene:
            self._scene = self._id_to_scene[id]
        else:
            self._scene = Scene(
                id=id,
                get_cloud_and_segments=self._data_prov.get_cloud_and_segments,
                voxel_r=self.voxel_r,
                seed_r=self.seed_r,
                color_i=self.color_i,
                normal_i=self.normal_i,
                spatial_i=self.spatial_i,
                min_superpoint_size=self._min_superpoint_size
                )
            if self._scene.error:
                blacklist = open("blacklist.txt", "a")
                blacklist.write("\n")
                blacklist.write(id)
                blacklist.close()
                return self.reset()
            self._id_to_scene[id] = self._scene
        self._reset()
        self._next_superpoint()
        if self._done:
            return self.reset()
        return self.next_state()

    def next_state(self):
        """
        Returns the point cloud + cloud idxs of the main superpoint + idxs
        of a neighbour.
        """
        self._P_idxs =\
            self._scene.superpoint_storage.get_sp_point_idxs(
                self._main_segment_idx)
        self._neighbour_idxs =\
            self._scene.superpoint_storage.get_sp_point_idxs(
                self._neighbour_segment_idx)
        return (
         self._scene.P,
         self._P_idxs,
         self._neighbour_idxs,
         self._assigned_segments)

    def close(self):
        pass
