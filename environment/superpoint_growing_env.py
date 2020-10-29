import numpy as np
from .base_environment import BaseEnvironment
from .segmentation_ext import PCLCloud, vccs, get_plane_segment
from collections import deque
from .utils import\
    generate_colors,\
    render_point_cloud,\
    render_point_cloud4,\
    mkdir,\
    file_exists
from .subsegment_storage import SubsegmentStorage
import json


class SuperpointGrowingEnv(BaseEnvironment):

    def __init__(
            self,
            get_cloud_and_segments,
            voxel_r=0.1,
            seed_r=1,
            color_i=0.75,
            normal_i=0.75,
            spatial_i=0.0,
            subsegment_thres=20,
            object_bonus_factor=0.1,
            object_bonus_exp=1):
        super().__init__()
        self._get_cloud_and_segments = get_cloud_and_segments
        self.voxel_r = voxel_r
        self.seed_r = seed_r
        self.color_i = color_i
        self.normal_i = normal_i
        self.spatial_i = spatial_i
        self._unsegmented = 0
        self._subsegment_thres = subsegment_thres
        self._init = False
        self._object_bonus_factor = object_bonus_factor
        self._object_bonus_exp = object_bonus_exp
        print(locals())

    def _update_segment(self, subsegment_idx):
        # get the point idxs of the subsegment
        P_idxs = self._subsegment_storage.get_segment_point_idxs(
            subsegment_idx)
        if P_idxs.shape[0] == 0:
            return
        # assign segment value to assigned segments
        segment = self._subsegments[subsegment_idx]
        self._assigned_segments[P_idxs] = segment

    def _compute_object_bonus(self):
        """
        Compute a triangle with the top with value 1 at the nr of original
        segments. Will be zero at min_assignable_segments and
        max_assignable_segments. The edges of the triangle can be bended with
        the object_bonus_exp variable
        """
        # get necessary information
        n_orig_segments = self._orig_segment_values.shape[0]
        max_assignable_segments = self._subsegment_coverage.shape[1]
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
        y = pow(y, self._object_bonus_exp)
        return y

    def _neighbourhood_assignment(
            self,
            subsegment_idx,
            orig_segment,
            orig_segment_length,
            visited_neighbours):
        """
        Apply depth breath search to determine the potential assignment
        of the unsegmented neighbourhood.
        subsegment_idx:
            Idx of unsegmented subsegment
        orig_segment:
            Value of original segment
        orig_segment_length:
            Length (nr of points) of the orig_segment
        visited_neighbours:
            Idxs of subsegments that already have been seen

        Return the possible relative assignment of the unsegmented points.
        """
        rel_assignment = 0
        if subsegment_idx in visited_neighbours:
            return rel_assignment
        # prevent to consider subsegment_idx multiple times
        visited_neighbours.append(subsegment_idx)
        # get the neighbours
        neighbours = self._subsegment_storage.get_neighbours_of_segment(
            subsegment_idx)
        # if there is no neighbour
        if neighbours.shape[0] == 0:
            return rel_assignment
        # consider neighbours
        for neighbour_idx in neighbours:
            # neighbour should be unsegmented
            neighbour_subsegment = self._subsegments[neighbour_idx]
            if neighbour_subsegment != self._unsegmented:
                continue
            # and not considered so far
            if neighbour_idx in visited_neighbours:
                continue
            visited_neighbours.append(neighbour_idx)
            # which orig segments are crossed by the neighbour_idx
            orig_segments = self._subsegment_to_segment_vals[neighbour_idx]
            # if neighbour has nothing to do with orig_segment
            if orig_segment not in orig_segments:
                continue
            # determine the relative assignment
            k = self._orig_segment_to_idx[orig_segment]
            rel_assignment += (self._subsegment_coverage[(
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
        Determine the possible maximum assignment of the unsegmented
        subsegments within an orig_segment
        """
        # potential maximum assignment
        max_rel_assignment = 0
        # idx of the original segment
        k = self._orig_segment_to_idx[orig_segment]
        # subsegments within orig_segment
        subsegment_list = self._segment_to_subsegment[orig_segment]
        for subsegment_idx in subsegment_list:
            # temporary assignment of that subsegment
            rel_assignment = 0
            if subsegment_idx in self._merged_idxs:
                continue
            # value of the subsegment with idx subsegment_idx
            subsegment = self._subsegments[subsegment_idx]
            # value should be unsegmented
            if subsegment != self._unsegmented:
                continue
            # abs coverage / orig segment length
            rel_assignment = (self._subsegment_coverage[(
                k, subsegment_idx)] / orig_segment_length)
            """
            structure to prevent double consideration in case of
            bidirectional connections
            """
            visited_neighbours = []
            # determine the assignment of the neighbour/s of the subsegment
            n_assignment = self._neighbourhood_assignment(
                subsegment_idx,
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
        returns the number of points which are assigned with assigned_segment
        within the interval of the orig_segment
        """
        idx = self._orig_segment_to_idx[orig_segment]
        # start = self._orig_indices[idx]
        # length = self._orig_segment_counts[idx]
        # stop = start + length
        _, _, length = self._get_interval(idx)

        # _assigned_segments does also include unsegmented points
        n_main = self._current_cov[idx, assigned_segment_idx]
        return n_main, length

    def _compute_reward(self, assigned_segment_idx):
        """
        The assigned segment with idx assigned_segment_idx could lie in
        multiple true orig segments. The value of the assigned_segment_idx
        can only be assigned to one orig_segment.
        assigned_segment:
            value of the assigned segment
        assigned_segment_idx:
            idx of the subsegment of the assigned segment
        """
        reward = 0
        # if subsegment was merged to another subseg
        if assigned_segment_idx in self._merged_idxs:
            return reward, True
        # get the assigned segment value
        assigned_segment = self._subsegments[assigned_segment_idx]
        if assigned_segment in self._assignment.values():
            return reward, True
        # update the assigned segments vector
        self._update_segment(assigned_segment_idx)
        # orig values of subsegment
        segment_vals = self._subsegment_to_segment_vals[assigned_segment_idx]
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
            subsegs = self._segment_to_subsegment[orig_segment]
            only_assigned_idxs = []
            abs_counts = []
            for seg_idx in subsegs:
                if seg_idx in self._merged_idxs:
                    continue
                if seg_idx == assigned_segment_idx:
                    continue
                subseg_val = self._subsegments[seg_idx]
                if subseg_val == self._unsegmented:
                    continue
                if subseg_val in self._assignment.values():
                    continue
                only_assigned_idxs.append(seg_idx)
                k = self._orig_segment_to_idx[orig_segment]
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
            pot_reward = n_main / self._n_P
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
        """ Sets the next neighbour as merging candidate. """
        self._neighbour_segment_idx = self._neighbours_to_visit.pop()

    def _next_subsegment(self):
        """Select main object and neighbours."""
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
        self._subsegments[self._main_segment_idx] = self._current_object
        neighbour_idxs = self._subsegment_storage.get_neighbours_of_segment(
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
        try to compute reward for all assigned segments in _to_evaluate
        TODO is it better to consider the subsegments in a FIFO or LIFO manner?
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
        """Prepare for the next object."""
        if last_one:
            self._done = True
        else:
            if len(self._neighbours_to_visit) == 0:
                self._evaluate()
                self._next_subsegment()
            else:
                self._next_neighbour()

    def _merge_segment(self, action):
        """Merging of neighbour with main segment. """
        if action == 1:
            self._subsegment_storage.merge(
                self._main_segment_idx,
                self._neighbour_segment_idx)

            # add neighbour to list of already considered neighbours
            # if self._neighbour_segment_idx not in self._merged_idxs:
            self._merged_idxs.append(self._neighbour_segment_idx)

            # remove that neighbour from the todo list
            self._to_do.remove(self._neighbour_segment_idx)

            """
            update the _neighbours_to_visit list
            already considered neighbours will not be in this array
            """
            neighbours = self._subsegment_storage.get_neighbours_of_segment(
                self._main_segment_idx)
            self._neighbours_to_visit.clear()
            for idx in neighbours:
                self._neighbours_to_visit.append(idx)

            """
            change subsegment orig_segment relationship according to the
            merge - main subsegment is now in orig segments of neighbour
            """
            main_orig_segments =\
                self._subsegment_to_segment_vals[self._main_segment_idx]
            n_orig_segments =\
                self._subsegment_to_segment_vals[self._neighbour_segment_idx]
            # concat original segments values
            main_orig_segments = np.concatenate(
                (main_orig_segments, n_orig_segments))
            # filter multiple values
            main_orig_segments = np.unique(main_orig_segments)
            # reassign the relationship
            self._subsegment_to_segment_vals[self._main_segment_idx] =\
                main_orig_segments

            for i in range(len(self._to_evaluate)):
                segment_idx = self._to_evaluate[i]
                if segment_idx == self._neighbour_segment_idx:
                    self._to_evaluate[i] = self._main_segment_idx

            """
            update the information from orig segment values to subsegments
            update subsegment coverage to prevent np.where's in reward
            calculation
            """
            for orig_segment in main_orig_segments:
                k = self._orig_segment_to_idx[orig_segment]
                self._current_cov[k, self._main_segment_idx] +=\
                    self._subsegment_coverage[k, self._neighbour_segment_idx]
                subsegs = self._segment_to_subsegment[orig_segment]
                """
                prevent same value  multiple times - check if the main
                subsegment is available in the subsegments
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
            self._subsegment_storage.break_connection(
                self._main_segment_idx, self._neighbour_segment_idx)

    def step(self, action):
        self._reward = 0
        self._merge_segment(action)
        # reward will be calculated here
        self._setup(action)
        if self._done:
            # for all subsegments - check if reward is missing
            # self._subsegment_to_segment_vals[]
            self._evaluate()
            # print("remaining", self._to_evaluate)
            # print(self._assignment)
            # punish for incorrect number of segments
            object_bonus = self._compute_object_bonus()
            object_bonus = 1 - object_bonus
            object_bonus *= self._object_bonus_factor
            self._reward -= object_bonus
            state = (
                self._P,
                np.zeros((0, 1), dtype=(np.int32)),
                np.zeros((0, 1), dtype=(np.int32)),
                self._assigned_segments)
        else:
            state = self.next_state()
        info = {"neighbours_to_visit": self._neighbours_to_visit}
        return state, self._reward, self._done, info

    def render(self, r_segments=False):
        P = self._P
        if r_segments:
            uni_segments = np.unique(self._subsegments)
            n_segments = uni_segments.shape[0]
            print("n segments:", n_segments)
            colors = generate_colors(max_colors=(n_segments + 10))
            render_point_cloud(
                P=(P[:, :3]),
                segments=(self._assigned_segments),
                colors=colors)
        else:
            render_point_cloud4(
                P=P,
                P_idxs=(self._P_idxs),
                neighbour_idxs=(self._neighbour_idxs))

    def _reset(self):
        """
        assignment from sub segments to objects
        (e.g. 3 sub segments can be 1 object)
        the idx of each object can be associated with a sub segment
        """
        self._subsegment_storage.reset()
        self._done = False
        self._assigned_segments[:] = self._unsegmented
        # 1 for the plane
        self._n_assigned_segments = 1
        self._subsegments[:] = self._unsegmented
        self._to_do = deque()
        # copy orig todo list
        self._to_do.extend(self._orig_to_do)
        self._neighbours_to_visit = deque()
        self._P_idxs = None
        self._neighbour_idxs = None
        self._main_segment_idx = 0
        self._current_object = self._unsegmented + 2
        self._neighbour_segment_idx = 0
        self._to_evaluate = []
        self._assignment = {}
        self._current_cov[:, :] = self._subsegment_coverage[:, :]
        # idxs of neighbour subsegments that are merged to a main subsegments
        self._merged_idxs = []
        # assigned segment values that are already rewarded
        # copy structures to change them during runtime
        self._subsegment_to_segment_vals = []
        for i in range(len(self._orig_subsegment_to_segment_vals)):
            arr = self._orig_subsegment_to_segment_vals[i]
            # copy array
            self._subsegment_to_segment_vals.append(np.array(arr, copy=True))
        self._segment_to_subsegment = {}
        for k, v in self._orig_segment_to_subsegment.items():
            # copy list
            self._segment_to_subsegment[k] = v.copy()

    def _load_scene(self):
        self._P, orig_segments = self._get_cloud_and_segments()
        self._orig_segment_values,\
            self._orig_indices,\
            self._orig_segment_counts = np.unique(
                orig_segments, return_index=True, return_counts=True)
        self._n_P = self._P.shape[0]
        self._orig_pcl_cloud = PCLCloud(self._P)
        self._orig_pcl_cloud.BuildTree()
        print("Tree builded")
        self._assigned_segments = np.zeros((self._n_P,), dtype=(np.int32))
        max_iterations = 1000
        threshold = 0.06
        probability = 0.9
        plane_result = get_plane_segment(
            self._orig_pcl_cloud, max_iterations, threshold, probability)
        ok = plane_result[(-1)]
        if not ok:
            raise Exception('Could not segment plane')
        plane_idxs = plane_result[0]
        print("{0} points segmented in plane step".format(
            plane_idxs.shape[0]))
        self._assigned_segments[plane_idxs] = self._unsegmented + 1
        all_idxs = np.arange(self._n_P)
        remaining_idxs = np.delete(all_idxs, plane_idxs)
        P_ = self._P[remaining_idxs]
        P_cloud = PCLCloud(P_)
        vccs_result = vccs(
            P_cloud,
            self.voxel_r,
            self.seed_r,
            self.color_i,
            self.normal_i,
            self.spatial_i)
        n_segments = int(len(vccs_result) / 3)
        idxs_to_del = []
        segments_to_del = []
        del_sum = 0
        for i in range(n_segments):
            idx = i * 3
            segment = vccs_result[idx]
            P_idxs = vccs_result[(idx + 1)]
            points_ = P_[P_idxs]
            P_idxs = self._orig_pcl_cloud.Search(points_)
            vccs_result[idx + 1] = P_idxs
            if P_idxs.shape[0] < self._subsegment_thres:
                idxs_to_del.extend([idx, idx + 1, idx + 2])
                segments_to_del.append(segment)
                del_sum += 1
                continue

        print("{0}/{1} have been deleted from vccs result".format(
            del_sum, plane_idxs.shape[0]))
        segments_to_del = np.array(segments_to_del, dtype=(np.int32))
        for i in range(n_segments):
            n_idx = i * 3 + 2
            neighbours = vccs_result[n_idx]
            _, n_idxs, _ = np.intersect1d(
                neighbours,
                segments_to_del,
                return_indices=True)
            if n_idxs.shape[0] > 0:
                neighbours = np.delete(neighbours, n_idxs)
                vccs_result[n_idx] = neighbours

        print("delete {0} elements in vccs result structure".format(
            len(idxs_to_del)))
        for idx in reversed(idxs_to_del):
            del vccs_result[idx]

        n_segments = int(len(vccs_result) / 3)
        print("n_vccs_segments:", n_segments)
        vccsSegmentToIdx = {}
        for i in range(n_segments):
            segment = vccs_result[(i * 3)]
            vccsSegmentToIdx[segment] = i

        vccs_res = [list(), list()]
        for i in range(n_segments):
            neighbours = vccs_result[(i * 3 + 2)]
            for j in range(neighbours.shape[0]):
                neighbour = int(neighbours[j])
                idx = vccsSegmentToIdx[neighbour]
                neighbours[j] = idx
                # assert idx < n_segments

            vccs_result[i * 3 + 2] = neighbours
            vccs_res[0].append(vccs_result[(i * 3 + 1)])
            vccs_res[1].append(vccs_result[(i * 3 + 2)])

        self._subsegment_storage = SubsegmentStorage(pns_orig=vccs_res)
        self._subsegment_storage.reset()
        # original containers that will ne change
        self._orig_subsegment_to_segment_vals = []
        self._orig_segment_to_subsegment = {}
        # for all original segment values
        for orig_segment in self._orig_segment_values:
            # add a list
            self._orig_segment_to_subsegment[int(orig_segment)] = list()

        # for all subsegments
        for i in range(n_segments):
            # get the cloud idxs of the subsegment
            P_idxs = self._subsegment_storage.get_segment_point_idxs(i)
            # get the original segment values of the subsegment idxs
            orig_segment_values = orig_segments[P_idxs]
            # which original values are there?
            uni_orig_segments, counts = np.unique(
                orig_segment_values,
                return_counts=True)
            """
            sort the occurences of the orig segment values in
            descending order (begin by the biggest values)
            """
            idxs = np.argsort(counts)[::-1]
            uni_orig_segments = uni_orig_segments[idxs]
            # append the subsegment to the lists of the orig segments
            for orig_segment in uni_orig_segments:
                self._orig_segment_to_subsegment[orig_segment].append(i)
            # this subsegment lies in uni_orig_segments
            self._orig_subsegment_to_segment_vals.append(uni_orig_segments)

        # nr of orig segments
        n_orig_segments = self._orig_segment_values.shape[0]
        # how much points of a orig_segment are covered by a subsegment
        self._subsegment_coverage = np.zeros((n_orig_segments, n_segments))
        """
        the orig segment values will not start by zero
        structure to convert segment value to an idx
        """
        self._orig_segment_to_idx = {}
        # for all orig values
        for i in range(self._orig_segment_values.shape[0]):
            orig_segment = self._orig_segment_values[i]
            # save the idx of the orig segment value
            self._orig_segment_to_idx[int(orig_segment)] = i
            # get the interval of the orig_segment
            # start = self._orig_indices[i]
            # length = self._orig_segment_counts[i]
            # stop = start + length
            start, stop, length = self._get_interval(i)
            orig_idxs = np.arange(start, stop)
            # get all subsegments that are in orig_segment
            subsegment_list =\
                self._orig_segment_to_subsegment[orig_segment]
            # for all subsegments
            for subsegment_idx in subsegment_list:
                # get a list of points for all subsegments
                P_idxs = self._subsegment_storage.get_segment_point_idxs(
                    subsegment_idx)
                """
                intersection between orig_segment interval and
                subsegment points
                """
                intersection = np.intersect1d(orig_idxs, P_idxs)
                """
                absolute nr of points that are covered by the subsegment
                in context of the orig segment
                """
                self._subsegment_coverage[i, subsegment_idx] =\
                    intersection.shape[0]
        """
        Construct an euclidean distance based KNN graph for the
        orig_segments
        """
        # self._orig_to_do = range(n_segments)
        self._orig_to_do = []
        """
        idxs_to_check = deque()

        def append_neighbours(i):
            ns = self._subsegment_storage.get_neighbours_of_segment(i)
            for n in ns:
                if n in self._orig_to_do:
                    continue
                idxs_to_check.append(n)
                self._orig_to_do.append(n)
        for i in range(n_segments):
            if i in self._orig_to_do:
                continue
            append_neighbours(i)
            while len(idxs_to_check) != 0:
                n = idxs_to_check.pop()
                append_neighbours(n)
            if len(self._orig_to_do) == n_segments:
                break
        """
        # """
        distances = np.zeros((n_orig_segments, n_orig_segments))
        orig_segs = np.zeros((n_orig_segments, n_orig_segments))
        sizes = np.zeros((n_orig_segments, ))
        for i in range(n_orig_segments):
            start, stop, length = self._get_interval(i)
            sizes[i] = length
            P_orig_seg_a = self._P[start:stop]
            center_a = np.mean(P_orig_seg_a, axis=0)
            for j in range(n_orig_segments):
                start, stop, _ = self._get_interval(j)
                P_orig_seg_b = self._P[start:stop]
                center_b = np.mean(P_orig_seg_b, axis=0)

                distances[i, j] = np.linalg.norm(center_a - center_b)
                orig_segs[i, j] = self._orig_segment_values[j]
            # sort the i-th row
            idxs = np.argsort(distances[i])
            distances[i] = distances[i, idxs]
            orig_segs[i] = orig_segs[i, idxs]
        # determine the row with the smallest segment
        size_idxs = np.argsort(sizes)
        # smallest segment should be
        start_idx = size_idxs[0]
        orig_values = orig_segs[start_idx]
        for orig_val in orig_values:
            subsegs = self._orig_segment_to_subsegment[orig_val]
            # print(orig_val, subsegs)
            for subseg_idx in subsegs:
                if subseg_idx in self._orig_to_do:
                    continue
                self._orig_to_do.append(subseg_idx)
        # print(len(self._orig_to_do), n_segments)
        # """
        assert len(self._orig_to_do) == n_segments
        # assert np.sum(self._subsegment_coverage[i]) <=\
        #     orig_idxs.shape[0]

    def reset(self):
        if not self._init:
            mkdir("./cache")
            if file_exists("./cache/cloud_env_cache.npz"):
                self._load()
            else:
                self._load_scene()
                self._save()
            n_orig_segments = self._subsegment_coverage.shape[0]
            n_segments = self._subsegment_coverage.shape[1]
            self._subsegments = np.zeros(n_segments, dtype=(np.int32))
            # coverage of the merged superpoints (isles)
            self._current_cov = np.zeros((n_orig_segments, n_segments))
            self._init = True
        self._reset()
        self._next_subsegment()
        if self._done:
            return self.reset()
        return self.next_state()

    def _load(self):
        data = np.load("./cache/cloud_env_cache.npz")
        self._P = data["P"]
        self._n_P = self._P.shape[0]
        self._orig_segment_values = data["orig_segment_values"]
        self._orig_indices = data["orig_indices"]
        self._orig_segment_counts = data["orig_segment_counts"]
        self._subsegment_coverage = data["subsegment_coverage"]
        self._assigned_segments = data["assigned_segments"]
        with open("./cache/orig_segment_to_idx.json") as infile:
            self._orig_segment_to_idx = json.load(infile)
            self._orig_segment_to_idx =\
                {int(k): int(v) for k, v in self._orig_segment_to_idx.items()}
        with open("./cache/orig_to_do.json") as infile:
            self._orig_to_do = json.load(infile)
        with open("./cache/orig_segment_to_subsegment.json") as infile:
            self._orig_segment_to_subsegment = json.load(infile)
            self._orig_segment_to_subsegment =\
                {int(k): v for k, v in self._orig_segment_to_subsegment.items()}
        with open("./cache/orig_subsegment_to_segment_vals.json") as infile:
            tmp_subsegment_to_segment_vals = json.load(infile)
            self._orig_subsegment_to_segment_vals = []
            for i in range(len(tmp_subsegment_to_segment_vals)):
                svs = tmp_subsegment_to_segment_vals[i]
                self._orig_subsegment_to_segment_vals.append(
                    np.array(svs, np.int32))
        self._subsegment_storage = SubsegmentStorage()
        self._subsegment_storage.load()

    def _save(self):
        np.savez(
            "./cache/cloud_env_cache.npz",
            P = self._P,
            orig_segment_values = self._orig_segment_values,
            orig_indices = self._orig_indices,
            orig_segment_counts = self._orig_segment_counts,
            subsegment_coverage = self._subsegment_coverage,
            assigned_segments = self._assigned_segments)
        with open("./cache/orig_segment_to_idx.json", "w") as outfile:
            json.dump(self._orig_segment_to_idx, outfile, indent=4)
        with open("./cache/orig_to_do.json", "w") as outfile:
            json.dump(self._orig_to_do, outfile, indent=4)
        with open("./cache/orig_segment_to_subsegment.json", "w") as outfile:
            json.dump(self._orig_segment_to_subsegment, outfile, indent=4)
        tmp_subsegment_to_segment_vals = []
        for i in range(len(self._orig_subsegment_to_segment_vals)):
            svs = self._orig_subsegment_to_segment_vals[i]
            tmp_subsegment_to_segment_vals.append(svs.tolist())
        with open("./cache/orig_subsegment_to_segment_vals.json", "w") as outfile:
            json.dump(tmp_subsegment_to_segment_vals, outfile, indent=4)
        self._subsegment_storage.save()

    def _get_interval(self, orig_seg_idx):
        start = self._orig_indices[orig_seg_idx]
        length = self._orig_segment_counts[orig_seg_idx]
        stop = start + length
        return start, stop, length

    def next_state(self):
        """
        Returns the point cloud + cloud idxs of the main segment + idxs
        of a neighbour.
        """
        self._P_idxs =\
            self._subsegment_storage.get_segment_point_idxs(
                self._main_segment_idx)
        self._neighbour_idxs =\
            self._subsegment_storage.get_segment_point_idxs(
                self._neighbour_segment_idx)
        return (
         self._P,
         self._P_idxs,
         self._neighbour_idxs,
         self._assigned_segments)

    def close(self):
        pass
