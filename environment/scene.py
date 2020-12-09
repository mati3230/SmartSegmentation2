import numpy as np
from .segmentation_ext import PCLCloud, vccs, get_plane_segment
import json
from .superpoint_storage import SuperpointStorage
from .utils import\
    get_interval,\
    mkdir,\
    file_exists,\
    render_point_cloud,\
    generate_colors


class Scene():
    """Class to manage (load, save, render) a point cloud scene. We use a C++
    interface via boost to use methods of the PCL such as the VCCS
    oversegmentation.

    Parameters
    ----------
    id : int
        The scenes are identified by numbers.
    get_cloud_and_segments : function
        The function should return a point cloud and its segments.
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

    Attributes
    ----------
    scene_id : int
        The scenes are identified by numbers.
    unsegmented : int
        Label of points that are unsegmented.
    error : boolean
        deprecated.
    get_cloud_and_segments : function
        The function should return a point cloud and its segments.
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
    P : np.ndarray
        The point cloud.
    orig_segments_nrs : np.ndarray
        Unique segment values in the ground truth segmentation.
    orig_indices : np.ndarray
        The start indices of the different segment values.
    orig_segment_counts : np.ndarray
        The number of points that have the different segment values.
    n_P : int
        The number of points in the point cloud.
    orig_pcl_cloud : PCLCloud
        Point cloud interface to use the PCL functions.
    assigned_segments : np.ndarray
        Segment values of presegmented points. Presegmentation is conducted
        with the RANSAC algorithm.
    superpoint_storage : SuperpointStorage
        Storage of the superpoints.
    orig_superpoint_to_segment_nr : list(np.ndarray)
        Enter a superpoint index and get ground truth segments with a force >0.
    orig_segment_nr_to_superpoint : dict
        Insert a segment value of a ground truth segment and get the superpoint
        that have a force >0 with the inserted segment.
    n_orig_segments : int
        Number of ground truth segments.
    superpoint_coverage : np.ndarray
        How much points of a orig_segment are covered by a superpoint? Initial
        forces matrix.
    orig_segment_to_idx : dict
        Enter a ground truth segment value and get the corresponding segment
        index.
     orig_to_do : deque
        Ground truth order of main superpoints that should be considered in the
        superpoint growing environment.

    """

    def __init__(
            self,
            id,
            get_cloud_and_segments,
            voxel_r=0.1,
            seed_r=1,
            color_i=0.75,
            normal_i=0.75,
            spatial_i=0.0,
            min_superpoint_size=20):
        """Constructor. See https://pointclouds.org/documentation/tutorials/supervoxel_clustering.html
        regarding the parameters of the VCCS algorithm. We consider a scene as
        preprocessed if a the segments of a point cloud as well as the
        superpoints by the VCCS algorithm are computed. Finally, a scene
        consist of the point cloud, its segments and the superpoints.

        Parameters
        ----------
        id : int
            The scenes are identified by numbers.
        get_cloud_and_segments : function
            The function should return a point cloud and its segments.
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
        """
        self.scene_id = id
        self.get_cloud_and_segments = get_cloud_and_segments
        self.voxel_r = voxel_r
        self.seed_r = seed_r
        self.color_i = color_i
        self.normal_i = normal_i
        self.spatial_i = spatial_i
        self.min_superpoint_size = min_superpoint_size
        self.unsegmented = 0
        self.error = False
        # create a cache directory to store preprocessed scenes
        mkdir("./cache")
        # create a directory for that specific scene
        mkdir("./cache/" + self.scene_id)
        # if a cached scene already exists
        if file_exists("./cache/" + self.scene_id + "/cloud_env_cache.npz"):
            self.load()
        else:
            try:
                self.prepare_scene()
            except Exception as e:
                self.error = True
                print(e)
                return
            self.save()

    def prepare_scene(self):
        """
        Method that loads the point cloud and computes the superpoints by
        the VCCS algorithm.
        """
        # load the point cloud and its segments
        self.P, orig_segments, self.scene_id = self.get_cloud_and_segments()
        self.orig_segment_nrs,\
            self.orig_indices,\
            self.orig_segment_counts = np.unique(
                orig_segments, return_index=True, return_counts=True)
        # number of points
        self.n_P = self.P.shape[0]
        # init C++ PCL class
        self.orig_pcl_cloud = PCLCloud(self.P)
        # build kd-tree
        self.orig_pcl_cloud.BuildTree()
        # print("Tree builded")
        self.assigned_segments = np.zeros((self.n_P,), dtype=(np.int32))

        """
        Extract the ground plane -- most often the ground is one object and
        does not need to be oversegmented
        The ground plane segmentation is realised with the RANSAC algorithm.
        See https://pointclouds.org/documentation/tutorials/random_sample_consensus.html
        for more information regarding the RANSAC algorithm.
        """
        max_iterations = 1000
        threshold = 0.06
        probability = 0.9
        # apply the RANSAC algorithm
        plane_result = get_plane_segment(
            self.orig_pcl_cloud, max_iterations, threshold, probability)
        ok = plane_result[(-1)]
        # if an error occured
        if not ok:
            raise Exception("Could not segment plane")

        # point indices of the plane
        plane_idxs = plane_result[0]
        # print("{0} points segmented in plane step".format(plane_idxs.shape[0]))
        # now the ground is segmented - thus, we can set a segment number
        self.assigned_segments[plane_idxs] = self.unsegmented + 1

        # all indices of the point cloud
        all_idxs = np.arange(self.n_P)
        # point cloud indices without the indices of the ground
        remaining_idxs = np.delete(all_idxs, plane_idxs)
        # point cloud without the ground points
        P_ = self.P[remaining_idxs]
        P_cloud = PCLCloud(P_)
        # apply the vccs oversegmentation
        vccs_result = vccs(
            P_cloud,
            self.voxel_r,
            self.seed_r,
            self.color_i,
            self.normal_i,
            self.spatial_i)
        """
        Delete superpoint that are smaller than a user defined minimum size.
        """
        # determine the number of superpoints
        n_superpoints = int(len(vccs_result) / 3)
        # indices of segments that should be deleted
        idxs_to_del = []
        # segment number which should be deleted
        segments_to_del = []
        for i in range(n_superpoints):
            idx = i * 3
            """
            Extract the segment number and the point indices of the segment.
            Note, that the point indices of each segment are related to the
            point cloud without the ground plane. We want to find the point
            indices of the original point cloud that includes the points of the
            ground plane.
            """
            P_idxs = vccs_result[(idx + 1)]
            segment = vccs_result[idx]

            """
            find the point indices of the point cloud that includes the ground
            plane
            """
            points_ = P_[P_idxs]
            P_idxs = self.orig_pcl_cloud.Search(points_)
            vccs_result[idx + 1] = P_idxs

            # if a segment is to small, then add it to a deletion list
            if P_idxs.shape[0] < self.min_superpoint_size:
                idxs_to_del.extend([idx, idx + 1, idx + 2])
                segments_to_del.append(segment)
                continue

        # delete the small segments
        # print("{0}/{1} have been deleted from vccs result".format(len(segments_to_del), plane_idxs.shape[0]))
        segments_to_del = np.array(segments_to_del, dtype=(np.int32))
        for i in range(n_superpoints):
            # get the neighbours of a segment
            n_idx = i * 3 + 2
            neighbours = vccs_result[n_idx]

            # determine if a neighbour should be deleted
            _, n_idxs, _ = np.intersect1d(
                neighbours,
                segments_to_del,
                return_indices=True)
            # deletion of the neighbours
            if n_idxs.shape[0] > 0:
                neighbours = np.delete(neighbours, n_idxs)
                vccs_result[n_idx] = neighbours

        # delete segment nr, point indices of segment and neighbour indices
        # print("delete {0} elements in vccs result structure".format(len(idxs_to_del)))
        for idx in reversed(idxs_to_del):
            del vccs_result[idx]

        # reduced number of superpoints
        n_superpoints = int(len(vccs_result) / 3)
        # print("n_vccs_segments:", n_superpoints)
        # dictionary: segment nr to index in vccs result
        vccsSegmentToIdx = {}
        for i in range(n_superpoints):
            segment = vccs_result[(i * 3)]
            vccsSegmentToIdx[segment] = i

        """
        Transform vccs_result to an array which consists a list of point
        indices and neighbour indices of each superpoint. Note, that the
        neighbours in vccs_result are stored as segment numbers. We replace
        those segment numbers with segment indices.
        """
        vccs_res = [list(), list()]
        for i in range(n_superpoints):
            neighbours = vccs_result[(i * 3 + 2)]
            for j in range(neighbours.shape[0]):
                neighbour = int(neighbours[j])
                # get the segment index of vccs_result of a neighbour
                idx = vccsSegmentToIdx[neighbour]
                # replacement
                neighbours[j] = idx
                # assert idx < n_superpoints

            vccs_result[i * 3 + 2] = neighbours
            vccs_res[0].append(vccs_result[(i * 3 + 1)])
            vccs_res[1].append(vccs_result[(i * 3 + 2)])

        self.superpoint_storage = SuperpointStorage(pns_orig=vccs_res)
        self.superpoint_storage.reset()
        # original containers that will not change
        self.orig_superpoint_to_segment_nr = []
        self.orig_segment_nr_to_superpoint = {}
        # for all original segment values
        for orig_segment in self.orig_segment_nrs:
            # add a list
            self.orig_segment_nr_to_superpoint[int(orig_segment)] = list()

        # for all superpoints
        for i in range(n_superpoints):
            # get the cloud idxs of the superpoint
            P_idxs = self.superpoint_storage.get_sp_point_idxs(i)
            # get the original segment numbers of the superpoint indices
            orig_segment_nrs = orig_segments[P_idxs]
            # which original values are there?
            uni_orig_segments, counts = np.unique(
                orig_segment_nrs,
                return_counts=True)
            """
            sort the occurences of the orig segment numbers of a superpoint in
            descending order (begin by the largest values)
            """
            idxs = np.argsort(counts)[::-1]
            uni_orig_segments = uni_orig_segments[idxs]
            # append the superpoint to the lists of the orig segments
            for orig_segment in uni_orig_segments:
                self.orig_segment_nr_to_superpoint[orig_segment].append(i)
            # this superpoint lies in uni_orig_segments
            self.orig_superpoint_to_segment_nr.append(uni_orig_segments)

        # nr of orig segments
        n_orig_segments = self.orig_segment_nrs.shape[0]
        # how much points of a orig_segment are covered by a superpoint
        self.superpoint_coverage = np.zeros((n_orig_segments, n_superpoints))
        """
        Note, that the orig segment numbers will not start by zero.
        Structure to convert segment number to an idx.
        """
        self.orig_segment_to_idx = {}
        # for all orig values
        for i in range(self.orig_segment_nrs.shape[0]):
            orig_segment = self.orig_segment_nrs[i]
            # save the idx of the orig segment number
            self.orig_segment_to_idx[int(orig_segment)] = i
            # get the interval of the orig_segment
            start, stop, length = get_interval(
                i,
                self.orig_indices,
                self.orig_segment_counts)
            orig_idxs = np.arange(start, stop)
            # get all superpoints that are in orig_segment
            superpoint_list =\
                self.orig_segment_nr_to_superpoint[orig_segment]
            # for all superpoints
            for superpoint_idx in superpoint_list:
                # get a list of points for all superpoints
                P_idxs = self.superpoint_storage.get_sp_point_idxs(
                    superpoint_idx)
                """
                intersection between orig_segment interval and
                superpoint points
                """
                intersection = np.intersect1d(orig_idxs, P_idxs)
                """
                absolute nr of points that are covered by the superpoint
                in context of the orig segment
                """
                self.superpoint_coverage[i, superpoint_idx] =\
                    intersection.shape[0]
        """
        Construct an euclidean distance based KNN graph for the
        orig_segments
        """
        self.orig_to_do = []
        distances = np.zeros((n_orig_segments, n_orig_segments))
        orig_segs = np.zeros((n_orig_segments, n_orig_segments))
        # sizes of the original segments
        sizes = np.zeros((n_orig_segments, ))
        # for each original segment
        for i in range(n_orig_segments):
            start, stop, length = get_interval(
                i,
                self.orig_indices,
                self.orig_segment_counts)
            sizes[i] = length
            """
            determine the center of a segment and measure the distance to all
            other segment centers
            """
            P_orig_seg_a = self.P[start:stop]
            center_a = np.mean(P_orig_seg_a, axis=0)
            for j in range(n_orig_segments):
                start, stop, _ = get_interval(
                    j,
                    self.orig_indices,
                    self.orig_segment_counts)
                P_orig_seg_b = self.P[start:stop]
                center_b = np.mean(P_orig_seg_b, axis=0)

                distances[i, j] = np.linalg.norm(center_a - center_b)
                orig_segs[i, j] = self.orig_segment_nrs[j]
            # sort the i-th row
            idxs = np.argsort(distances[i])
            distances[i] = distances[i, idxs]
            orig_segs[i] = orig_segs[i, idxs]
        # determine the row with the smallest segment
        size_idxs = np.argsort(sizes)
        # smallest segment should be
        start_idx = size_idxs[0]
        orig_segment_nrs = orig_segs[start_idx]
        for orig_segment_nr in orig_segment_nrs:
            superpoints = self.orig_segment_nr_to_superpoint[orig_segment_nr]
            # print(orig_segment_nr, subsegs)
            for superpoint_idx in superpoints:
                if superpoint_idx in self.orig_to_do:
                    continue
                self.orig_to_do.append(superpoint_idx)
        assert len(self.orig_to_do) == n_superpoints

    def load(self):
        """
        Load all containers that are generated in the expensive method
        prepare_scene.
        """
        data = np.load("./cache/" + self.scene_id + "/cloud_env_cache.npz")
        self.P = data["P"]
        self.n_P = self.P.shape[0]
        self.orig_segment_nrs = data["orig_segment_nrs"]
        self.orig_indices = data["orig_indices"]
        self.orig_segment_counts = data["orig_segment_counts"]
        self.superpoint_coverage = data["superpoint_coverage"]
        self.assigned_segments = data["assigned_segments"]
        with open("./cache/" + self.scene_id + "/orig_segment_to_idx.json") as infile:
            self.orig_segment_to_idx = json.load(infile)
            self.orig_segment_to_idx =\
                {int(k): int(v) for k, v in self.orig_segment_to_idx.items()}
        with open("./cache/" + self.scene_id + "/orig_to_do.json") as infile:
            self.orig_to_do = json.load(infile)
        with open("./cache/" + self.scene_id + "/orig_segment_nr_to_superpoint.json") as infile:
            self.orig_segment_nr_to_superpoint = json.load(infile)
            self.orig_segment_nr_to_superpoint =\
                {int(k): v for k, v in self.orig_segment_nr_to_superpoint.items()}
        with open("./cache/" + self.scene_id + "/superpoint_to_orig_segment_nrs.json") as infile:
            tmp_superpoint_to_segment_vals = json.load(infile)
            self.orig_superpoint_to_segment_nr = []
            for i in range(len(tmp_superpoint_to_segment_vals)):
                svs = tmp_superpoint_to_segment_vals[i]
                self.orig_superpoint_to_segment_nr.append(
                    np.array(svs, np.int32))
        self.superpoint_storage = SuperpointStorage()
        self.superpoint_storage.load(id=self.scene_id)

    def save(self):
        """
        Save all containers that are generated in the expensive method
        prepare_scene.
        """
        np.savez(
            "./cache/" + self.scene_id + "/cloud_env_cache.npz",
            P=self.P,
            orig_segment_nrs=self.orig_segment_nrs,
            orig_indices=self.orig_indices,
            orig_segment_counts=self.orig_segment_counts,
            superpoint_coverage=self.superpoint_coverage,
            assigned_segments=self.assigned_segments)
        with open("./cache/" + self.scene_id + "/orig_segment_to_idx.json", "w") as outfile:
            json.dump(self.orig_segment_to_idx, outfile, indent=4)
        with open("./cache/" + self.scene_id + "/orig_to_do.json", "w") as outfile:
            json.dump(self.orig_to_do, outfile, indent=4)
        with open("./cache/" + self.scene_id + "/orig_segment_nr_to_superpoint.json", "w") as outfile:
            json.dump(self.orig_segment_nr_to_superpoint, outfile, indent=4)
        tmp_superpoint_to_segment_vals = []
        for i in range(len(self.orig_superpoint_to_segment_nr)):
            svs = self.orig_superpoint_to_segment_nr[i]
            tmp_superpoint_to_segment_vals.append(svs.tolist())
        with open("./cache/" + self.scene_id + "/superpoint_to_orig_segment_nrs.json", "w") as outfile:
            json.dump(tmp_superpoint_to_segment_vals, outfile, indent=4)
        self.superpoint_storage.save(id=self.scene_id)

    def render_vccs(self):
        """
        Render the VCCS superpoints.
        """
        self.superpoint_storage.reset()
        n_superpoints = self.superpoint_coverage.shape[1]
        # create a segment for each superpoint
        segments = self.unsegmented * np.ones((self.P.shape[0], ), np.int32)
        seg = 1
        for sup in range(n_superpoints):
            P_idxs = self.superpoint_storage.get_sp_point_idxs(idx=sup)
            segments[P_idxs] = seg
            seg += 1
        # render the cloud and segments of the VCCS algorithm
        colors = generate_colors(max_colors=(n_superpoints + 10))
        render_point_cloud(
            P=self.P[:, :3],
            segments=segments,
            colors=colors,
            animate=True)
