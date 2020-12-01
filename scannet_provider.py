from environment.segmentation_ext import PCLCloud, region_growing_radius
import open3d as o3d
import numpy as np
import environment
from environment.utils import\
    generate_colors,\
    render_point_cloud,\
    create_segments,\
    filter_small_segments,\
    argsort_2D_mat_by_vec
import os
from scipy.spatial.transform import Rotation as R
from sklearn.decomposition import PCA
import argparse
import math


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


def angle(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def get_rotation_mat(angle, axis):
    r = R.from_rotvec(angle * axis)
    return r.as_matrix()


def split_segments(segments, P, dist=0.5, verbose=False):
    uni_segments, idxs, counts = np.unique(
        segments, return_index=True, return_counts=True)
    segments_r = np.zeros(segments.shape, dtype=np.int32)

    segment_offset = 0
    for i in range(uni_segments.shape[0]):
        idx = idxs[i]
        count = counts[i]

        start = idx
        stop = idx+count

        P_seg = P[start:stop]
        pcl_seg = PCLCloud(P_seg)

        sub_segments = region_growing_radius(pcl_seg, dist)
        n_sub_segments = np.max(sub_segments) + 1
        if verbose:
            print("n_sub_segments", n_sub_segments)
        sub_segments += segment_offset
        segments_r[start:stop] = sub_segments

        segment_offset += n_sub_segments

    return segments_r, segment_offset


def load(
        segments_filepath,
        color_filepath,
        filtered_idxs_filepath,
        save_path,
        small_segment_size=12,
        verbose=False):
    if os.path.isfile(save_path):
        # load
        data = np.load(save_path)
        P = data["P"]
        C_color = data["C_color"]
        segments = data["segments"]
        if verbose:
            print("raw segments loaded")
        return P, C_color, segments

    filtered_idxs = np.loadtxt(filtered_idxs_filepath, dtype=np.int32)

    pcd_segments = o3d.io.read_point_cloud(segments_filepath)
    P = np.asarray(pcd_segments.points)
    C_segments = np.asarray(pcd_segments.colors)

    pcd_color = o3d.io.read_point_cloud(color_filepath)
    C_color = np.asarray(pcd_color.colors)

    C_color = np.delete(C_color, filtered_idxs, axis=0)

    sorted_idxs = argsort_2D_mat_by_vec(C_segments)
    C_segments = C_segments[sorted_idxs]
    P = P[sorted_idxs]
    C_color = C_color[sorted_idxs]

    idx_to_del = filter_small_segments(
        arr=C_segments,
        small_segment_size=small_segment_size,
        axis=0,
        verbose=verbose)

    C_segments = np.delete(C_segments, idx_to_del[:], axis=0)
    P = np.delete(P, idx_to_del[:], axis=0)
    C_color = np.delete(C_color, idx_to_del[:], axis=0)

    segments, _, uni_C = create_segments(arr=C_segments, axis=0)
    if verbose:
        print("n remaining segments", uni_C.shape[0])

    np.savez(save_path, P=P, C_color=C_color, segments=segments)

    return P, C_color, segments


class DataProvider(environment.base_data_provider.BaseDataProvider):
    def __init__(
            self,
            max_scenes=2000,
            verbose=False,
            train_mode=False,
            train_p=0.8,
            n_cpus=None,
            batch_id=-1):
        """Constructor.

        Parameters
        ----------
        max_scenes : int
            Number of scenes/point clouds that should be used.
        verbose : type
            If True, log internal values of this class in the terminal.
        train_mode : boolean
            If True, the scenes will be splitted into train and test scenes.
        train_p : float
            Percentage of scenes that will be used for training.
        n_cpus : int
            Number of cpus that will be used for training.
        batch_id : int
            Only use a certain batch with batch_id. The batch size is equal to
            the number of cpus.
        """
        super().__init__()
        self.current_scene_idx = 0
        self.max_scenes = max_scenes
        self.verbose = verbose
        self.scenes = os.listdir("./ScannetScenes/")
        if os.path.isfile("blacklist.txt"):
            blacklist = open("blacklist.txt", "r")
            black_files = blacklist.readlines()
            for bf in black_files:
                if len(bf) < 3:
                    continue
                if len(bf) == 13:
                    bf = bf[0:-1]
                if bf in self.scenes:
                    self.scenes.remove(bf)
        if self.verbose:
            print(self.scenes)
        self.max_scenes = min(self.max_scenes, len(self.scenes))
        self.train_mode = train_mode
        print("Use:", self.max_scenes, "scenes - batch_id:", batch_id, "train_mode:", train_mode)
        if train_mode:
            assert(train_p < 1)
            n_train_samples = int(self.max_scenes * train_p)
            if batch_id != -1:
                frac = n_train_samples / n_cpus
                frac = math.floor(frac)
                if batch_id == n_cpus - 1:
                    n_train_samples = n_train_samples - ((n_cpus - 1) * frac)
                else:
                    n_train_samples = frac
                if self.verbose:
                    print("-------------batch_id", batch_id, n_train_samples)
                start = frac * batch_id
                stop = start + n_train_samples
                if self.verbose:
                    print("-------------batch_id", batch_id, start, stop)
                self.train_scenes = self.scenes[start:stop]
                self.train_idxs = np.arange(n_train_samples)
            else:
                self.train_scenes = self.scenes[:n_train_samples]
                self.train_idxs = np.arange(n_train_samples)
            self.test_scenes = self.scenes[n_train_samples:self.max_scenes]
            np.random.shuffle(self.train_idxs)
            self.train_idx = 0
            self.test_idx = 0
        else:
            self.scenes = self.scenes[:self.max_scenes]

    def next_id(self, scenes, idx, idxs=None):
        if idxs is not None:
            _idx = idxs[idx]
            self.id = scenes[_idx]
            idx += 1
            if idx == idxs.shape[0]:
                np.random.shuffle(idxs)
                idx = 0
        else:
            self.id = scenes[idx]
            idx += 1
            if idx == len(scenes):
                idx = 0
        return idx

    def select_id(self, train=True):
        if self.train_mode:
            if train:
                self.train_idx = self.next_id(
                    self.train_scenes, self.train_idx, idxs=self.train_idxs)
            else:
                self.test_idx = self.next_id(self.test_scenes, self.test_idx)
        else:
            self.current_scene_idx = self.next_id(
                self.scenes, self.current_scene_idx)
        return self.id

    def get_cloud_and_segments(self, dist=0.2, small_segment_size=250):
        segments_filepath = "./ScannetScenes/" + self.id + "/" + self.id + "_segments.pcd"
        color_filepath = "./ScannetScenes/" + self.id + "/" + self.id + "_color.pcd"
        filtered_idxs_filepath = "./ScannetScenes/" + self.id + "/" + self.id + "_indices.txt"
        save_path = "./ScannetScenes/" + self.id + "/" + self.id + ".npz"
        segments_path = "./ScannetScenes/" + self.id + "/" + self.id + "_segments.npz"

        if not os.path.isfile(segments_path):
            P, C, orig_segments = load(
                segments_filepath=segments_filepath,
                color_filepath=color_filepath,
                filtered_idxs_filepath=filtered_idxs_filepath,
                save_path=save_path,
                verbose=self.verbose)

            uni_segments = np.unique(orig_segments)
            n_segments = uni_segments.shape[0]
            if self.verbose:
                print("n segments:", n_segments)
            P = np.hstack((P, C))
            if self.verbose:
                print("split segments")
            orig_segments, n_segments = split_segments(
                P=P,
                segments=orig_segments,
                dist=dist,
                verbose=self.verbose)
            if self.verbose:
                print("n segments", n_segments)
                print("sort points according to segment idxs")
            sorted_idxs = np.argsort(orig_segments)
            orig_segments = orig_segments[sorted_idxs]
            P = P[sorted_idxs]

            uni_segments, uni_idxs, uni_counts = np.unique(
                orig_segments, return_index=True, return_counts=True)
            n_segments = uni_segments.shape[0]
            if self.verbose:
                print("n segments:", n_segments)

            np.savez(segments_path, P=P, orig_segments=orig_segments)
            if self.verbose:
                print("segments saved")
        else:
            data = np.load(segments_path)
            P = data["P"]
            orig_segments = data["orig_segments"]
            if self.verbose:
                print("segments loaded")

        P[:, 3:] *= 255
        if self.verbose:
            print("filter small segments")
        idx_to_del = filter_small_segments(
            arr=orig_segments,
            small_segment_size=small_segment_size,
            verbose=self.verbose)
        P = np.delete(P, idx_to_del, axis=0)
        orig_segments = np.delete(orig_segments, idx_to_del)
        if self.verbose:
            print("reassgin segments")
        orig_segments, n_segments, _ = create_segments(arr=orig_segments)

        xyz_mean = np.mean(P[:, :3], axis=0)
        P[:, :3] = P[:, :3] - xyz_mean

        pca = PCA()
        pca.fit(P[:, :3])
        vec = pca.components_[0]
        ang = angle(vec, np.array([1, 0, 0]))
        if self.verbose:
            print("rotate P around", -np.rad2deg(ang), "°")
        R = get_rotation_mat(angle=-ang, axis=np.array([0, 0, 1]))
        P[:, :3] = np.transpose(np.matmul(R, np.transpose(P[:, :3])))

        self.P = P
        self.segments = orig_segments
        return self.P, self.segments, self.id


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        type=str,
        default="precalc",
        help="options: precalc, visualize_all, visualize_single")
    parser.add_argument(
        "--scene",
        type=str,
        default="scene0700_00",
        help="scene from the scannet dataset")
    parser.add_argument(
        "--dist",
        type=float,
        default=0.2,
        help="distance to split semantic segments into single objects")
    parser.add_argument(
        "--small",
        type=int,
        default=250,
        help="size to filter small segments")
    parser.add_argument(
        "--render_segs",
        type=bool,
        default=False,
        help="flag to render every segment of scene")
    parser.add_argument(
        "--verbose",
        type=bool,
        default=False,
        help="flag to print output")
    args = parser.parse_args()
    dat_p = DataProvider(verbose=args.verbose)
    print("mode:", args.mode)
    if args.mode == "visualize_single":
        dat_p.id = args.scene
        print("scene:", args.scene)
        P, segs, id = dat_p.get_cloud_and_segments(
            dist=args.dist,
            small_segment_size=args.small)
        print(id, P.shape, segs.shape)
        n_segments = np.unique(segs).shape[0]
        colors = generate_colors(max_colors=n_segments + 10)
        render_point_cloud(P=P, animate=True)
        render_point_cloud(
            P=P, segments=segs, colors=colors, animate=True)
        if args.render_segs:
            uni_segs, uni_idxs, uni_counts = np.unique(
                segs, return_index=True, return_counts=True)
            for i in range(uni_segs.shape[0]):
                idx = uni_idxs[i]
                count = uni_counts[i]
                P_ = P[idx:idx+count, :3]
                segs_ = np.zeros((count, ), dtype=np.int32)
                segs_[:] = uni_segs[i]
                render_point_cloud(
                    P=P_, segments=segs_, colors=colors, animate=True)
    else:
        for i in range(len(dat_p.scenes)):
            try:
                P, segs, id = dat_p.get_cloud_and_segments(
                    dist=args.dist,
                    small_segment_size=args.small)
            except Exception as e:
                print("Error while loading scene", dat_p.id, e)
                blacklist = open("blacklist.txt", "a")
                blacklist.write("\n")
                blacklist.write(dat_p.id)
                blacklist.close()
                dat_p.select_id()
                continue
            print(id, P.shape, segs.shape)
            n_segments = np.unique(segs).shape[0]
            colors = generate_colors(max_colors=n_segments + 10)
            if args.mode == "visualize_all":
                render_point_cloud(P=P[:, :3], segments=segs, colors=colors)
            dat_p.select_id()