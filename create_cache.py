from environment.scene import Scene
from scannet_provider import DataProvider
import argparse
from multiprocessing import Process
import math


def create_scene(scene_id, dat_p, args):
    """Short summary.

    Parameters
    ----------
    scene_id : int
        Description of parameter `scene_id`.
    dat_p : BaseDataProvider
        Data Provider.
    args : dict
        Arguments of the VCCS algorithm.

    Returns
    -------
    Scene
        Point cloud scene that is processed by the VCCS algorithm.

    """
    return Scene(
        id=scene_id,
        get_cloud_and_segments=dat_p.get_cloud_and_segments,
        voxel_r=args.voxel_r,
        seed_r=args.seed_r,
        color_i=args.color_i,
        normal_i=args.normal_i,
        spatial_i=args.spatial_i)


def process_range(id, min_i, max_i):
    """Processes a range of scene from min to max. This method is used for
    parallel processing.

    Parameters
    ----------
    id : int
        Worker ID.
    min_i : type
        Minimum index of the scene.
    max_i : type
        Maximum index of the scene.
    """
    print(id, max_i, min_i, max_i - min_i)
    dat_p = DataProvider(max_scenes=2000)
    for i in range(min_i, max_i):
        scene_id = dat_p.scenes[i]
        dat_p.id = scene_id
        scene = create_scene(scene_id, dat_p, args)
        if scene.error:
            blacklist = open("blacklist.txt", "a")
            blacklist.write("\n")
            blacklist.write(dat_p.id)
            blacklist.close()
        del scene
        print(id, ((i-min_i+1)/(max_i - min_i)))


def main(args):
    """Main entry point to generate a cache for the scenes.

    Parameters
    ----------
    args : dict
        Arguments, e.g. of the VCCS algorithm, to process the scenes.

    """
    dat_p = DataProvider(max_scenes=2000)
    if args.mode == "visualize_single":
        scene_id = args.scene
        dat_p.id = scene_id
        scene = create_scene(scene_id, dat_p, args)
        scene.render_vccs()
        return
    len_scenes = len(dat_p.scenes)
    if args.n_cpus == 1:
        for i in range(len_scenes):
            scene_id = dat_p.scenes[i]
            dat_p.id = scene_id
            scene = create_scene(scene_id, dat_p, args)
            if scene.error:
                blacklist = open("blacklist.txt", "a")
                blacklist.write("\n")
                blacklist.write(dat_p.id)
                blacklist.close()
            print((i + 1)/len_scenes)
            if args.mode == "visualize_all":
                scene.render_vccs()
    else:
        processes = []
        intervals = math.floor(len_scenes / args.n_cpus)
        for i in range(args.n_cpus):
            min_i = i*intervals
            max_i = min_i + intervals
            if max_i > len_scenes:
                diff = max_i - len_scenes
                max_i -= diff
            if max_i == min_i:
                continue
            p = Process(target=process_range, args=(i, min_i, max_i))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


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
        "--voxel_r",
        type=float,
        default=0.1,
        help="voxel resolution")
    parser.add_argument(
        "--seed_r",
        type=float,
        default=1,
        help="seed resolution")
    parser.add_argument(
        "--color_i",
        type=float,
        default=0.75,
        help="color importance")
    parser.add_argument(
        "--normal_i",
        type=float,
        default=0.75,
        help="normal importance")
    parser.add_argument(
        "--spatial_i",
        type=float,
        default=0.0,
        help="spatial importance")
    parser.add_argument(
        "--min_superpoint_size",
        type=int,
        default=20,
        help="superpoints smaller than threshold will be deleted")
    parser.add_argument(
        "--n_cpus",
        type=int,
        default=1,
        help="n cpus used for the cache creation")
    args = parser.parse_args()
    main(args=args)
