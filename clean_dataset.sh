#!/bin/sh

echo "Warning: All preprocessed scene files of the scannet dataset will be deleted"

scannet_path="./ScannetScenes"

for dir in ${scannet_path}/*/
do
    scene_name=${dir#*/}
    scene_name=${scene_name%/}
    scene_name=$(basename $scene_name)
    rm "${dir}${scene_name}.npz"
    rm "${dir}${scene_name}_segments.npz"
done
