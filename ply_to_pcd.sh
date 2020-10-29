#!/bin/sh

if [ $# != 2 ] ; then
    echo "Need input of scannet path and target path"
    exit 1
fi

scannet_path=$1

if [ -d "${scannet_path}" ] ; then
    echo "'${scannet_path}' is a directory";
else
    echo "1. arg '${scannet_path}' is no directory"
    exit 1
fi

target_path=$2

if [ -d "${target_path}" ] ; then
    echo "'${target_path}' is a directory";
else
    echo "2. arg '${target_path}' is no directory"
    exit 1
fi

ply_l_file_ext="_vh_clean_2.labels.ply"
ply_c_file_ext="_vh_clean_2.ply"

echo "Files from '${scannet_path}' will be transformed into '${target_path}'"

for dir in ${scannet_path}/*/
do
    scene_name=${dir#*/}
    scene_name=${scene_name%/}
    scene_name=$(basename $scene_name)
    # echo ${scene_name}
    ply_l_file="${dir}${scene_name}${ply_l_file_ext}"
    ply_c_file="${dir}${scene_name}${ply_c_file_ext}"
    # echo ${ply_file}
    # create target directory
    target_dir="${target_path}/${scene_name}"
    mkdir ${target_dir}
    target_l_file="${target_dir}/${scene_name}_segments.pcd"
    target_txt_file="${target_dir}/${scene_name}_indices.txt"
    target_c_file="${target_dir}/${scene_name}_color.pcd"
    echo "Transform '${ply_c_file}' into '${target_l_file}'"
    pcl_mesh_sampling ${ply_l_file} ${target_l_file} ${target_txt_file} -n_samples 50000 -write_colors 1 -no_vis_result 1 -filter_in_between 1
    echo "Transform '${ply_c_file}' into '${target_c_file}'"
    pcl_mesh_sampling ${ply_c_file} ${target_c_file} -n_samples 50000 -write_colors 1 -no_vis_result 1
done
