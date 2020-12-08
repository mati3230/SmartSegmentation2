# Smart Segmentation 2

This project is developed on a linux system. Hence, most of the manuals where commands are typed in the terminal assuming a linux system. However, this project can also be used on Windows or Mac. To do so, it is recommended to transform the meshes to point clouds and create the cache on a linux system (see Section [PLY2PCD](#ply2pcd) and [CREATE CACHE](#create_cache)). Subsequently, the folders with the generated files should be copied to the Windows or Mac system. Note, that the steps in Section [PLY2PCD](#ply2pcd) and [CREATE CACHE](#create_cache) only have to be executed if this project is used in combination with the ScanNet dataset [2]. Moreover, errors can occur on Windows or Mac due to the multiprocessing that is used.

## REQUIREMENTS

* python >= 3.6.8
* pip
* doxygen (optional)
* PCL
* Boost
* ScanNet dataset [2] (optional -- if applying point cloud processing)
* Telegram (optional -- if remote training with the telegram app should be used)

## GENERATE DOCUMENTATION

[Doxygen](https://www.doxygen.nl/index.html) can be used to generate a code documentation. Open a terminal in the root directory and type *doxygen Doxyfile*. After that, the index file can be linked with *ln -s ./documentation/html/index.html ./documentation.html*

## PCL LIBRARY

The following step is only necessary, if the steps in Section [PLY2PCD](#ply2pcd) should be executed. We extended the *pcl_mesh_sampling* tool of the PCL that is used to execute the steps in Section [PLY2PCD](#ply2pcd). Copy the script in the directory */pcl/...* and build the PCL.

<a name="ply2pcd"></a>
## PLY2PCD

* In case of using multiple disks -- make sure all disks of the paths are mounted!
* The pcd will be generated by an extended pcl_mesh_sampling tool which is in the pcl folder. Copy the .cpp file in your pcl folder. The tool can be compiled by recompiling the pcl with *make* followed by *sudo make install*
* Use the tool *ply_to_pcd.sh* to transform the meshes to point cloud scenes: *sh ./ply_to_pcd.sh DIR_TO_SCANNET_SCENES DESIRED_DIR* -- e.g. *sh ./ply_to_pcd.sh ${SCANNET_ROOT}/scans ./ScannetScenes* -- this will take some time.

<a name="create_cache"></a>
## CREATE CACHE

* The cache stores the superpoints that are generated by an algorithm such as the VCCS algorithm.
* If you want to cache all scenes, then just type *python create_cache.py*
* You can inspect the parameters with *python create_cache.py -h*
* The VCCS superpoints of a specific scene can be visualized with: *python create_cache.py --mode=visualize_single --scene=scene0268_01*

## VISUALIZATION OF THE GROUND TRUTH SEGMENTS

* Use *python scannet_provider.py --mode=visualize_single --scene=scene0268_01* to visualize the ground truth segments of a certain scene.

## VISUALISATION OF THE ORIGINAL POINT CLOUD

* The original point cloud can be visualized with the pc_tools by *python ../pc_tools/pcd_viewer.py --input=./ScannetScenes/scene0268_01/scene0268_01_color.pcd --format=xyzrgb*
* The corresponding segments can be visualized by *python ../pc_tools/pcd_viewer.py --input=./ScannetScenes/scene0268_01/scene0268_01_color.pcd --format=xyzrgb*

## VISUALISATION OF THE AGENT SEGMENTATION

* Use *python agent.py --model=mvcnn_ac_1.0_Imi --train_mode=imitation --train_types=imitation_types.json --train_args=imitation_args.json --render_suggestions=True --render_size=128* to visualize the segmentation of the agent

## TRAINING

<a name="prerequisites_json"></a>
### PREREQUISITES - JSON

Before the training can be started, a .json file has to be filled out. Each training type such as *reinforcement learning* or *imitation learning* has a *_types.json* file. For example, have a look at the file [*ppo2_types.json*](ppo2_types.json). The PPO algorithm [1] is used for the reinforcement learning. The corresponding .json file [*ppo2_types.json*](ppo2_types.json) includes the parameters that are associated with the training process with the PPO algorithm. Moreover, it specifies the data types of the parameters. The possible data types are listed below.

| String         | Datatype | Range       |
|----------------|----------|-------------|
| pos int        | int      | >=0         |
| neg int        | int      | <=0         |
| real pos int   | int      | >0          |
| real neg int   | int      | <0          |
| pos float      | float    | >0          |
| neg float      | float    | <0          |
| real pos float | float    | >=0         |
| real neg float | float    | <=0         |
| str            | str      |             |
| tuple          | tuple    |             |
| bool           | boolean  | true, false |

The case of *tuple real pos int* can be used for shapes of arrays such as *(4,64,64,3)*. An exception will be thrown if an argument does not fulfill the requirements on the range such *<0*. According to the [*ppo2_types.json*](ppo2_types.json) file, the user can input a *_args.json* file. The values of the parameters are specified in this file. For example, the values of the parameters of the PPO algorithm [1] are specified in the file [*ppo2_args.json*](./sample_args/ppo2_gym_args.json). It is located in the folder *./sample_args* where example argument files can be found. A short description of the parameters can be found below.

| Parameter            | Description                                                                                                                                                                                   |
|----------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| token                | Your telegram bot token. **Only for remote training in combination with the telegram app.**                                                                                                   |
| learning_rate        | Learning rate that is used by the optimizer such as ADAM.                                                                                                                                     |
| n_cpus               | Number of CPUs that are used for the sampling of data.                                                                                                                                        |
| n_batches            | Number of Batches that should be sampled.                                                                                                                                                     |
| batch_size           | Size of one batch.                                                                                                                                                                            |
| global_norm          | Global threshold to clip the gradients. The norm of the gradients should not exceed this threshold. See in the [tensorflow](https://www.tensorflow.org/ ) documentation for more information. |
| seed                 | Random seed that should be used.                                                                                                                                                              |
| gamma                | Discount factor of the PPO algorihtm.                                                                                                                                                         |
| K_epochs             | Number of epochs to update the policy in one update cycle.                                                                                                                                    |
| eps_clip             | The epsilon clipping factor.                                                                                                                                                                  |
| lmbda                | The generalized advantage estimation factor lambda.                                                                                                                                           |
| entropy_factor       | Factor for the entropy loss.                                                                                                                                                                  |
| value_factor         | Factor for the value loss.                                                                                                                                                                    |
| test_interval        | Specifies after how many update cycles a test will be conducted.                                                                                                                              |
| env_name             | Name of the environment.                                                                                                                                                                      |
| n_actions            | Number of actions of the environment.                                                                                                                                                         |
| n_ft_outpt           | Size of the vector that is calculated by the feature detector. Read the [README.md](./policies/README.md) in the folder policies for more information.                                        |
| model_name           | Name of the policy.                                                                                                                                                                           |
| state_size           | Size of an observation.                                                                                                                                                                       |
| policy_path          | Relative or absolute path to the policy.                                                                                                                                                      |
| policy_type          | Class type of the policy.                                                                                                                                                                     |
| env_path             | Relative or absolute path to the environment.                                                                                                                                                 |
| env_type             | Class type of the environment.                                                                                                                                                                |
| data_provider_path   | Path to the data provider. Only used in case of point cloud data.                                                                                                                             |
| normalize_returns    | If true, the returns will be normalized.                                                                                                                                                      |
| normalize_advantages | If true, the advantages will be normalized.                                                                                                                                                   |
| stddev               | deprecated                                                                                                                                                                                    |
| initializer          | Initializer that will be used to initialize the weights. See in the [tensorflow](https://www.tensorflow.org/) documentation for more information.                                             |
| max_scenes           | Maximum number of point cloud scenes that will be used for the training. **Only used in case of point cloud data.**                                                                           |
| main_gpu_mem         | GPU memory that will be used for the update cycle.                                                                                                                                            |
| w_gpu_mem            | GPU memory that will be used to sample transitions.                                                                                                                                           |
| train_p              | Fraction of scenes that will be used for the training. **Only used in case of point cloud data.**                                                                                             |
| train_mode           | If true, the fraction *train_p* will be used to split the data into training and test scenes. **Only used in case of point cloud data.**                                                      |

The parameters that are marked with a **Only used in case of point cloud data.** should not be used in case of training with a standard gym environment. For example, see the file [*ppo2_gym_args.json*](./sample_args/ppo2_gym_args.json). The corresponding types file for the imitation learning is [*imitation_types.json*](imitation_types.json).
The parameters that are not appear in the PPO parameters are explained below.

| Parameter   | Description                                     |
|-------------|-------------------------------------------------|
| ce_factor   | Factor for the cross entropy loss.              |
| beta        | Factor for the L2 regularization.               |
| expert_path | Relative or absolute path to the expert policy. |
| expert_type | Class type of the expert policy.                |

A sample file can be found under [*./sample_args/imitation_args.json*](./sample_args/imitation_args.json). Currently, the imitation learning can only be used in combination with the point cloud data.

### PREREQUISITES - TELEGRAM (OPTIONAL)

The framework can be used in combination with telegram to remotely control the training process. First you have to create a telegram account. After that, a [bot](https://docs.microsoft.com/en-us/azure/bot-service/bot-service-channel-connect-telegram?view=azure-bot-service-4.0) have to be created. Subsequently, the bot [token](https://docs.microsoft.com/en-us/azure/bot-service/bot-service-channel-connect-telegram?view=azure-bot-service-4.0) should be identified and inserted in the *_args.json*-file which is described in the Section above.
The commands to remotely control the training process can be send to the bot. To change paramteres of the training process, a parameter from the *_args.json*-file should be inserted followed by a blank and the parameter value. For example, type *learning_rate 0.01* to set the learning rate to 0.01.
Multiple parameters can be specified in one message, too. The parameter name and its value should be seperated by a blank, e.g. *learning_rate 0.01 eps 0.03*. The remaining commands are listed below.

| Command | Description                         |
|---------|-------------------------------------|
| start   | Start the training process.         |
| stop    | Stop a training process.            |
| help    | Get a help message.                 |
| params  | Get the parameters of the training. |

It is recommended to use a http routing of the tensorboard to view the training progress remotely. We recommend to use [ngrok](https://ngrok.com/download) for this (see [manual](https://medium.com/samkirkiles/access-tensorboard-on-your-phone-from-anywhere-6c2eb7fa673e)). By using telegram in combination with ngrok, the training process can be remotely controlled.

### START OF THE TRAINING WITH POINT CLOUD DATA

We assume that the preprocessing steps in Section [PLY2PCD](#ply2pcd) and [CREATE CACHE](#create_cache) are executed. Or that alternatively the output of that processes is stored in the project. This can be the case if using Windows or Mac. Moreover, it is assumed that the [*_args.json*-file](#prerequisites_json) is set up. To create a new policy, see the [README.md](./policies/README.md) in the policies folder.

To start the training with the remote control, type *python remote_control.py --train_mode=train --train_args=ppo2_args.json --train_types=ppo2_types.json* in a terminal. The options can be printed with *python remote_control.py --help*. After starting the remote training process, the bot will react to the commands and the tensorboard can be routed with [ngrok](https://ngrok.com/download).

The training types can also be tested without the remote control. Type *python test_ppo2.py* to test the point cloud training with the PPO algorithm. Type *python test_imitation.py* to test the point cloud training with the imitation learning. Type *python test_gym_env.py* to test the PPO algorithm in combination with a gym environment.

### LOGS

The logs will be stored into the directory *./logs/environment_name/train/*. The environment_name is the name of the environment in which the agent has trained. The training progress will be written into a tensorboard file. To open the tensorboard, open a terminal in the logs-directory and type *tensorboard --logdir=.*. Each log-folder is specified by the date of its creation. There is a file with additional informations, e.g. the learning rate, in this folder. It is named *config.txt*.

## ACKNOWLEDGEMENTS

This project is sponsored by: German Federal Ministry of Education and Research (BMBF) under the project number 13FH022IX6. Project name: Interactive body-near production technology 4.0 (German: Interaktive körpernahe Produktionstechnik 4.0 (iKPT4.0))

![bmbf](figures/bmbflogo.jpg)

## CITATION

If you use this repository, please use the following citation:

@inproceedings{Tiator2020,

address = {Utrecht, Netherlands},

author = {Tiator, Marcel and Kerkmann, Anna and Geiger, Christian and Grimm, Paul},

booktitle = {Proceedings of the 3rd International Conference on Artificial Intelligence & Virtual Reality - AIVR '20},

publisher = {IEEE},

title = {{Using Semantic Segmentation to Assist the Creation of Interactive VR Applications}},

year = {2020}

}

## REFERENCES

[1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. Computing Research Repository (CoRR), jul 2017.

[2] Angela Dai, Angel X Chang, Manolis Savva, Maciej Halber, Thomas Funkhouser, and Matthias Niebner. ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes. In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition - CVPR ’17, Honolulu, Hawaii, 2017. IEEE.
