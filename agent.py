from environment.segmentation_ext import PCLCloud
import tensorflow as tf
from optimization.remote_trainer import RemoteTrainer
from optimization.remote_pretrainer import RemotePretrainer
from optimization.remote_imitation_trainer import RemoteImitationTrainer
from optimization.utils import get_type
from environment.utils import mkdir
import argparse
import numpy as np
import cv2

def main():
    """Test a trained agent."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="model",
        help="name of the model you want to load from models directory")
    parser.add_argument(
        "--train_mode",
        type=str,
        default="train",
        help="options: train, pretrain, imitation")
    parser.add_argument(
        "--train_args",
        type=str,
        default="ppo2_args.json",
        help="file (relative path) to default configuration (e.g. ppo2_args.json or dagger_args.json)")
    parser.add_argument(
        "--train_types",
        type=str,
        default="ppo2_types.json",
        help="file (relative path) to types that should match train_args (e.g. ppo2_types.json or dagger_types.json)")
    parser.add_argument(
        "--train_scene",
        type=bool,
        default=False,
        help="If true, a training scene will be chosen")
    parser.add_argument(
        "--render_suggestions",
        type=bool,
        default=False,
        help="If true, a the grouping suggestions will be rendered.")
    parser.add_argument(
        "--render_size",
        type=int,
        default=512,
        help="Size a rendered view.")
    parser.add_argument(
        "--render_save",
        type=bool,
        default=False,
        help="If true, rendered views will be saved to file.")
    parser.add_argument(
        "--render_path",
        type=str,
        default="./render",
        help="path to store rendered images.")
    parser.add_argument(
        "--render_steps",
        type=bool,
        default=False,
        help="render images stepwise.")

    args = parser.parse_args()

    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices("GPU")
            print(
                len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    net_mode = "full"
    if args.train_mode == "train":
        remote_trainer = RemoteTrainer(args.train_args, args.train_types)
    elif args.train_mode == "imitation":
        remote_trainer = RemoteImitationTrainer(args.train_args, args.train_types)
        net_mode = "half"
    else:
        remote_trainer = RemotePretrainer(args.train_args, args.train_types)

    env_type = get_type(
        remote_trainer.params["env_path"], remote_trainer.params["env_type"])

    env_args = {}
    if remote_trainer.params["data_provider_path"]:
        if remote_trainer.params["data_provider_path"] != "":
            data_prov_type = get_type(
                remote_trainer.params["data_provider_path"], "DataProvider")
            env_args = {
                "data_prov_type": data_prov_type,
                "max_scenes": remote_trainer.params["max_scenes"],
                "train_mode": args.train_mode
            }
    env = env_type(**env_args)

    state_size = remote_trainer.params["state_size"]
    n_actions = remote_trainer.params["n_actions"]
    policy_args = {
        "name": "target_policy",
        "n_ft_outpt": remote_trainer.params["n_ft_outpt"],
        "n_actions": n_actions,
        "state_size": state_size,
        "seed": remote_trainer.params["seed"],
        "stddev": remote_trainer.params["stddev"],
        "initializer": remote_trainer.params["initializer"],
        "stateful": True,
        "trainable": True,
        "mode": net_mode}
    policy_type = get_type(
        remote_trainer.params["policy_path"],
        remote_trainer.params["policy_type"])
    policy = policy_type(**policy_args)

    model_dir = "./models/" + remote_trainer.params["env_name"] + "/" + remote_trainer.params["model_name"]
    model = args.model

    # sample computation to force var initialization
    data = np.zeros(state_size)
    policy.action(data)
    policy.reset()

    print("Load model '", model, "'...")
    policy.load(model_dir, model)
    print("Success")

    state = env.reset(train=args.train_scene)
    episode_reward = 0

    # env.render(r_segments=False, animate=True)
    training = False
    step = 0

    scale = args.render_size / state_size[1]

    if args.render_save:
        mkdir(args.render_path)

    while(True):
        m_state = policy.preprocess(state)
        if args.render_suggestions:
            im1 = np.concatenate((m_state[0], m_state[1]), axis=0)
            im2 = np.concatenate((m_state[2], m_state[3]), axis=0)
            im = np.concatenate((im1, im2), axis=1)
            width = int(im.shape[1] * scale)
            height = int(im.shape[0] * scale)
            dim = (width, height)
            im = cv2.resize(im, dim)
            cv2.imshow("Views", im)
            if args.render_steps:
                # waits until a key is pressed
                cv2.waitKey(0)
                # destroys the window showing image
                cv2.destroyAllWindows()
            else:
                if cv2.waitKey(30) & 0xFF == ord("q"):
                    break
            if args.render_save:
                im *= 255
                cv2.imwrite(args.render_path + "/obs_" + str(step) + ".jpg", im)
        pi_action = policy.action(m_state, training=training)
        action = pi_action["action"]
        action = policy.preprocess_action(action)
        # action = int(action)
        state, reward, done, info = env.step(action)
        episode_reward += reward
        step += 1
        # print(action, reward, done, info)
        # env.render()
        if done:
            print("done: ", episode_reward, "steps:", step, "Psi:", (episode_reward/step))
            env.render(r_segments=True, animate=True)
            episode_reward = 0
            step = 0
            state = env.reset(train=args.train_scene)
            policy.reset()
            # break
    cv2.destroyAllWindows()
    print("finish")


if __name__ == "__main__":
    main()
