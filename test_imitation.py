from optimization.remote_imitation_trainer import RemoteImitationTrainer


if __name__ == "__main__":
    rem_trainer = RemoteImitationTrainer("./sample_args/imitation_args.json", "imitation_types.json")
    msg = rem_trainer.execute_command("start")
    if msg == -1:
        raise Exception("Cannot start training.")
