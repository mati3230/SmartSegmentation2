from optimization.remote_ppo2_trainer import RemotePPO2Trainer


if __name__ == "__main__":
    rem_trainer = RemotePPO2Trainer("./sample_args/ppo2_args.json", "ppo2_types.json")
    msg = rem_trainer.execute_command("start")
    if msg == -1:
        raise Exception("Cannot start training.")
