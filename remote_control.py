from telegram.ext import Updater
import logging
from telegram.ext import MessageHandler, Filters
from optimization.remote_trainer import RemoteTrainer
from optimization.remote_pretrainer import RemotePretrainer
from optimization.remote_imitation_trainer import RemoteImitationTrainer
import argparse


parser = argparse.ArgumentParser()
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
args = parser.parse_args()
if args.train_mode == "train":
    rem_trainer = RemoteTrainer(args.train_args, args.train_types)
elif args.train_mode == "imitation":
    rem_trainer = RemoteImitationTrainer(args.train_args, args.train_types)
else:
    rem_trainer = RemotePretrainer(args.train_args, args.train_types)


def on_msg_recv(update, context):
    # print(update.message.text)
    global rem_trainer
    if not update.message:
        return
    msg = rem_trainer.execute_command(update.message.text)
    if msg == -1:
        return
    context.bot.send_message(
        chat_id=update.effective_chat.id,
        text=msg)


def main():
    updater = Updater(
        token=rem_trainer.params["token"],
        use_context=True)
    dispatcher = updater.dispatcher
    logging.basicConfig(
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        level=logging.INFO)

    msg_handler = MessageHandler(Filters.text, on_msg_recv)
    dispatcher.add_handler(msg_handler)

    updater.start_polling()


if __name__ == "__main__":
    main()
