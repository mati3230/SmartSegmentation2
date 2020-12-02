from abc import ABC, abstractmethod


class BaseOptimization(ABC):
    def __init__(
            self,
            policy_type,
            n_actions,
            optimizer,
            train_summary_writer,
            batch_size=64,
            global_norm=0.5,
            check_numerics=False,
            write_tensorboard=True,
            **kwargs):
        """Constructor.

        Parameters
        ----------
        policy_type : type
            Type of the policy that should be trained to initialize the target
            policy.
        n_actions : int
            Number of available actions.
        optimizer : tf.keras.optimizers.Optimizer
            Optimizer such as SGD or ADAM.
        train_summary_writer : tf.summary.SummaryWriter
            Summary writer to write tensorboard logs.
        batch_size : int
            Number of samples in one batch.
        global_norm : float
            Threshold to clip the gradients according to a maximum global norm.
        check_numerics : boolean
            If True, an exception is thrown in case of NaN values.
        write_tensorboard : boolean
            If True, summary will be written.
        **kwargs : dict
            Additional input arguments.
        """
        super().__init__()
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.train_summary_writer = train_summary_writer
        self.global_norm = global_norm
        self.check_numerics = check_numerics
        self.n_actions = n_actions
        self.write_tensorboard = write_tensorboard

    @abstractmethod
    def get_online_policy(self):
        pass

    @abstractmethod
    def create_policies(
            self,
            policy_type,
            n_actions,
            check_numerics,
            **kwargs):
        pass

    @abstractmethod
    def update(self, buf, step):
        pass

    @abstractmethod
    def step_update(self, buf, step):
        pass
