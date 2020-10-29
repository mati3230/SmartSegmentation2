from .agent_process import AgentProcess
import multiprocessing as mp
import threading


class MasterProcess():
    def __init__(
            self,
            n_process,
            n_samples,
            agent_type,
            agent_args,
            env_type,
            env_args,
            model_dir,
            model_filename,
            agent,
            agent_process_type,
            seed=42,
            shared_value=None,
            agent_process_args=None,
            async_mode=False,
            **kwargs):
        mp.set_start_method("spawn", force=True)
        self.processes = {}
        self.n_process = n_process
        assert(n_samples % n_process == 0)
        self.n_per_process = int(n_samples / n_process)
        self.agent_type = agent_type
        self.agent_args = agent_args
        self.env_type = env_type
        self.env_args = env_args
        self.model_dir = model_dir
        self.model_filename = model_filename
        self.update_f = None
        self.seed = seed
        self.shared_value = shared_value
        self.agent = agent
        self.agent_process_type = agent_process_type
        self.agent_process_args = agent_process_args
        self.async_mode = async_mode

    def sample_transitions(self):
        pipes = {}
        for i in range(self.n_process):
            parent_conn, child_conn = mp.Pipe()
            pipes[i] = parent_conn
            p = self.agent_process_type(
                conn=child_conn,
                id=i,
                n_cpus=self.n_process,
                n_steps=self.n_per_process,
                agent_type=self.agent_type,
                agent_args=self.agent_args,
                env_type=self.env_type,
                env_args=self.env_args,
                model_dir=self.model_dir,
                model_filename=self.model_filename,
                seed=self.seed,
                add_args=self.agent_process_args,
                async_mode=self.async_mode)
            p.start()
            self.processes[i] = p

        transitions = {}

        def listen_agent(id):
            while self.shared_value.value:
                msg = pipes[id].recv()
                transitions[id] = msg[0]

        threads_listen = []
        # print("Threads to start")
        for id in pipes:
            t = threading.Thread(target=listen_agent, args=(id, ))
            t.start()
            threads_listen.append(t)
        # print("Threads started")

        def send_msg_to_pipes(msg):
            for j in range(len(pipes)):
                pipes[j].send((msg, self.agent.get_vars()))

        def stop():
            for j in range(len(self.processes)):
                p = self.processes[j]
                pipes[j].close()
                p.terminate()
                # print("agent", j , "terminate")
                p.join()
                # print("agent", j, "join")

        if self.async_mode:
            print("master: send load")
            send_msg_to_pipes("load")
        while True:
            if(len(transitions) == self.n_process):
                if not self.shared_value.value:
                    send_msg_to_pipes("stop")
                    stop()
                    break
                if self.update_f:
                    self.update_f(transitions)
                transitions.clear()
                send_msg_to_pipes("load")
        print("done: master")
