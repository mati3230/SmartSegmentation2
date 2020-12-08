# CREATION OF A POLICY
We split a policy into two parts. The first part is the feature detector. The second part is the head. The feature detector calculates a feature vector from the data. The head takes the feature vector and computes an action. Additional values such as the state value if using reinforcement learning is calculated. Hence, we use the following naming convention for the policy *mlp_k_ac.py*.

*  mlp: Multilayer Perceptron
*  k: Keras
*  ac: Actor Critic

The actor critic policy *mlp_k_ac.py* has two heads. One head for the action estimation. One head for the state value estimation. The feature detector for this policy is defined in *mlp_k.py*. To define a new policy, the base classes have to be inherited from the head and feature detector class. The feature detector should inherit from the *BaseNet* class which is defined in *optimization.base_net*. The head should inherit from the *ActorCritic* class which is defined in *optimization.actor_critic*. Note, that we only use the actor critic architecture as we only use the Proximal Policy Optimization (PPO) algorithm of Schulman et al. [1].

Currently, two modes for the head are defined. In *full* mode, the policy will estimate all head values. In *half* mode, the policy will only estimate the action. This can be useful if using imitation learning where only actions are imitated.

# REFERENCES
[1] John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. Computing Research Repository (CoRR), jul 2017.