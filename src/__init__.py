import gymnasium as gym
import platform
machine = platform.machine()
system = platform.system()

# Use brax on apple silicon or linux x86
if machine == "arm64" and system == "Darwin" or \
    machine == "x86_64" and system == "Linux": 
    
    gym.register(
        id="brax-swimmer",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "swimmer",
            "episode_length": 1024,
            "action_repeat": 1,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-4,
            "reset_noise_scale": 0.1,
            "exclude_current_positions_from_observation": True,
            "legacy_reward": False,
            "legacy_spring": False,
        }
    )

    gym.register(
        id="brax-humanoid-standup",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "humanoidstandup",
            "episode_length": 1024,
            "action_repeat": 1,
            "legacy_spring": False
        }
    )

    gym.register(
        id="brax-ant",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "ant",
            "episode_length": 1024,
            "healthy_reward": 1.0,
            "ctrl_cost_weight": 0.5,
            "contact_cost_weight": 1e-4,
            "use_contact_forces": True,
            "terminate_when_unhealthy": True,
            "exclude_current_positions_from_observation": True,
            "action_repeat": 1
        }
    )

    gym.register(
        id="brax-hopper",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "hopper",
            "episode_length": 1024,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-3,
            "healthy_reward": 1.0,
            "terminate_when_unhealthy": True,
            "reset_noise_scale": 5e-3,
            "exclude_current_positions_from_observation": True,
            "action_repeat": 1,
            "legacy_spring": False,
            "healthy_z_range": (.7, float('inf'))
        }
    )

    gym.register(
        id="half-cheetah-hopper",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "hopper",
            "episode_length": 1024,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-3,
            "healthy_reward": 1.0,
            "terminate_when_unhealthy": True,
            "reset_noise_scale": 5e-3,
            "exclude_current_positions_from_observation": True,
            "action_repeat": 1,
            "healthy_z_range": (.7, float('inf'))
        }
    )

    gym.register(
        id="brax-humanoid",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "humanoid",
            "episode_length": 1024,
            "ctrl_cost_weight": 0.1,
            "forward_reward_weight": 1.25,
            "healthy_reward": 5.0,
            "terminate_when_unhealthy": True,
            "reset_noise_scale": 1e-2,
            "exclude_current_positions_from_observation": True,
            "action_repeat": 1,
            
        }
    )

    gym.register(
        id="brax-half-cheetah",
        entry_point="src.brax_to_gymnasium:convert_brax_to_gym",
        kwargs={
            "name": "halfcheetah",
            "episode_length": 1024,
            "forward_reward_weight": 1.0,
            "ctrl_cost_weight": 1e-3,
            "legacy_spring" : False,
            "reset_noise_scale": 5e-3,
            "exclude_current_positions_from_observation": True,
            "action_repeat": 1
        }
    )
