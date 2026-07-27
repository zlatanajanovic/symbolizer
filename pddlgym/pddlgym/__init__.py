"""PDDLGym - Gym environments for PDDL planning domains.

Minimal build: only Blocksworld and Hanoi variants are registered.
"""

from . import core
from . import structs
from . import spaces

import matplotlib
from pddlgym.rendering import *
from gym.envs.registration import register
import gym

import os


def make(*args, **kwargs):
    """Create a PDDLGym environment (wraps gym.make with env checker disabled)."""
    return gym.make(*args, disable_env_checker=True, **kwargs)


def register_pddl_env(name, is_test_env, other_args):
    dir_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "pddl")
    domain_file = os.path.join(dir_path, "{}.pddl".format(name.lower()))
    gym_name = name.capitalize()
    problem_dirname = name.lower()
    if is_test_env:
        gym_name += 'Test'
        problem_dirname += '_test'
    problem_dir = os.path.join(dir_path, problem_dirname)

    register(
        id='PDDLEnv{}-v0'.format(gym_name),
        entry_point='pddlgym.core:PDDLEnv',
        kwargs=dict({'domain_file' : domain_file, 'problem_dir' : problem_dir,
                     **other_args}),
    )


# --- Only register Blocksworld and Hanoi variants (used in paper) ---
for env_name, kwargs in [
        ("blocks", {'render' : blocks_render}),
        ("blocks_operator_actions", {'render' : blocks_render,
                                     'operators_as_actions' : True,
                                     'dynamic_action_space' : True}),
        ("blocks_operator_actions_encoded", {'render' : blocks_render_encoded,
                                     'operators_as_actions' : True,
                                     'dynamic_action_space' : True}),
        ("blocks_medium", {'render' : blocks_render,
                           'operators_as_actions' : True,
                           'dynamic_action_space' : True}),
        ("hanoi", {'render' : hanoi_render}),
        ("hanoi_operator_actions", {'render' : hanoi_render,
                                    'operators_as_actions' : True,
                                    'dynamic_action_space' : True}),
        ("hanoi_color", {'render' : hanoi_color_render,
                         'operators_as_actions' : True,
                         'dynamic_action_space' : True}),
        ("hanoi_color_operator_actions", {'render' : hanoi_color_render,
                                          'operators_as_actions' : True,
                                          'dynamic_action_space' : True}),
]:
    other_args = {
        "raise_error_on_invalid_action": False,
    }
    kwargs.update(other_args)
    for is_test in [False, True]:
        register_pddl_env(env_name, is_test, kwargs)


__pdoc__ = {'downward_translate': False, 'procedural_generation': False}
