"""ActionEffect — data config that also loads the future observation o_{t+H}.

Identical to LIBERO's ``Libero4in1DataConfig`` except the *video* modality samples
two frames per item: the current frame (delta 0) and the frame after the action
chunk (delta = ACTION_HORIZON). The Stage-1 tokenizer uses both to form the
visual-effect target Δf; the Stage-2 ``QwenEffect`` framework consumes the future
frame (guarded by ``include_future_obs`` in the dataloader) to compute target
effect tokens on the fly.
"""

from starVLA.dataloader.gr00t_lerobot.datasets import ModalityConfig
from starVLA.dataloader.gr00t_lerobot.transform.base import ComposedModalityTransform
from starVLA.dataloader.gr00t_lerobot.transform.state_action import StateActionToTensor, StateActionTransform
from starVLA.dataloader.gr00t_lerobot.embodiment_tags import EmbodimentTag

ACTION_HORIZON = 8  # H — keep in sync with framework.action_model.action_horizon


class EffectLibero4in1DataConfig:
    embodiment_tag = EmbodimentTag.FRANKA
    video_keys = ["video.primary_image", "video.wrist_image"]
    state_keys = [
        "state.x", "state.y", "state.z", "state.roll",
        "state.pitch", "state.yaw", "state.pad", "state.gripper",
    ]
    action_keys = [
        "action.x", "action.y", "action.z", "action.roll",
        "action.pitch", "action.yaw", "action.gripper",
    ]
    language_keys = ["annotation.human.action.task_description"]

    # KEY DIFFERENCE: also sample the post-chunk frame for the effect target.
    observation_indices = [0, ACTION_HORIZON]
    action_indices = list(range(ACTION_HORIZON))
    state_indices = [0]

    def modality_config(self):
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.state_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=[0], modality_keys=self.language_keys),
        }

    def transform(self):
        return ComposedModalityTransform(transforms=[
            StateActionToTensor(apply_to=self.action_keys),
            StateActionTransform(
                apply_to=self.action_keys,
                normalization_modes={
                    "action.x": "min_max",
                    "action.y": "min_max",
                    "action.z": "min_max",
                    "action.roll": "min_max",
                    "action.pitch": "min_max",
                    "action.yaw": "min_max",
                },
            ),
        ])


ROBOT_TYPE_CONFIG_MAP = {
    "libero_franka_effect": EffectLibero4in1DataConfig(),
}

ROBOT_TYPE_TO_EMBODIMENT_TAG = {}

DATASET_NAMED_MIXTURES = {
    "libero_all_effect": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
    ],
    "libero_goal_effect": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
    ],
    "libero_90_effect": [
        ("libero_90_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
    ],
    "libero_all_plus_90_effect": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
        ("libero_90_no_noops_1.0.0_lerobot", 1.0, "libero_franka_effect"),
    ],
}
