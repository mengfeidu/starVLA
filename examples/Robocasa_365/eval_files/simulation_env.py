"""Run RoboCasa365 (PandaOmron) evaluation against a starVLA websocket policy.

Imports the existing wrappers from ``examples.Robocasa_tabletop`` to avoid
duplication. Differences vs. the GR1 tabletop runner:
  * uses upstream ``robocasa.wrappers.gym_wrapper`` (env id = ``robocasa/<TaskName>``)
  * single-arm 12-d action via ``model2robocasa365_interface.PolicyWarper``
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import time
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np

# Required for upstream robocasa env registration: ``robocasa/<TaskName>``
import robocasa  # noqa: F401
import robocasa.wrappers.gym_wrapper  # noqa: F401  (registers envs)
import robosuite  # noqa: F401
import tyro

from examples.Robocasa_365.eval_files.model2robocasa365_interface import PolicyWarper
from examples.Robocasa_tabletop.eval_files.wrappers.multistep_wrapper import MultiStepWrapper
from examples.Robocasa_tabletop.eval_files.wrappers.video_recording_wrapper import (
    VideoRecorder,
    VideoRecordingWrapper,
)

TARGET_TASK_NAMES = [
    "CloseBlenderLid",
    "CloseFridge",
    "CloseToasterOvenDoor",
    "CoffeeSetupMug",
    "NavigateKitchen",
    "OpenCabinet",
    "OpenDrawer",
    "OpenStandMixerHead",
    "PickPlaceCounterToCabinet",
    "PickPlaceCounterToStove",
    "PickPlaceDrawerToCounter",
    "PickPlaceSinkToCounter",
    "PickPlaceToasterToCounter",
    "SlideDishwasherRack",
    "TurnOffStove",
    "TurnOnElectricKettle",
    "TurnOnMicrowave",
    "TurnOnSinkFaucet",
    "ArrangeBreadBasket",
    "ArrangeTea",
    "BreadSelection",
    "CategorizeCondiments",
    "CuttingToolSelection",
    "DeliverStraw",
    "GarnishPancake",
    "GatherTableware",
    "GetToastedBread",
    "HeatKebabSandwich",
    "KettleBoiling",
    "LoadDishwasher",
    "MakeIceLemonade",
    "PackIdenticalLunches",
    "PanTransfer",
    "PortionHotDogs",
    "PreSoakPan",
    "PrepareCoffee",
    "RecycleBottlesByType",
    "RinseSinkBasin",
    "ScrubCuttingBoard",
    "SearingMeat",
    "SeparateFreezerRack",
    "SetUpCuttingStation",
    "StackBowlsCabinet",
    "SteamInMicrowave",
    "StirVegetables",
    "StoreLeftoversInBowl",
    "WaffleReheat",
    "WashFruitColander",
    "WashLettuce",
    "WeighIngredients",
]


@dataclass
class VideoConfig:
    video_dir: Optional[str] = None
    steps_per_render: int = 2
    fps: int = 20
    codec: str = "h264"
    input_pix_fmt: str = "rgb24"
    crf: int = 22
    thread_type: str = "FRAME"
    thread_count: int = 1


@dataclass
class MultiStepConfig:
    video_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    state_delta_indices: np.ndarray = field(default_factory=lambda: np.array([0]))
    n_action_steps: int = 8
    max_episode_steps: int = 500


@dataclass
class SimulationConfig:
    env_name: str
    n_episodes: int = 5
    n_envs: int = 1
    video: VideoConfig = field(default_factory=VideoConfig)
    multistep: MultiStepConfig = field(default_factory=MultiStepConfig)


def _create_single_env(config: SimulationConfig, idx: int) -> gym.Env:
    env = gym.make(config.env_name, enable_render=True, split="target")
    if config.video.video_dir is not None:
        video_recorder = VideoRecorder.create_h264(
            fps=config.video.fps,
            codec=config.video.codec,
            input_pix_fmt=config.video.input_pix_fmt,
            crf=config.video.crf,
            thread_type=config.video.thread_type,
            thread_count=config.video.thread_count,
        )
        env = VideoRecordingWrapper(
            env,
            video_recorder,
            video_dir=Path(config.video.video_dir),
            steps_per_render=config.video.steps_per_render,
        )
    env = MultiStepWrapper(
        env,
        video_delta_indices=config.multistep.video_delta_indices,
        state_delta_indices=config.multistep.state_delta_indices,
        n_action_steps=config.multistep.n_action_steps,
        max_episode_steps=config.multistep.max_episode_steps,
    )
    return env


def run_simulation(model: PolicyWarper, config: SimulationConfig) -> Tuple[str, List[bool]]:
    print(f"[robocasa365] running {config.n_episodes} eps on {config.env_name}")
    env_fns = [partial(_create_single_env, config=config, idx=i) for i in range(config.n_envs)]
    env = (
        gym.vector.SyncVectorEnv(env_fns)
        if config.n_envs == 1
        else gym.vector.AsyncVectorEnv(env_fns, shared_memory=False, context="spawn")
    )

    completed = 0
    episode_successes: List[bool] = []
    current_successes = [False] * config.n_envs
    obs, _ = env.reset()
    t0 = time.time()
    while completed < config.n_episodes:
        out = model.step(obs)
        actions = out["actions"] if "actions" in out else out
        next_obs, rewards, terminations, truncations, env_infos = env.step(actions)
        for i in range(config.n_envs):
            current_successes[i] |= bool(env_infos["success"][i][0])
            if terminations[i] or truncations[i]:
                episode_successes.append(current_successes[i])
                current_successes[i] = False
                completed += 1
        obs = next_obs

    try:
        env.close()
    except Exception as e:  # noqa: BLE001
        print(f"[robocasa365] env.close ignored: {e}")
    print(f"[robocasa365] {completed} eps done in {time.time() - t0:.1f}s")
    return config.env_name, episode_successes


def _parse_task_ids(task_ids: str) -> List[int]:
    if not task_ids.strip():
        return []
    ids = [int(x) for x in task_ids.replace(",", " ").split()]
    invalid = [x for x in ids if x < 0 or x >= len(TARGET_TASK_NAMES)]
    if invalid:
        raise ValueError(f"task_ids {invalid} out of range [0, {len(TARGET_TASK_NAMES)})")
    return ids


def _video_dir_for_task(video_out_path: Optional[str], task_name: str) -> Optional[str]:
    if video_out_path is None:
        return None
    return str(Path(video_out_path) / task_name)


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 5678
    resize_size: tuple = (224, 224)
    env_name: str = "robocasa/OpenDrawer"
    task_ids: str = ""
    n_episodes: int = 5
    n_envs: int = 1
    max_episode_steps: int = 500
    n_action_steps: int = 8
    video_out_path: Optional[str] = "results/robocasa365_eval_test/videos"
    seed: int = 7
    pretrained_path: str = (
        "playground/Checkpoints/robocasa365_qwenoft_OpenDrawer_100step/checkpoints/steps_100_pytorch_model.pt"
    )
    unnorm_key: Optional[str] = None
    result_json: str = ""


def main(args: Args) -> None:
    logging.info(json.dumps(dataclasses.asdict(args), indent=2, default=str))
    model = PolicyWarper(
        policy_ckpt_path=args.pretrained_path,
        unnorm_key=args.unnorm_key,
        host=args.host,
        port=args.port,
        image_size=args.resize_size,
        n_action_steps=args.n_action_steps,
    )
    task_ids = _parse_task_ids(args.task_ids)
    env_names = [f"robocasa/{TARGET_TASK_NAMES[i]}" for i in task_ids] if task_ids else [args.env_name]

    task_results = []
    total_successes = 0
    total_episodes = 0
    for env_name in env_names:
        task_name = env_name.split("/", 1)[-1]
        cfg = SimulationConfig(
            env_name=env_name,
            n_episodes=args.n_episodes,
            n_envs=args.n_envs,
            video=VideoConfig(video_dir=_video_dir_for_task(args.video_out_path, task_name)),
            multistep=MultiStepConfig(
                n_action_steps=args.n_action_steps, max_episode_steps=args.max_episode_steps
            ),
        )
        name, successes = run_simulation(model, cfg)
        num_successes = int(sum(successes))
        num_episodes = len(successes)
        sr = float(np.mean(successes)) if successes else 0.0
        total_successes += num_successes
        total_episodes += num_episodes
        task_results.append(
            {
                "env": name,
                "success_rate": sr,
                "total_episodes": num_episodes,
                "total_successes": num_successes,
                "successes": [bool(s) for s in successes],
            }
        )
        print(f"\n=== {name} ===")
        print(f"success rate: {sr:.2f} ({num_successes}/{num_episodes})")

    overall_sr = (total_successes / total_episodes) if total_episodes else 0.0
    summary = {
        "task_ids": task_ids,
        "env_names": env_names,
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "success_rate": overall_sr,
        "task_results": task_results,
    }

    out_path = Path(args.result_json) if args.result_json else None
    if out_path is None:
        out_dir = Path(args.pretrained_path).with_suffix(".eval")
        out_path = out_dir / f"{'_'.join(name.replace('/', '_') for name in env_names)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"[robocasa365] wrote result summary to {out_path}")


if __name__ == "__main__":
    tyro.cli(main)
