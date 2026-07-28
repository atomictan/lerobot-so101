# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Helper to set motor ids and baudrate.

Example:

```shell
lerobot-setup-motors \
    --teleop.type=so100_leader \
    --teleop.port=/dev/tty.usbmodem575E0031751
```

To (re)configure a single motor instead of all of them, pass `--single_motor=true`. You will be
prompted to pick which motor to set up and which ID to assign it:

```shell
lerobot-setup-motors \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --single_motor=true
```
"""

from dataclasses import dataclass

import draccus

from lerobot.robots import (  # noqa: F401
    RobotConfig,
    bi_rebot_b601_follower,
    bi_so_follower,
    koch_follower,
    lekiwi,
    make_robot_from_config,
    omx_follower,
    rebot_b601_follower,
    so_follower,
)
from lerobot.teleoperators import (  # noqa: F401
    TeleoperatorConfig,
    bi_openarm_mini,
    bi_rebot_102_leader,
    bi_so_leader,
    koch_leader,
    make_teleoperator_from_config,
    omx_leader,
    openarm_mini,
    rebot_102_leader,
    so_leader,
)

COMPATIBLE_DEVICES = [
    "koch_follower",
    "koch_leader",
    "omx_follower",
    "omx_leader",
    "openarm_mini",
    "so100_follower",
    "so100_leader",
    "so101_follower",
    "so101_leader",
    "lekiwi",
]


@dataclass
class SetupConfig:
    teleop: TeleoperatorConfig | None = None
    robot: RobotConfig | None = None
    single_motor: bool = False

    def __post_init__(self):
        if bool(self.teleop) == bool(self.robot):
            raise ValueError("Choose either a teleop or a robot.")

        self.device = self.robot if self.robot else self.teleop


def _select_motor(motor_names: list[str]) -> str:
    print("Available motors:")
    for i, name in enumerate(motor_names, start=1):
        print(f"  {i}. {name}")

    while True:
        choice = input(f"Select a motor to set up [1-{len(motor_names)}]: ").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(motor_names):
            return motor_names[int(choice) - 1]
        print("Invalid selection, please try again.")


def _prompt_motor_id() -> int:
    while True:
        choice = input("Enter the desired motor ID: ").strip()
        if choice.isdigit():
            return int(choice)
        print("Invalid ID, please enter a positive integer.")


def setup_single_motor(device) -> None:
    motor = _select_motor(list(device.bus.motors))
    device.bus.motors[motor].id = _prompt_motor_id()
    input(f"Connect the controller board to the '{motor}' motor only and press enter.")
    device.bus.setup_motor(motor)
    print(f"'{motor}' motor id set to {device.bus.motors[motor].id}")


@draccus.wrap()
def setup_motors(cfg: SetupConfig):
    if cfg.device.type not in COMPATIBLE_DEVICES:
        raise NotImplementedError

    if isinstance(cfg.device, RobotConfig):
        device = make_robot_from_config(cfg.device)
    else:
        device = make_teleoperator_from_config(cfg.device)

    if cfg.single_motor:
        setup_single_motor(device)
    else:
        device.setup_motors()


def main():
    setup_motors()


if __name__ == "__main__":
    main()
