from __future__ import annotations

import sys

from train_n_step_td3_agent import main


if __name__ == "__main__":
    main(
        [
            "--sensor-only",
            "--sensor-clean-observation-v3",
            "--observation-noise-std",
            "0",
            *sys.argv[1:],
        ]
    )
