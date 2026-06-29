import subprocess
import sys
import time
import ctypes
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
GYM_TORCS_DIR = ROOT_DIR / "torcs-wrapper" / "gym_torcs"

sys.path.append(str(GYM_TORCS_DIR))

from gym_torcs import TorcsEnv  # type: ignore[import]


class TorcsRunner:
    OFF_TRACK_SHUTDOWN_DELAY = 3.0
    STARTUP_DELAY = 8.0

    def __init__(self):
        self.env: Optional[TorcsEnv] = None
        self.torcs_process: Optional[subprocess.Popen] = None
        self.torcs_path = ROOT_DIR / "torcs" / "wtorcs.exe"

    def launch(self):
        print("\nLaunching TORCS...")

        if not self.torcs_path.is_file():
            raise FileNotFoundError(f"TORCS executable not found: {self.torcs_path}")

        if self.torcs_process is not None and self.torcs_process.poll() is None:
            return

        # This Windows build does not support the Linux `-r <race.xml>` option.
        # Passing it while forcing CREATE_NEW_CONSOLE leaves only the black
        # console visible. CREATE_NO_WINDOW hides that console without hiding
        # the separate TORCS OpenGL application window.
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        self.torcs_process = subprocess.Popen(
            [str(self.torcs_path)],
            cwd=str(self.torcs_path.parent),
            creationflags=creation_flags,
        )

        # Give TORCS time to open its graphical window and UDP socket.
        time.sleep(self.STARTUP_DELAY)

        return_code = self.torcs_process.poll()
        if return_code is not None:
            self.torcs_process = None
            raise RuntimeError(f"TORCS closed during startup (exit code {return_code})")

        try:
            self._start_practice_race()
        except Exception:
            self.shutdown()
            raise

    def _start_practice_race(self):
        if sys.platform != "win32" or self.torcs_process is None:
            return

        user32 = ctypes.windll.user32
        window_handles = []
        enum_windows_callback = ctypes.WINFUNCTYPE(
            wintypes.BOOL,
            wintypes.HWND,
            wintypes.LPARAM,
        )

        @enum_windows_callback
        def find_torcs_window(window_handle, _parameter):
            process_id = wintypes.DWORD()
            user32.GetWindowThreadProcessId(window_handle, ctypes.byref(process_id))

            class_name = ctypes.create_unicode_buffer(256)
            user32.GetClassNameW(window_handle, class_name, len(class_name))

            if (
                process_id.value == self.torcs_process.pid
                and user32.IsWindowVisible(window_handle)
                and class_name.value != "ConsoleWindowClass"
            ):
                window_handles.append(window_handle)
                return False

            return True

        user32.EnumWindows(find_torcs_window, 0)

        if not window_handles:
            raise RuntimeError("TORCS started, but its graphical window was not found")

        # This is the Windows equivalent of gym_torcs/autostart.sh. It moves
        # through TORCS's menus and starts the configured SCR practice race.
        wm_key_down = 0x0100
        wm_key_up = 0x0101
        vk_return = 0x0D
        vk_up = 0x26

        for key in (vk_return, vk_return, vk_up, vk_up, vk_return, vk_return):
            user32.PostMessageW(window_handles[0], wm_key_down, key, 0)
            user32.PostMessageW(window_handles[0], wm_key_up, key, 0)
            time.sleep(0.25)

    def shutdown(self):
        process = self.torcs_process
        self.torcs_process = None

        if process is None or process.poll() is not None:
            return

        print("Closing TORCS...")
        process.terminate()

        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)

    def connect(self):
        print("Connecting to TORCS...")

        self.env = TorcsEnv(
            vision=False,
            throttle=True,
            gear_change=False,
        )

    def load_track(self, track_name):
        print(f"Loading track: {track_name}")

        # The selected track is currently configured in practice.xml.

    def run(self, agent):
        assert self.env is not None, "Call connect() before run()"
        started_at = datetime.now(timezone.utc).isoformat()
        started_timer = time.perf_counter()
        observation = self.env.reset(relaunch=False)

        agent.reset()

        max_steps = 10000
        stuck_counter = 0

        results = {
            "started_at": started_at,
            "steps": 0,
            "total_score": 0.0,
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "off_track": 0,
            "duration_seconds": 0.0,
            "termination_reason": "max_steps",
        }

        speed_sum = 0.0
        previous_damage = float(self.env.client.S.d.get("damage", 0.0))

        try:
            for _ in range(max_steps):
                if self.torcs_process is not None and self.torcs_process.poll() is not None:
                    results["termination_reason"] = "torcs_closed"
                    print("\nTORCS closed unexpectedly.")
                    break

                action = np.array(agent.act(observation))
                observation, reward, done, _info = self.env.step(action)

                # speedX is normalised by default_speed (50) inside gym_torcs.
                speed = float(observation.speedX)
                off_track = bool(observation.track.min() < 0)
                damage = float(self.env.client.S.d.get("damage", previous_damage))
                crashed = damage > previous_damage
                previous_damage = damage

                speed_sum += speed
                results["total_score"] += float(reward)
                results["steps"] += 1

                if speed > results["max_speed"]:
                    results["max_speed"] = speed

                if off_track:
                    results["off_track"] += 1
                    results["termination_reason"] = "off_track"
                    print(
                        "\nAgent went off track. "
                        f"Closing TORCS in {self.OFF_TRACK_SHUTDOWN_DELAY:g} seconds..."
                    )
                    time.sleep(self.OFF_TRACK_SHUTDOWN_DELAY)
                    break

                if crashed:
                    results["termination_reason"] = "crashed"
                    print("\nAgent crashed.")
                    break

                # 0.1 normalised speed is 5 km/h.
                if speed < 0.1:
                    stuck_counter += 1
                else:
                    stuck_counter = 0

                if stuck_counter > 100:
                    results["termination_reason"] = "stuck"
                    print("\nAgent became stuck.")
                    break

                if done:
                    results["termination_reason"] = "finished"
                    print("\nRace finished.")
                    break
        finally:
            results["avg_speed"] = speed_sum / max(results["steps"], 1)
            results["duration_seconds"] = time.perf_counter() - started_timer
            self.env.end()
            self.shutdown()

        return results
