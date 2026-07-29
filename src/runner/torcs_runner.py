import subprocess
import sys
import time
import ctypes
from collections.abc import Mapping
from ctypes import wintypes
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

ROOT_DIR = Path(__file__).resolve().parents[2]
GYM_TORCS_DIR = ROOT_DIR / "torcs-wrapper" / "gym_torcs"

sys.path.append(str(GYM_TORCS_DIR))

from gym_torcs import TorcsEnv  # type: ignore[import]

from runner.lap_tracker import LapTracker, practice_finish_is_plausible


class TorcsRunner:
    OFF_TRACK_SHUTDOWN_DELAY = 3.0
    STARTUP_DELAY = 8.0
    MENU_KEY_DELAY = 1.0
    MENU_ENTER_ATTEMPTS = 8
    SERVER_PORT = 3001
    SERVER_START_TIMEOUT = 25.0
    MANUAL_START_TIMEOUT = 120.0
    STUCK_SPEED_THRESHOLD = 0.08
    STUCK_STEP_LIMIT = 250
    STUCK_MIN_LAP_TIME_SECONDS = 8.0

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

        process = self.torcs_process
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
                process_id.value == process.pid
                and user32.IsWindowVisible(window_handle)
                and class_name.value != "ConsoleWindowClass"
            ):
                window_handles.append(window_handle)
                return False

            return True

        user32.EnumWindows(find_torcs_window, 0)

        if not window_handles:
            raise RuntimeError("TORCS started, but its graphical window was not found")

        vk_return = 0x0D
        key_event_type = 1
        key_up_flag = 0x0002

        class KeyboardInput(ctypes.Structure):
            _fields_ = (
                ("virtual_key", wintypes.WORD),
                ("scan_code", wintypes.WORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("extra_info", wintypes.WPARAM),
            )

        class MouseInput(ctypes.Structure):
            _fields_ = (
                ("x", wintypes.LONG),
                ("y", wintypes.LONG),
                ("mouse_data", wintypes.DWORD),
                ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD),
                ("extra_info", wintypes.WPARAM),
            )

        class HardwareInput(ctypes.Structure):
            _fields_ = (
                ("message", wintypes.DWORD),
                ("parameter_low", wintypes.WORD),
                ("parameter_high", wintypes.WORD),
            )

        class InputValue(ctypes.Union):
            _fields_ = (
                ("keyboard", KeyboardInput),
                ("mouse", MouseInput),
                ("hardware", HardwareInput),
            )

        class Input(ctypes.Structure):
            _fields_ = (
                ("input_type", wintypes.DWORD),
                ("value", InputValue),
            )

        def send_key(key, flags=0):
            value = InputValue()
            value.keyboard = KeyboardInput(key, 0, flags, 0, 0)
            input_event = Input(key_event_type, value)
            sent = user32.SendInput(1, ctypes.byref(input_event), ctypes.sizeof(Input))
            if sent != 1:
                raise RuntimeError("Windows could not send input to TORCS")

        window_handle = window_handles[0]
        target_thread = user32.GetWindowThreadProcessId(window_handle, None)
        current_thread = ctypes.windll.kernel32.GetCurrentThreadId()
        attached = bool(user32.AttachThreadInput(current_thread, target_thread, True))

        try:
            user32.ShowWindow(window_handle, 9)
            user32.BringWindowToTop(window_handle)
            user32.SetForegroundWindow(window_handle)
            user32.SetFocus(window_handle)
        finally:
            if attached:
                user32.AttachThreadInput(current_thread, target_thread, False)

        time.sleep(0.5)

        # Practice is first in the race list, producing a stable novice flow:
        # Main menu -> Race -> Practice -> New Race. TORCS sometimes restores
        # a slightly different menu depth, so keep advancing until the SCR
        # server socket appears rather than assuming exactly three presses.
        deadline = time.monotonic() + self.SERVER_START_TIMEOUT
        for _ in range(self.MENU_ENTER_ATTEMPTS):
            if process.poll() is not None:
                raise RuntimeError("TORCS closed before the SCR race started")
            if self._server_is_listening(process.pid):
                print("TORCS Practice race started automatically.")
                return

            send_key(vk_return)
            send_key(vk_return, key_up_flag)
            time.sleep(self.MENU_KEY_DELAY)

        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("TORCS closed before the SCR race started")
            if self._server_is_listening(process.pid):
                print("TORCS Practice race started automatically.")
                return
            time.sleep(0.5)

        self._wait_for_manual_practice_start(process)

    def _wait_for_manual_practice_start(self, process):
        print(
            "Automatic Practice startup did not open the SCR socket. "
            "TORCS is still running."
        )
        print("In TORCS, choose Race > Practice > New Race. Waiting for the race...")

        deadline = time.monotonic() + self.MANUAL_START_TIMEOUT
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("TORCS closed before the SCR race started")
            if self._server_is_listening(process.pid):
                print("TORCS Practice race detected.")
                return
            time.sleep(0.5)

        raise RuntimeError(
            "TORCS opened, but Practice did not start before the manual timeout"
        )

    def _server_is_listening(self, process_id):
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        try:
            result = subprocess.run(
                ["netstat", "-ano", "-p", "UDP"],
                capture_output=True,
                text=True,
                check=False,
                timeout=2,
                creationflags=creation_flags,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False

        if self._netstat_reports_server(result.stdout, self.SERVER_PORT, process_id):
            return True

        # Some Windows TORCS builds report the UDP socket under a helper process
        # rather than the OpenGL window process. If the expected PID check does
        # not match, still accept an open SCR server port; connect() will fail
        # quickly if it is stale or unrelated.
        return self._netstat_reports_server(result.stdout, self.SERVER_PORT)

    @staticmethod
    def _netstat_reports_server(output, port, process_id=None):
        expected_process_id = None if process_id is None else str(process_id)
        port_suffix = f":{port}"

        for line in output.splitlines():
            columns = line.split()
            if len(columns) < 3:
                continue

            local_address = columns[1]
            owner_process_id = columns[-1]
            if not local_address.endswith(port_suffix):
                continue
            if expected_process_id is None or owner_process_id == expected_process_id:
                return True

        return False

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

    def _step_full_control_agent(self, action):
        """Apply a legacy controller action without gym's automatic controls."""
        assert self.env is not None
        client = self.env.client
        previous_damage = float(client.S.d.get("damage", 0.0))

        client.R.d["steer"] = action["steer"]
        client.R.d["accel"] = action["accel"]
        client.R.d["brake"] = action["brake"]
        client.R.d["gear"] = action["gear"]
        client.R.d["meta"] = 0
        client.respond_to_server()
        client.get_servers_input()

        raw_observation = client.S.d
        server_stopped = client.so is None
        self.env.time_step += 1

        reward = float(raw_observation["speedX"]) * float(
            np.cos(raw_observation["angle"])
        )
        if float(raw_observation.get("damage", 0.0)) > previous_damage:
            reward = -1.0
        return raw_observation, reward, server_stopped, {}

    def run(
        self,
        agent,
        *,
        stop_requested=None,
        telemetry_callback=None,
        telemetry_interval_steps=1,
        telemetry_interval_seconds=None,
    ):
        assert self.env is not None, "Call connect() before run()"
        started_at = datetime.now(timezone.utc).isoformat()
        started_timer = time.perf_counter()
        observation = self.env.reset(relaunch=False)
        telemetry_interval_seconds = self._normalise_telemetry_interval_seconds(
            telemetry_interval_seconds
        )
        next_telemetry_sample_at = started_timer

        agent.reset()

        max_steps = getattr(agent, "max_steps", 10000)
        target_laps = getattr(agent, "target_laps", None)
        stuck_counter = 0
        uses_full_control = bool(getattr(agent, "uses_full_control", False))
        initial_telemetry = dict(self.env.client.S.d)
        lap_tracker = LapTracker(initial_telemetry.get("lastLapTime", 0.0))

        results = {
            "started_at": started_at,
            "steps": 0,
            "total_score": 0.0,
            "max_speed": 0.0,
            "avg_speed": 0.0,
            "off_track": 0,
            "laps_completed": 0,
            "best_lap_time_seconds": None,
            "average_lap_time_seconds": None,
            "duration_seconds": 0.0,
            "termination_reason": "max_steps",
            "telemetry_samples": [],
        }

        speed_sum = 0.0
        previous_damage = float(self.env.client.S.d.get("damage", 0.0))

        try:
            for _ in range(max_steps):
                if stop_requested is not None and stop_requested():
                    results["termination_reason"] = "stopped"
                    break

                if self.torcs_process is not None and self.torcs_process.poll() is not None:
                    results["termination_reason"] = "torcs_closed"
                    print("\nTORCS closed unexpectedly.")
                    break

                raw_telemetry = self.env.client.S.d
                agent_action = agent.act(observation, raw_telemetry)
                if uses_full_control:
                    if not isinstance(agent_action, Mapping):
                        raise ValueError("Full-control agents must return an action mapping")
                    if agent_action.get("terminate", False):
                        self.env.client.R.d.update(
                            {
                                "steer": agent_action["steer"],
                                "accel": agent_action["accel"],
                                "brake": agent_action["brake"],
                                "gear": agent_action["gear"],
                                "meta": 1,
                            }
                        )
                        self.env.client.respond_to_server()
                        results["termination_reason"] = agent_action[
                            "termination_reason"
                        ]
                        break
                    raw_observation, reward, done, _info = self._step_full_control_agent(
                        agent_action
                    )
                else:
                    action = np.asarray(agent_action, dtype=float)
                    if action.shape not in [(2,), (3,)]:
                        raise ValueError(
                            "Agent actions must be [steering, throttle] or "
                            "[steering, throttle, brake]"
                        )
                    brake = float(action[2]) if action.shape == (3,) else 0.0
                    self.env.client.R.d["brake"] = float(
                        np.clip(brake, 0.0, 1.0)
                    )
                    observation, reward, done, _info = self.env.step(action[:2])

                if uses_full_control:
                    # Keep this path as light as the legacy socket loop. Building
                    # gym's full NumPy observation every frame can leave control
                    # commands queued behind newer TORCS sensor packets.
                    speed = float(raw_observation["speedX"]) / self.env.default_speed
                    off_track = min(raw_observation["track"]) < 0
                    action_snapshot = agent_action
                else:
                    # speedX is normalised by default_speed (50) inside gym_torcs.
                    speed = float(observation.speedX)
                    off_track = bool(observation.track.min() < 0)
                    action_snapshot = {
                        "steer": float(action[0]),
                        "accel": float(action[1]),
                        "brake": brake,
                        "gear": self.env.client.R.d.get("gear"),
                    }
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

                now = time.perf_counter()
                should_sample_telemetry = self._should_emit_telemetry_sample(
                    results["steps"],
                    telemetry_interval_steps,
                    now=now,
                    next_sample_at=(
                        next_telemetry_sample_at
                        if telemetry_interval_seconds is not None
                        else None
                    ),
                )
                if should_sample_telemetry:
                    self._record_telemetry_sample(
                        results,
                        self.env.client.S.d,
                        action_snapshot,
                        reward,
                        off_track,
                    )
                    telemetry_sample = results["telemetry_samples"][-1]
                    if telemetry_callback is not None:
                        telemetry_callback(telemetry_sample)
                    if telemetry_interval_seconds is not None:
                        next_telemetry_sample_at = now + telemetry_interval_seconds

                completed_lap = lap_tracker.update(self.env.client.S.d)
                if (
                    completed_lap is None
                    and done
                    and uses_full_control
                    and practice_finish_is_plausible(
                        initial_telemetry,
                        self.env.client.S.d,
                        agent,
                    )
                ):
                    # Practice mode closes the SCR socket at the timing line
                    # without populating lastLapTime. The final curLapTime is
                    # the official completed Practice time in that case.
                    completed_lap = lap_tracker.record(
                        self.env.client.S.d.get("curLapTime", 0.0)
                    )
                if completed_lap is not None:
                    results["laps_completed"] = lap_tracker.laps_completed
                    results["best_lap_time_seconds"] = lap_tracker.best_lap_time
                    results["average_lap_time_seconds"] = lap_tracker.average_lap_time
                    print(
                        f"\nLap {lap_tracker.laps_completed} completed in "
                        f"{completed_lap:.3f} seconds."
                    )
                    if target_laps and lap_tracker.laps_completed >= target_laps:
                        results["termination_reason"] = "target_laps_completed"
                        break

                if self._is_stuck_speed(speed):
                    stuck_counter += 1
                else:
                    stuck_counter = 0

                if self._should_stop_for_stuck(
                    stuck_counter,
                    self._lap_time_seconds(self.env.client.S.d),
                ):
                    results["termination_reason"] = "stuck"
                    print("\nAgent became stuck.")
                    break

                if uses_full_control:
                    if done:
                        results["termination_reason"] = "race_finished"
                        break
                    continue

                if off_track:
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

                if done:
                    results["termination_reason"] = "finished"
                    print("\nRace finished.")
                    break
        finally:
            results["avg_speed"] = speed_sum / max(results["steps"], 1)
            results["duration_seconds"] = time.perf_counter() - started_timer
            close_agent = getattr(agent, "close", None)
            if close_agent is not None:
                close_agent()
            self.env.end()
            self.shutdown()

        return results

    @staticmethod
    def _should_emit_telemetry_sample(
        step,
        interval_steps=None,
        *,
        now=None,
        next_sample_at=None,
    ):
        if step == 1:
            return True

        if next_sample_at is not None:
            if now is None:
                raise ValueError("now is required for time-based telemetry sampling")
            return now >= next_sample_at

        interval = max(1, int(interval_steps or 1))
        return step % interval == 0

    @staticmethod
    def _normalise_telemetry_interval_seconds(interval_seconds):
        if interval_seconds is None:
            return None
        return max(0.0, float(interval_seconds))

    @classmethod
    def _is_stuck_speed(cls, speed):
        return speed < cls.STUCK_SPEED_THRESHOLD

    @classmethod
    def _should_stop_for_stuck(cls, stuck_counter, lap_time_seconds):
        return (
            stuck_counter > cls.STUCK_STEP_LIMIT
            and lap_time_seconds >= cls.STUCK_MIN_LAP_TIME_SECONDS
        )

    @staticmethod
    def _lap_time_seconds(telemetry):
        try:
            return float(telemetry.get("curLapTime") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    def _record_telemetry_sample(
        self,
        results,
        telemetry,
        action,
        reward,
        off_track,
    ):
        track_sensors = self._track_sensor_values(telemetry)
        front_sensor = track_sensors[9] if len(track_sensors) > 9 else None
        min_track_sensor = min(track_sensors) if track_sensors else None
        results["telemetry_samples"].append(
            {
                "step": results["steps"],
                "dist_from_start": telemetry.get("distFromStart"),
                "dist_raced": telemetry.get("distRaced"),
                "speed_x": telemetry.get("speedX"),
                "speed_y": telemetry.get("speedY"),
                "angle": telemetry.get("angle"),
                "track_pos": telemetry.get("trackPos"),
                "damage": telemetry.get("damage"),
                "steer": action.get("steer") if isinstance(action, Mapping) else None,
                "accel": action.get("accel") if isinstance(action, Mapping) else None,
                "brake": action.get("brake") if isinstance(action, Mapping) else None,
                "gear": action.get("gear") if isinstance(action, Mapping) else None,
                "reward": reward,
                "off_track": off_track,
                "front_sensor": front_sensor,
                "min_track_sensor": min_track_sensor,
                "track_sensors": track_sensors,
                "cur_lap_time": telemetry.get("curLapTime"),
                "last_lap_time": telemetry.get("lastLapTime"),
            }
        )

    @staticmethod
    def _track_sensor_values(telemetry):
        track = telemetry.get("track")
        if track is None:
            return []
        try:
            return [float(value) for value in list(track)[:19]]
        except (TypeError, ValueError):
            return []
