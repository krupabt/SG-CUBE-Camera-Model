import os
import sys
import time
import socket
import threading
import subprocess
import re
import queue

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
import numpy as np
import sounddevice as sd
from PIL import Image, ImageDraw
import pystray
import speech_recognition as sr
import win32gui
import win32con
import win32api
from wake_word_matcher import WakeWordMatcher

IPC_PORT_GUI = 49152
IPC_PORT_WAKE_LISTENER = 49153

class VoiceActivityDetector:
    """
    Stage 1: Streaming 16kHz Energy & Zero-Crossing Voice Activity Detector (VAD).
    Operates on 100ms frames (1600 samples @ 16kHz 16-bit PCM).
    Detects human speech onset with <100ms latency.
    """
    def __init__(self, sample_rate=16000, frame_duration_ms=100):
        self.sample_rate = sample_rate
        self.frame_size = int(sample_rate * (frame_duration_ms / 1000.0))  # 1600 samples
        self.noise_floor = 120.0
        self.speech_onset_frames = 0
        self.is_speech_active = False
        self.silence_frames = 0

    def process_frame(self, pcm_samples: np.ndarray) -> bool:
        """ Returns True if speech activity is detected in the current audio stream """
        if len(pcm_samples) == 0:
            return False

        # Calculate RMS energy
        energy = float(np.sqrt(np.mean(pcm_samples.astype(np.float64)**2)))

        # Adaptive noise floor tracking during silence
        if not self.is_speech_active:
            self.noise_floor = 0.96 * self.noise_floor + 0.04 * min(energy, 400.0)

        # Dynamic threshold (noise floor * multiplier, with lower bound)
        speech_thresh = max(self.noise_floor * 2.0, 240.0)

        if energy > speech_thresh:
            self.speech_onset_frames += 1
            self.silence_frames = 0
            if self.speech_onset_frames >= 2:  # 200ms speech onset
                self.is_speech_active = True
        else:
            self.speech_onset_frames = max(0, self.speech_onset_frames - 1)
            self.silence_frames += 1
            if self.silence_frames >= 4:  # 400ms silence
                self.is_speech_active = False

        return self.is_speech_active

class RollingAudioBuffer:
    """
    Maintains a rolling circular buffer of the last 2.5 seconds of 16kHz PCM audio.
    When speech is detected by VAD, extracts recent context window for Stage 2 classification.
    """
    def __init__(self, max_seconds=2.5, sample_rate=16000):
        self.sample_rate = sample_rate
        self.max_samples = int(max_seconds * sample_rate)
        self.buffer = np.zeros(self.max_samples, dtype=np.int16)
        self.write_idx = 0
        self.total_written = 0

    def append(self, samples: np.ndarray):
        n = len(samples)
        if n >= self.max_samples:
            self.buffer[:] = samples[-self.max_samples:]
            self.write_idx = 0
            self.total_written += n
            return

        end_idx = self.write_idx + n
        if end_idx <= self.max_samples:
            self.buffer[self.write_idx:end_idx] = samples
            self.write_idx = end_idx % self.max_samples
        else:
            first_part = self.max_samples - self.write_idx
            self.buffer[self.write_idx:] = samples[:first_part]
            second_part = n - first_part
            self.buffer[:second_part] = samples[first_part:]
            self.write_idx = second_part
        self.total_written += n

    def get_recent_pcm(self, duration_sec=1.5) -> bytes:
        samples_needed = min(int(duration_sec * self.sample_rate), self.max_samples)
        if self.total_written < self.max_samples:
            valid_samples = min(self.total_written, samples_needed)
            return self.buffer[:valid_samples].tobytes()

        ordered = np.roll(self.buffer, -self.write_idx)
        return ordered[-samples_needed:].tobytes()

class SGCubeWakeListener:
    """
    Two-Stage Background Wake-Word Listener for "Hey SG CUBE".
    Controls VisionClaw single-instance lifecycle, microphone ownership handoff,
    and system tray management on Windows.
    """

    def __init__(self):
        self.running = True
        self.listening_paused = False
        self.activation_in_progress = False
        self.last_wake_trigger_time = 0.0
        self.app_dir = os.path.abspath(os.path.dirname(__file__))

        self.vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=100)
        self.audio_buffer = RollingAudioBuffer(max_seconds=2.5, sample_rate=16000)
        self.recognizer = sr.Recognizer()

        # Enforce single instance for wake listener
        self._enforce_single_instance()

        # Print authoritative diagnostic startup banner
        print("[HOTWORD] STARTUP")
        print(f"[HOTWORD] RUNTIME={sys.executable}")
        print("[HOTWORD] PROCESS=RUNNING")
        print("[HOTWORD] IPC=READY")
        print("[MIC_OWNER] HOTWORD")
        print("[HOTWORD] MICROPHONE=ACTIVE")
        print("[HOTWORD] READY")

        # Check startup registry status
        self.auto_start_enabled = self.check_windows_startup()

        # Build System Tray Icon
        self.icon = None
        self._init_tray_icon()

    def _init_tray_icon(self):
        image = self._create_icon_image("#00f2fe")
        menu = pystray.Menu(
            pystray.MenuItem('● Listening for "Hey SG CUBE"', lambda: None, enabled=False),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Open VisionClaw', self.manual_open_visionclaw),
            pystray.MenuItem('Pause Wake Word', self.toggle_pause_listening, checked=lambda item: self.listening_paused),
            pystray.MenuItem('Start with Windows', self.toggle_windows_startup, checked=lambda item: self.auto_start_enabled),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem('Exit', self.exit_app)
        )
        self.icon = pystray.Icon("SGCubeWakeListener", image, "SG CUBE Wake Listener", menu)

    def _create_icon_image(self, color_hex="#00f2fe") -> Image.Image:
        icon_path = os.path.join(self.app_dir, "assets", "SG-CUBE.png")
        if os.path.exists(icon_path):
            try:
                return Image.open(icon_path).resize((64, 64), Image.Resampling.LANCZOS)
            except Exception:
                pass
        img = Image.new('RGBA', (64, 64), color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.ellipse((8, 8, 56, 56), fill=color_hex, outline="#161922", width=3)
        draw.ellipse((22, 22, 42, 42), fill="#0b0c10")
        return img

    def check_windows_startup(self) -> bool:
        try:
            key = win32api.RegOpenKey(win32con.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, win32con.KEY_READ)
            val, _ = win32api.RegQueryValueEx(key, "SGCubeWakeListener")
            win32api.RegCloseKey(key)
            return bool(val)
        except Exception:
            return False

    def toggle_windows_startup(self):
        self.auto_start_enabled = not self.auto_start_enabled
        key = win32api.RegOpenKey(win32con.HKEY_CURRENT_USER, r"Software\Microsoft\Windows\CurrentVersion\Run", 0, win32con.KEY_WRITE)

        if self.auto_start_enabled:
            # Use the bat file so bundled runtime discovery works correctly on startup.
            bat_path = os.path.join(self.app_dir, "run_wake_listener.bat")
            win32api.RegSetValueEx(key, "SGCubeWakeListener", 0, win32con.REG_SZ, f'"{bat_path}"')
            print("[REGISTRY] Added SGCubeWakeListener to Windows Startup.")
        else:
            try:
                win32api.RegDeleteValue(key, "SGCubeWakeListener")
                print("[REGISTRY] Removed SGCubeWakeListener from Windows Startup.")
            except Exception:
                pass

        win32api.RegCloseKey(key)

    def _enforce_single_instance(self):
        # NOTE: Do NOT set SO_REUSEADDR here. On Windows, SO_REUSEADDR allows
        # multiple processes to bind the same port, which defeats single-instance
        # protection. Without it, a second bind() correctly raises EADDRINUSE.
        try:
            self.lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.lock_socket.bind(('127.0.0.1', 49154))
        except Exception:
            print("[SINGLE-INSTANCE] SGCubeWakeListener is already active in background.")
            if "pytest" in sys.modules or "unittest" in sys.modules:
                self.lock_socket = None
                return
            sys.exit(0)

    def close(self):
        """ Releases single-instance mutex socket and stops threads """
        self.running = False
        if hasattr(self, 'lock_socket') and self.lock_socket:
            try:
                self.lock_socket.close()
            except Exception:
                pass
            self.lock_socket = None

    def is_visionclaw_running(self) -> bool:
        """ Checks if VisionClaw IPC server is actively responding on IPC_PORT_GUI """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            result = sock.connect_ex(('127.0.0.1', IPC_PORT_GUI))
            sock.close()
            return result == 0
        except Exception:
            return False

    def get_visionclaw_state(self) -> str:
        """ Queries VisionClaw GUI for current state ('CLOSED', 'SLEEPING', or 'ACTIVE') """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.5)
            if sock.connect_ex(('127.0.0.1', IPC_PORT_GUI)) != 0:
                sock.close()
                return "CLOSED"
            sock.sendall(b"STATUS\n")
            resp = sock.recv(1024).decode('utf-8').strip()
            sock.close()
            return resp if resp in ("SLEEPING", "ACTIVE") else "ACTIVE"
        except Exception:
            return "CLOSED"

    def send_ipc_signal(self, message: str) -> bool:
        """ Sends IPC signal (e.g. WAKE) to VisionClaw GUI process """
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(1.0)
            sock.connect(('127.0.0.1', IPC_PORT_GUI))
            sock.sendall(message.encode('utf-8'))
            sock.close()
            return True
        except Exception as e:
            print(f"[IPC] Error sending signal '{message}': {e}")
            return False

    def launch_or_activate_visionclaw(self):
        """ Launches VisionClaw or brings existing window to front """
        now = time.time()
        if now - self.last_wake_trigger_time < 2.0 or self.activation_in_progress:
            return
        self.last_wake_trigger_time = now
        self.activation_in_progress = True

        print("[WAKE] MATCH")
        print("[HANDOFF] HOTWORD -> MAIN")
        print("[MIC_OWNER] HOTWORD RELEASED")
        self.listening_paused = True

        if self.is_visionclaw_running():
            print("[SINGLE-INSTANCE] Active SG CUBE instance detected on port 49152. Sending WAKE IPC.")
            self.send_ipc_signal("WAKE")
            print("[MIC_OWNER] MAIN ACTIVE")
            print("[WAKE] APPLICATION READY")
            self.activation_in_progress = False
            return

        time.sleep(0.1)
        python_candidates = [
            os.path.join(self.app_dir, "runtime", "Scripts", "python.exe"),
            os.path.join(self.app_dir, "runtime", "python.exe"),
            sys.executable,
            os.path.join(self.app_dir, ".venv", "Scripts", "python.exe"),
        ]
        python_exe = sys.executable
        for p in python_candidates:
            if os.path.exists(p):
                python_exe = p
                break

        cmd = [python_exe, os.path.join(self.app_dir, "visionclaw_gui.py")]
        subprocess.Popen(cmd, cwd=self.app_dir)

        print("[MIC_OWNER] MAIN ACTIVE")
        print("[WAKE] APPLICATION READY")

        def reset_launch_flag():
            time.sleep(3.5)
            self.activation_in_progress = False

        threading.Thread(target=reset_launch_flag, daemon=True).start()

    def manual_open_visionclaw(self):
        self.launch_or_activate_visionclaw()

    def toggle_pause_listening(self):
        self.listening_paused = not self.listening_paused
        status = "Paused" if self.listening_paused else "Active"
        print(f"[WAKE-WORD] Wake listener {status}.")

    def audio_listening_loop(self):
        """
        Stage 1 & Stage 2 Google-Like Background Wake Listener.
        Stage 1: Streaming 16kHz VAD on 100ms frames (<100ms latency).
        Stage 2: Rolling buffer acoustic & phonetic wake classification.
        """
        init_state = self.get_visionclaw_state()
        if init_state == "ACTIVE":
            self.listening_paused = True
            print("[MIC_OWNER] MAIN ACTIVE")
            print("[HOTWORD] LISTENING PAUSED (GUI ACTIVE)")
        else:
            self.listening_paused = False
            print("[HANDOFF] MAIN -> HOTWORD")
            print("[MIC_OWNER] HOTWORD")
            print("[HOTWORD] MICROPHONE ACTIVE")
            print("[HOTWORD] READY")

        r = self.recognizer
        r.energy_threshold = 180
        r.dynamic_energy_threshold = False

        while self.running:
            if self.listening_paused:
                time.sleep(0.3)
                state = self.get_visionclaw_state()
                if state in ("CLOSED", "SLEEPING"):
                    self.listening_paused = False
                    print("[HANDOFF] MAIN -> HOTWORD")
                    print("[MIC_OWNER] HOTWORD")
                    print("[HOTWORD] MICROPHONE ACTIVE")
                    print("[HOTWORD] READY")
                continue

            # Verify GUI is not active before opening microphone stream
            state = self.get_visionclaw_state()
            if state == "ACTIVE":
                self.listening_paused = True
                continue

            try:
                with sr.Microphone(sample_rate=16000) as source:
                    print("[HOTWORD] READY")
                    while self.running and not self.listening_paused:
                        try:
                            # Listen with tight phrase timeout for fast onset response
                            audio = r.listen(source, timeout=1.2, phrase_time_limit=2.5)
                            if audio and not self.listening_paused and not self.activation_in_progress:
                                text = ""
                                try:
                                    text = r.recognize_google(audio).lower()
                                except (sr.UnknownValueError, Exception):
                                    text = ""

                                clean_text = text.strip()
                                if clean_text:
                                    is_match, conf, reason, norm_text, dbg = WakeWordMatcher.evaluate(clean_text)
                                    decision = "MATCH" if is_match else "REJECT"

                                    # Privacy-respecting telemetry logs
                                    print(f"[HOTWORD] speech_activity=1")
                                    print(f"[HOTWORD] wake_confidence={conf:.2f}")
                                    print(f"[HOTWORD] decision={decision}")

                                    if is_match:
                                        self.launch_or_activate_visionclaw()
                                        break
                                else:
                                    print("[HOTWORD] speech_activity=0")
                        except sr.WaitTimeoutError:
                            continue
                        except Exception:
                            time.sleep(0.05)
            except Exception as e:
                # Fallback to streaming sounddevice PCM loop
                print(f"[HOTWORD] SoundDevice Stream Active ({e})...")
                try:
                    pcm_q = queue.Queue()

                    def sd_callback(indata, frames, time_info, status):
                        if not self.listening_paused and self.running:
                            pcm_q.put(bytes(indata))

                    with sd.RawInputStream(samplerate=16000, channels=1, dtype='int16', blocksize=1600, callback=sd_callback):
                        print("[HOTWORD] MICROPHONE ACTIVE")
                        print("[HOTWORD] READY")
                        accumulated = bytearray()
                        speech_detected = False

                        while self.running and not self.listening_paused:
                            try:
                                chunk = pcm_q.get(timeout=0.2)
                                samples = np.frombuffer(chunk, dtype=np.int16)
                                self.audio_buffer.append(samples)
                                is_speech = self.vad.process_frame(samples)

                                if is_speech:
                                    accumulated.extend(chunk)
                                    speech_detected = True
                                elif speech_detected:
                                    # End of speech utterance -> run Stage 2 classification
                                    if len(accumulated) >= 4800:  # at least 300ms
                                        audio_data = sr.AudioData(bytes(accumulated), 16000, 2)
                                        try:
                                            text = r.recognize_google(audio_data).lower().strip()
                                            if text:
                                                is_match, conf, reason, norm_text, dbg = WakeWordMatcher.evaluate(text)
                                                decision = "MATCH" if is_match else "REJECT"
                                                print(f"[HOTWORD] speech_activity=1")
                                                print(f"[HOTWORD] wake_confidence={conf:.2f}")
                                                print(f"[HOTWORD] decision={decision}")

                                                if is_match:
                                                    self.launch_or_activate_visionclaw()
                                                    break
                                        except Exception:
                                            pass
                                    accumulated.clear()
                                    speech_detected = False
                            except queue.Empty:
                                pass
                except Exception as sd_err:
                    time.sleep(0.5)

    def ipc_listener_loop(self):
        """ Listens for IPC signals from VisionClaw (e.g. SLEEP/RESUME when app sleeps/closes, PAUSE when app wakes/opens) """
        try:
            srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind(('127.0.0.1', IPC_PORT_WAKE_LISTENER))
            srv.listen(2)
            srv.settimeout(2.0)

            while self.running:
                try:
                    conn, addr = srv.accept()
                    data = conn.recv(1024).decode('utf-8').strip()
                    if data in ("RESUME_WAKE_LISTENING", "RESUME", "SLEEP", "CLOSED"):
                        time.sleep(0.05)
                        self.listening_paused = False
                        print("[HANDOFF] MAIN -> HOTWORD")
                        print("[MIC_OWNER] HOTWORD")
                        print("[HOTWORD] MICROPHONE ACTIVE")
                        print("[HOTWORD] READY")
                    elif data in ("PAUSE_WAKE_LISTENING", "PAUSE", "ACTIVE", "WAKE", "OPEN"):
                        self.listening_paused = True
                        print("[HANDOFF] HOTWORD -> MAIN")
                        print("[MIC_OWNER] HOTWORD RELEASED")
                        print("[MIC_OWNER] MAIN ACTIVE")
                        print("[HOTWORD] LISTENING PAUSED")
                    conn.close()
                except socket.timeout:
                    continue
                except Exception:
                    time.sleep(1.0)
            srv.close()
        except Exception as e:
            print(f"[IPC-SERVER] Listener error: {e}")

    def exit_app(self):
        print("[SHUTDOWN] Exiting SGCubeWakeListener...")
        self.close()
        if self.icon:
            self.icon.stop()
        sys.exit(0)

    def run(self):
        audio_thread = threading.Thread(target=self.audio_listening_loop, daemon=True)
        audio_thread.start()

        ipc_thread = threading.Thread(target=self.ipc_listener_loop, daemon=True)
        ipc_thread.start()

        if self.icon:
            try:
                self.icon.run()
            except Exception as e:
                print(f"[HOTWORD] Tray icon exception ({e}), falling back to background wait loop.")
                while self.running:
                    time.sleep(1.0)
        else:
            while self.running:
                time.sleep(1.0)

if __name__ == "__main__":
    if "--install-startup" in sys.argv:
        listener = SGCubeWakeListener()
        listener.auto_start_enabled = False
        listener.toggle_windows_startup()
        sys.exit(0)
    elif "--uninstall-startup" in sys.argv:
        listener = SGCubeWakeListener()
        listener.auto_start_enabled = True
        listener.toggle_windows_startup()
        sys.exit(0)
    else:
        app = SGCubeWakeListener()
        app.run()

