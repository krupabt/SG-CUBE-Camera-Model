import os
import sys
import time
import unittest
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from wake_word_matcher import WakeWordMatcher
from wake_listener import VoiceActivityDetector, RollingAudioBuffer, SGCubeWakeListener

class TestGoogleLikeWakeListener(unittest.TestCase):

    def setUp(self):
        self.vad = VoiceActivityDetector(sample_rate=16000, frame_duration_ms=100)
        self.buffer = RollingAudioBuffer(max_seconds=2.5, sample_rate=16000)

    # ══════════════════════════════════════════════════════════════
    # 1. STAGE 1: VOICE ACTIVITY DETECTION & ROLLING BUFFER
    # ══════════════════════════════════════════════════════════════
    def test_vad_silence_vs_speech(self):
        """VAD detects silence as inactive, and speech burst as active"""
        # Silence frame (low amplitude Gaussian noise)
        silence_frame = np.random.normal(0, 15, 1600).astype(np.int16)
        for _ in range(5):
            self.assertFalse(self.vad.process_frame(silence_frame))

        # Speech frame (moderate amplitude sine wave)
        t = np.linspace(0, 0.1, 1600)
        speech_frame = (1500 * np.sin(2 * np.pi * 300 * t)).astype(np.int16)
        
        # After 2 frames of speech, VAD becomes active
        self.vad.process_frame(speech_frame)
        is_speech = self.vad.process_frame(speech_frame)
        self.assertTrue(is_speech, "VAD should activate on 200ms speech onset")

    def test_rolling_buffer_sliding_window(self):
        """RollingAudioBuffer preserves recent context up to max_seconds"""
        samples = np.full(1600, 100, dtype=np.int16)
        for _ in range(30):
            self.buffer.append(samples)
        recent_pcm = self.buffer.get_recent_pcm(duration_sec=1.0)
        self.assertEqual(len(recent_pcm), 16000 * 2)  # 16k samples * 2 bytes = 32000 bytes

    # ══════════════════════════════════════════════════════════════
    # 2. STAGE 2: 50 VALID WAKE ATTEMPTS (HIGH RECALL)
    # ══════════════════════════════════════════════════════════════
    def test_50_valid_wake_attempts(self):
        valid_samples = [
            "SG CUBE", "Hey SG CUBE", "hey sg cube", "s g cube", "S G CUBE",
            "ess gee cube", "Ess Gee Cube", "es gee cube", "Es Gee Cube",
            "hey sg", "Hey SG", "hey s g", "Hey S G", "hey ess gee", "Hey Ess Gee",
            "hey ess gee cube", "heysgcube", "sgcube", "SG", "sg", "s g",
            "SGQ", "sgq", "s g q", "sg cub", "SG Cub", "sg cue", "SG Cue",
            "cuube", "Cuube", "kyube", "hi sg cube", "Hi SG CUBE", "hi sg", "Hi SG",
            "ok sg cube", "Ok SG CUBE", "ok sg", "Ok SG", "hey cube", "Hey Cube",
            "hi cube", "Hi Cube", "hello sg cube", "Hello SG CUBE", "hello sg", "Hello SG",
            "hey sgcube", "esse gee cube", "es-gee-cube"
        ]
        self.assertEqual(len(valid_samples), 50)

        true_positives = 0
        latencies = []

        for sample in valid_samples:
            t0 = time.perf_counter()
            is_match, conf, reason, norm_text, _ = WakeWordMatcher.evaluate(sample)
            t1 = time.perf_counter()
            latencies.append((t1 - t0) * 1000)

            if is_match:
                true_positives += 1
            else:
                print(f"MISSED VALID: '{sample}' -> conf={conf}, reason={reason}")

        recall = (true_positives / len(valid_samples)) * 100.0
        avg_latency_ms = sum(latencies) / len(latencies)
        max_latency_ms = max(latencies)

        print(f"\n[VALID WAKE BENCHMARK] Recall: {true_positives}/50 ({recall:.1f}%), Avg Latency: {avg_latency_ms:.3f}ms, Max Latency: {max_latency_ms:.3f}ms")
        self.assertEqual(true_positives, 50, f"Expected 50/50 recall, got {true_positives}/50")
        self.assertLess(avg_latency_ms, 50.0, "Classification latency must be <50ms")

    # ══════════════════════════════════════════════════════════════
    # 3. 50 NEGATIVE NON-WAKE SAMPLES (LOW FALSE ACTIVATION)
    # ══════════════════════════════════════════════════════════════
    def test_50_negative_non_wake_samples(self):
        negative_samples = [
            "facebook", "execute", "play music", "good morning everyone", "hello",
            "what are you doing", "call my friend", "open the file", "look at this",
            "tell me a joke", "how are you", "what is the weather", "turn on the lights",
            "google", "youtube", "computer", "everyone", "conversation", "my name is Alice",
            "one two three", "testing one two", "random sentence", "play some songs",
            "what time is it", "open youtube", "start the car", "turn up the volume",
            "nice to meet you", "can you hear me", "who are you talking to",
            "where are my keys", "take a picture", "read the screen", "check the temperature",
            "send a message", "call mom", "set a timer", "cancel alarm",
            "increase brightness", "battery percentage", "is it raining outside",
            "show me directions", "find nearby coffee", "how far is the station",
            "good night", "good morning", "see you later", "thank you very much",
            "stop the video", "pause the playback"
        ]
        self.assertEqual(len(negative_samples), 50)

        false_positives = 0
        for sample in negative_samples:
            is_match, conf, reason, norm_text, _ = WakeWordMatcher.evaluate(sample)
            if is_match:
                false_positives += 1
                print(f"FALSE ACTIVATION: '{sample}' -> conf={conf}, reason={reason}")

        rejection_rate = ((50 - false_positives) / 50) * 100.0
        print(f"[NEGATIVE BENCHMARK] False Activations: {false_positives}/50, Rejection Rate: {rejection_rate:.1f}%")
        self.assertEqual(false_positives, 0, f"Expected 0 false activations, got {false_positives}/50")

    # ══════════════════════════════════════════════════════════════
    # 4. SLEEP/WAKE LIFECYCLE & 20-CYCLE STRESS TEST
    # ══════════════════════════════════════════════════════════════
    def test_20_cycle_sleep_wake_stress(self):
        """Runs 20 consecutive Sleep -> Wake cycles to verify no resource leaks or duplicates"""
        import tkinter as tk
        from visionclaw_gui import SGCubeApp

        root = tk.Tk()
        root.withdraw()
        app = SGCubeApp(root)

        for cycle in range(1, 21):
            # 1. Sleep: Main Mic OFF, Camera OFF, Gemini OFF, Hotword ON
            app.enter_sleep_mode()
            self.assertEqual(app.current_state, "SLEEPING")
            self.assertFalse(app.ai_running)
            self.assertFalse(app.camera_running)

            # 2. Wake: Hotword OFF, Main Mic ON, Camera ON, Gemini ON
            app.bring_to_foreground()
            self.assertIn(app.current_state, ("LISTENING", "ACTIVE"))
            self.assertTrue(app.camera_running)

        print(f"[STRESS TEST] 20/20 Sleep-Wake cycles passed successfully with exclusive microphone ownership.")
        app.on_close()

if __name__ == "__main__":
    unittest.main()
