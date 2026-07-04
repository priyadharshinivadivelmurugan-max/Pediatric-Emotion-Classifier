# main.py
import os
import json
import socket
import threading
import time
import collections
from datetime import datetime, timedelta
from urllib.parse import urlparse

from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.textinput import TextInput
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.utils import platform
from kivy.graphics import Color, Rectangle
import requests
import shutil

IS_ANDROID = platform == "android"

# Android-specific imports
if IS_ANDROID:
    from jnius import autoclass, cast
    from android import activity
    from android.permissions import request_permissions, Permission
    PythonActivity = autoclass('org.kivy.android.PythonActivity')
    Uri = autoclass('android.net.Uri')
    request_permissions([
        Permission.RECORD_AUDIO,
        Permission.READ_EXTERNAL_STORAGE,
        Permission.WRITE_EXTERNAL_STORAGE,
        Permission.READ_MEDIA_AUDIO,
        Permission.READ_MEDIA_IMAGES
    ])

# Desktop audio libs
if not IS_ANDROID:
    try:
        import sounddevice as sd
        import soundfile as sf
        import librosa
        import numpy as np
        import scipy.signal as sg
    except Exception:
        sd = None
        sf = None

# Plyer imports
if IS_ANDROID:
    from plyer import audio
else:
    try:
        from plyer import filechooser
    except Exception:
        filechooser = None


DEFAULT_SERVER = "http://10.91.93.211:5000"
SAMPLE_RATE = 16000
DURATION = 4
LOCAL_FILENAME = "last_recording.wav"
SAMPLES_PER_MINUTE = 20
ACK_TIMEOUT = 5 * 60
SUMMARY_INTERVAL = 20 * 60


def config_path():
    return os.path.join(App.get_running_app().user_data_dir, "config.json")


def load_config():
    p = config_path()
    if os.path.exists(p):
        try:
            with open(p, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_config(cfg: dict):
    p = config_path()
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(cfg, f)


def normalize_server_text(txt: str):
    txt = txt.strip()
    if not txt:
        return None, None, None
    if txt.startswith("http://") or txt.startswith("https://"):
        base = txt.rstrip("/")
    else:
        if ":" in txt:
            base = "http://" + txt.rstrip("/")
        else:
            base = "http://" + txt.rstrip("/") + ":5000"
    return base, base + "/predict", base + "/send_alert"


def test_server_socket(base_url, timeout=3.0):
    try:
        parsed = urlparse(base_url)
        host = parsed.hostname
        port = parsed.port or 80
        sock = socket.create_connection((host, port), timeout=timeout)
        sock.close()
        return True, f"Connected to {host}:{port}"
    except Exception as e:
        return False, str(e)


class MainLayout(BoxLayout):
    def __init__(self, **kwargs):
        super().__init__(orientation="vertical", spacing=25, padding=[50, 80, 50, 80], **kwargs)

        # Background
        with self.canvas.before:
            Color(0.90, 0.95, 1, 1)
            self.rect = Rectangle(size=self.size, pos=self.pos)
        self.bind(size=self._update_rect, pos=self._update_rect)

        # Title
        title_bar = BoxLayout(size_hint_y=None, height=900, padding=(10, 10))
        self.title_label = Label(
            text="Cry Monitor",
            markup=True,
            font_size="36sp",
            bold=True,
            color=(0, 0, 0, 1),
            height=900,
            halign="center",
            valign="middle"
        )
        title_bar.add_widget(self.title_label)
        self.add_widget(title_bar)

        # Server row
        cfg_row = BoxLayout(size_hint_y=None, height=50, spacing=20)
        self.server_input = TextInput(
            multiline=False,
            hint_text="server:port or http://host:port",
            size_hint=(0.6, 1)
        )
        self.btn_save = Button(text="Save", size_hint=(0.2, 1),font_size="18sp")
        self.btn_test = Button(text="Test", size_hint=(0.2, 1),font_size="18sp")
        self.btn_save.bind(on_release=self.on_save_server)
        self.btn_test.bind(on_release=self.on_test_server)
        cfg_row.add_widget(self.server_input)
        cfg_row.add_widget(self.btn_save)
        cfg_row.add_widget(self.btn_test)
        self.add_widget(cfg_row)

        # Small spacing line
        self.add_widget(Label(size_hint_y=None, height=10))

        # Prediction / Status Card (fixed and reliable)
        self.pred_box = BoxLayout(size_hint_y=None, height=140, padding=10, orientation="vertical")
        with self.pred_box.canvas.before:
            Color(0.8, 0.9, 1, 1)  # light blue background
            self.pred_rect = Rectangle(size=self.pred_box.size, pos=self.pred_box.pos)
        self.pred_box.bind(size=self._update_pred_rect, pos=self._update_pred_rect)

        self.status = Label(
            text="Prediction: Waiting for input...",
            size_hint=(1, 1),
            color=(0, 0, 0, 1),
            font_size="10sp",
            halign="left",
            valign="top"
        )
        # ensure the text wraps and fits inside the box (prevents clipping)
        self.status.bind(size=self._update_label_text_size)
        self.pred_box.add_widget(self.status)
        self.add_widget(self.pred_box)

        # File chooser buttons
        self.btn_choose = Button(
            text="Choose WAV File",
            size_hint_y=None,
            height=80,
            background_color=(0.1, 0.3, 0.6, 1),
            font_size="22sp"
        )
        self.btn_choose.bind(on_release=self.on_choose)
        self.add_widget(self.btn_choose)

        self.btn_send = Button(
            text="Send Selected File",
            size_hint_y=None,
            height=80,
            background_color=(0.1, 0.5, 0.1, 1),
            font_size="22sp"
        )
        self.btn_send.bind(on_release=self.on_send)
        self.add_widget(self.btn_send)

        # File info
        self.last_file_label = Label(
            text="No file selected",
            size_hint_y=None,
            height=80,
            color=(0, 0, 0, 1),
            font_size="20sp"
        )
        self.add_widget(self.last_file_label)

        # Control buttons
        ctrl_row = BoxLayout(size_hint_y=None, height=90, spacing=20)
        self.btn_start = Button(text="Start Monitoring", background_color=(0.5, 0.3, 0.1, 1), font_size="22sp")
        self.btn_stop = Button(text="Stop Monitoring", background_color=(0.6, 0.5, 0.5, 1), font_size="22sp")
        self.btn_start.bind(on_release=self.start_monitoring)
        self.btn_stop.bind(on_release=self.stop_monitoring)
        self.btn_stop.disabled = True
        ctrl_row.add_widget(self.btn_start)
        ctrl_row.add_widget(self.btn_stop)
        self.add_widget(ctrl_row)

        # Acknowledge button
        self.btn_ack = Button(
            text="Acknowledge",
            size_hint_y=None,
            height=90,
            background_color=(0.3, 0.6, 0.8, 1),
            font_size="22sp"
        )
        self.btn_ack.bind(on_release=self.on_ack)
        self.btn_ack.disabled = True
        self.add_widget(self.btn_ack)

        # Variables
        self.selected_file = None
        self.monitoring = False
        self.pred_buffer = collections.deque(maxlen=SAMPLES_PER_MINUTE)
        self.current_minute_result = None
        self.last_minute_time = None
        self.last_ack_time = None
        self.last_sms_time = None
        self.alert_sent = False
        self.server_base = None
        self.server_predict = None
        self.server_sms = None

        Clock.schedule_once(lambda dt: self._load_config_and_update_ui(), 0)

        if IS_ANDROID:
            activity.bind(on_activity_result=self._on_activity_result)

        # Start threads AFTER UI and event loop are ready
        Clock.schedule_once(lambda dt: threading.Thread(target=self._minute_aggregator_loop, daemon=True).start(), 2)
        Clock.schedule_once(lambda dt: threading.Thread(target=self._alert_manager_loop, daemon=True).start(), 2)

    # ---------------- Layout helpers ----------------
    def _update_rect(self, *args):
        self.rect.pos = self.pos
        self.rect.size = self.size

    def _update_pred_rect(self, instance, value):
        # keep the background rectangle aligned with the prediction box
        self.pred_rect.pos = instance.pos
        self.pred_rect.size = instance.size

    def _update_label_text_size(self, instance, value):
        # set text_size so Label wraps and vertical alignment works
        instance.text_size = (instance.width - 20, None)

    # ---------------- CONFIG ----------------
    def _load_config_and_update_ui(self):
        cfg = load_config()
        s = cfg.get("server", DEFAULT_SERVER)
        base, predict, sms = normalize_server_text(s)
        self.server_base, self.server_predict, self.server_sms = base, predict, sms
        self.server_input.text = s
        self._update_status(f"Server: {self.server_base}")

    def on_save_server(self, instance):
        txt = self.server_input.text.strip()
        base, predict, sms = normalize_server_text(txt)
        if base is None:
            self._update_status("Invalid server input")
            return
        save_config({"server": base})
        self.server_base, self.server_predict, self.server_sms = base, predict, sms
        self._update_status(f"Saved server: {self.server_base}")

    def on_test_server(self, instance):
        txt = self.server_input.text.strip()
        base, predict, sms = normalize_server_text(txt)
        if base is None:
            self._update_status("Invalid server input")
            return
        self._update_status("Testing connection...")
        def _t():
            ok, msg = test_server_socket(base, timeout=3.0)
            Clock.schedule_once(lambda dt: self._update_status("OK: " + msg if ok else "FAILED: " + msg), 0)
        threading.Thread(target=_t, daemon=True).start()

    # ---------------- FILE PICKER ----------------
    def on_choose(self, instance):
        if IS_ANDROID:
            try:
                Intent = autoclass('android.content.Intent')
                intent = Intent(Intent.ACTION_OPEN_DOCUMENT)
                intent.setType("audio/*")
                intent.addCategory(Intent.CATEGORY_OPENABLE)
                intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
                PythonActivity.mActivity.startActivityForResult(intent, 42)
                self._update_status("Opening file picker...")
            except Exception as e:
                self._update_status(f"Picker error: {e}")
        else:
            if not filechooser:
                self._update_status("File chooser not available")
                return
            filechooser.open_file(on_selection=self._file_selected, filters=['*.wav', 'audio/*'])

    def _on_activity_result(self, request_code, result_code, intent):
        try:
            if request_code != 42 or result_code != -1 or intent is None:
                self._update_status("No file selected")
                return

            uri = intent.getData()
            if uri is None:
                self._update_status("No file selected")
                return

            # Access Android context and resolver
            context = PythonActivity.mActivity.getApplicationContext()
            resolver = context.getContentResolver()
            input_stream = resolver.openInputStream(uri)

            tmp_path = os.path.join(App.get_running_app().user_data_dir, "chosen_audio.wav")

            # Read Java input stream properly
            bytearray_size = 4096
            buffer = bytearray(bytearray_size)
            with open(tmp_path, "wb") as f:
                while True:
                    bytes_read = input_stream.read(buffer)
                    if bytes_read == -1 or bytes_read == 0:
                        break
                    f.write(buffer[:bytes_read])

            input_stream.close()

            # Update UI
            self.selected_file = tmp_path
            Clock.schedule_once(lambda dt: self._update_status("File ready for upload"), 0)
            self._update_status("File ready for upload")

        except Exception as e:
            self._update_status(f"Selection failed: {e}")

    def _file_selected(self, selection):
        if not selection:
            self._update_status("No file chosen")
            return
        path = selection[0]
        self.selected_file = path
        self.last_file_label.text = f"Selected: {os.path.basename(path)}"
        self._update_status("File ready for upload")

    def on_send(self, instance):
        if not self.selected_file or not os.path.exists(self.selected_file):
            self._update_status("No file selected")
            return
        self._update_status("Uploading...")
        threading.Thread(target=self._upload_thread, args=(self.selected_file,), daemon=True).start()

    def _upload_thread(self, filepath):
        try:
            files = {"file": open(filepath, "rb")}
            resp = requests.post(self.server_predict, files=files, timeout=30)
            text = resp.text
        except Exception as e:
            text = f"Upload failed: {e}"
        Clock.schedule_once(lambda dt: self._update_status(f"Prediction: {text}"), 0)

    # ---------------- MONITORING ----------------
    def start_monitoring(self, instance):
        self.monitoring = True
        self.btn_start.disabled = True
        self.btn_stop.disabled = False
        self._update_status("Monitoring started...")
        threading.Thread(target=self._monitor_loop, daemon=True).start()

    def stop_monitoring(self, instance):
        self.monitoring = False
        self.btn_start.disabled = False
        self.btn_stop.disabled = True
        self._update_status("Monitoring stopped")

    def _monitor_loop(self):
        user_dir = App.get_running_app().user_data_dir
        os.makedirs(user_dir, exist_ok=True)
        while self.monitoring:
            try:
                filepath = os.path.join(user_dir, LOCAL_FILENAME)
                if IS_ANDROID:
                    # Android audio via plyer audio wrapper (if used)
                    audio.file_path = filepath
                    audio.start()
                    Clock.schedule_once(lambda dt: audio.stop(), DURATION)
                    time.sleep(DURATION+1)
                else:
                    if sd is None:
                        raise RuntimeError("sounddevice not available")
                    frames = int(DURATION * SAMPLE_RATE)
                    data = sd.rec(frames, samplerate=SAMPLE_RATE, channels=1, dtype='float32')
                    sd.wait()
                    sf.write(filepath, data.flatten(), SAMPLE_RATE, subtype='PCM_16')
                if not os.path.exists(filepath):
                    raise IOError("Recorded file not found")
                files = {"file": open(filepath, "rb")}
                resp = requests.post(self.server_predict, files=files, timeout=30)
                j = resp.json()
                label = j.get("label", "Unknown")
                conf = float(j.get("confidence", 0.0))
            except Exception as e:
                label, conf = f"Error:{e}", 0.0
            self.pred_buffer.append((label, conf, time.time()))
            Clock.schedule_once(lambda dt, L=label, C=conf: self._live_update(L, C), 0)
            time.sleep(0.2)

    def _live_update(self, label, conf):
        # Update the prediction card text (keeps it inside the box)
        self._update_status(f"Live: {label} ({conf*100:.0f}%)")

    # ---------------- AGGREGATION / ALERT ----------------
    def _minute_aggregator_loop(self):
        while True:
            time.sleep(60)
            if not self.pred_buffer:
                print("[DEBUG] No predictions yet — skipping this minute.")
                continue

            print("[DEBUG] Aggregating minute predictions...")

            counts, sums = {}, {}
            for label, conf, ts in self.pred_buffer:
                counts[label] = counts.get(label, 0) + 1
                sums[label] = sums.get(label, 0.0) + conf

            majority = max(counts.items(), key=lambda x: x[1])[0]
            avg_conf = sums[majority] / counts[majority]

            print(f"[DEBUG] Minute result computed: {majority} ({avg_conf:.2f})")

            self.current_minute_result = (majority, avg_conf)
            self.last_minute_time = datetime.now()
            self.pred_buffer.clear()
            self.alert_sent = False
            self.last_sms_time = None

            Clock.schedule_once(lambda dt, L=majority, C=avg_conf: self._minute_update(L, C), 0)

    def _minute_update(self, label, conf):
        self.last_file_label.text = f"Minute emotion: {label} ({conf*100:.0f}%)"
        self.btn_ack.disabled = False

    def on_ack(self, instance):
        self.last_ack_time = datetime.now()
        self.btn_ack.disabled = True
        self._update_status(f"Acknowledged at {self.last_ack_time.strftime('%H:%M:%S')}")

    def _alert_manager_loop(self):
        while True:
            time.sleep(5)
            now = datetime.now()

            # If we never had a minute summary yet, skip
            if not self.current_minute_result or not self.last_minute_time:
                continue

            # Debug print to trace logic
            print(f"[DEBUG] Checking alerts @ {now.strftime('%H:%M:%S')} | Last minute: {self.last_minute_time}")

            # Has the user acknowledged this cycle?
            ack_missing = (self.last_ack_time is None) or (self.last_ack_time < self.last_minute_time)
            time_since_last = (now - self.last_minute_time).total_seconds()

            if ack_missing:
                print(f"[DEBUG] {time_since_last:.1f}s since last minute result — ack missing.")
                # If > ACK_TIMEOUT seconds have passed without ack, send SMS
                if time_since_last >= ACK_TIMEOUT:
                    if not self.alert_sent:
                        print("[DEBUG] Triggering alert SMS now...")
                        self._send_alert_sms(self.current_minute_result)
                        self.alert_sent = True
                        self.last_sms_time = datetime.now()
                    else:
                        print("[DEBUG] Alert already sent for this cycle.")
            else:
                # If acknowledged, check if we should send a summary after SUMMARY_INTERVAL
                if self.last_sms_time is not None:
                    since_last_sms = (now - self.last_sms_time).total_seconds()
                    if since_last_sms >= SUMMARY_INTERVAL:
                        print("[DEBUG] Sending summary SMS...")
                        self._send_summary_sms(self.current_minute_result)
                        self.last_sms_time = now


    def _send_alert_sms(self, minute_result):
        label, conf = minute_result
        try:
            print(f"[DEBUG] Sending alert to server: {label} ({conf:.2f}) → {self.server_sms}")
            r = requests.post(self.server_sms, json={"label": label, "confidence": conf})
            print(f"[DEBUG] Server responded: {r.status_code}, {r.text}")
            Clock.schedule_once(lambda dt: self._update_status(f"Alert SMS: {r.text}"), 0)
        except Exception as e:
            print(f"[ERROR] SMS send failed: {e}")
            Clock.schedule_once(lambda dt: self._update_status(f"SMS failed: {e}"), 0)

    def _send_summary_sms(self, minute_result):
        label, conf = minute_result
        try:
            r = requests.post(self.server_sms, json={"label": label, "confidence": conf, "summary": True})
            Clock.schedule_once(lambda dt: self._update_status(f"Summary SMS: {r.text}"), 0)
        except Exception as e:
            Clock.schedule_once(lambda dt: self._update_status(f"SMS failed: {e}"), 0)

    def _update_status(self, msg):
        # central place to update the prediction/status label text properly
        # ensures the label remains inside the prediction card and wraps
        self.status.text = msg

class CryApp(App):
    def build(self):
        return MainLayout()

if __name__ == "__main__":
    CryApp().run()


