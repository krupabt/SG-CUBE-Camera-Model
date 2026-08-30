import os
import sys
import time
import queue
import random
import math
import socket
import threading
import subprocess
import asyncio
import cv2
import numpy as np
from PIL import Image, ImageTk, ImageDraw, ImageFilter
import sounddevice as sd
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from dotenv import load_dotenv

import google.genai as genai
from google.genai import types

from assistive.vision_engine import VisionEngine
from assistive.command_router import OFFICIAL_INTRODUCTION

load_dotenv()

IPC_PORT_GUI = 49152
IPC_PORT_WAKE_LISTENER = 49153

SYSTEM_INSTRUCTION = f"""
You are SG CUBE, a warm, calm, intelligent personal AI companion and assistive camera assistant for blind and visually impaired users.

When asked to introduce yourself, who you are, or what SG CUBE is, give this official introduction:
"{OFFICIAL_INTRODUCTION}"

Guidelines:
1. Speak like a smart, friendly companion. Be natural, warm, respectful, and concise.
2. Provide spatially intuitive descriptions (e.g. "directly ahead", "slightly to your left", "on your right", "near", "farther ahead").
3. Be conservative with safety warnings. Never claim the user is guaranteed safe.
4. Express uncertainty gracefully when unsure about an object, face, or banknote.
5. Never use robotic phrasing or technical system jargon.
"""

MORNING_GREETINGS = [
    "Good morning! I'm here. Ready to get started?",
    "Morning! I'm here with you. What are we doing today?",
    "Good morning! Ready when you are."
]

AFTERNOON_GREETINGS = [
    "Good afternoon! I'm here. What can I help you with?",
    "Hey, good afternoon! I'm ready when you are.",
    "Good afternoon! Ready to help."
]

EVENING_GREETINGS = [
    "Good evening! I'm here with you. What do you need?",
    "Hey, good evening! I'm ready.",
    "Good evening! Nice to have you back."
]

NIGHT_GREETINGS = [
    "Hey, you're up late. I'm here if you need me.",
    "Good night hours! I'm still here with you.",
    "Late night! Ready whenever you are."
]

# --- SG CUBE Ultra-Dark HUD Color System (Pure Black #000000) ---
COLOR_BG_PRIMARY = "#000000"      # Pure Black Window Background
COLOR_BG_SECONDARY = "#030303"    # Secondary Deep Black Background
COLOR_PANEL_DEEP = "#050505"      # Visually Deeper Black Panels / Cards
COLOR_PANEL_SECONDARY = "#050505" # Inset Control Areas
COLOR_PANEL_HOVER = "#0A0A0A"     # Elevated Interactive Hover Panels
COLOR_BORDER_SUBTLE = "#141414"   # 1px Subtle Border
COLOR_BORDER_HOVER = "#242424"    # Hover Border
COLOR_BORDER_ACTIVE = "#00EDFF"   # Active Glowing Cyan Border
COLOR_NAV_ACTIVE_BG = "#061217"   # Active Home Tab Inset Glow Background

COLOR_CYAN_PRIMARY = "#00EDFF"    # Primary Brand Cyan
COLOR_TEAL_MINT = "#4DF7C4"       # Mint / Green Accents
COLOR_ORANGE = "#FDAB72"          # Orange / Gold Accents
COLOR_PINK = "#F678AB"            # Pink Accents
COLOR_PURPLE = "#B377F7"          # Purple Accents
COLOR_STATUS_GREEN = "#3ADC8C"    # Healthy Green Status
COLOR_WARNING_GOLD = "#F8D93D"    # Warning Gold
COLOR_ALERT_RED = "#FF4757"       # Alert Red
COLOR_TEXT_PRIMARY = "#F1F5F9"    # High-contrast Primary Text
COLOR_TEXT_SECONDARY = "#8B96A5"  # Secondary Muted Text
COLOR_TEXT_MUTED = "#526071"      # Inactive / Subtle Text

def _hex_to_rgb(hex_str):
    hex_str = hex_str.lstrip('#')
    return tuple(int(hex_str[i:i+2], 16) for i in (0, 2, 4))

class GlowingHUDIcon:
    """
    Renders pin-sharp 2D vector HUD icons with a subtle, beautiful neon halo glow
    using 4x supersampling, Gaussian alpha blur, and Lanczos downsampling.
    """
    _pil_cache = {}

    @classmethod
    def get_photo_image(cls, name: str, size: int = 24, color_hex: str = "#00EDFF", hover: bool = False, master=None, rotation: float = 0.0) -> ImageTk.PhotoImage:
        img = cls.get_pil_image(name, size, color_hex, hover=hover, rotation=rotation)
        return ImageTk.PhotoImage(img, master=master)

    @classmethod
    def get_pil_image(cls, name: str, size: int = 24, color_hex: str = "#00EDFF", hover: bool = False, rotation: float = 0.0) -> Image.Image:
        key = (name, size, color_hex, hover, round(rotation, 1))
        if key in cls._pil_cache:
            return cls._pil_cache[key]
        img = cls.render_icon_with_glow(name, size, color_hex, hover=hover, rotation=rotation)
        cls._pil_cache[key] = img
        return img

    @classmethod
    def render_icon_with_glow(cls, name: str, size: int = 24, color_hex: str = "#00EDFF", hover: bool = False, rotation: float = 0.0) -> Image.Image:
        scale = 4
        S = size * scale
        canvas_size = S + 16 * scale  # Padding for soft neon glow halo
        glow_offset = 8 * scale

        icon_canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(icon_canvas)
        rgb = _hex_to_rgb(color_hex)
        
        if hover:
            r_c = min(255, int(rgb[0] * 1.15 + 20))
            g_c = min(255, int(rgb[1] * 1.15 + 20))
            b_c = min(255, int(rgb[2] * 1.15 + 20))
            fg_col = (r_c, g_c, b_c, 255)
        else:
            fg_col = (*rgb, 255)

        pad = int(S * 0.08) + glow_offset
        w = S - 2 * int(S * 0.08)
        h = S - 2 * int(S * 0.08)
        stroke_w = max(3, int(S * 0.08))
        cx = canvas_size // 2
        cy = canvas_size // 2

        if name in ("logo", "sg_cube_logo"):
            # 3D isometric futuristic AI companion cube (32px brand logo)
            r_c = int(S * 0.38)
            pts_top = [(cx, cy - r_c), (cx + int(r_c * 0.866), cy - r_c // 2), (cx, cy), (cx - int(r_c * 0.866), cy - r_c // 2)]
            pts_left = [(cx - int(r_c * 0.866), cy - r_c // 2), (cx, cy), (cx, cy + r_c), (cx - int(r_c * 0.866), cy + r_c // 2)]
            pts_right = [(cx + int(r_c * 0.866), cy - r_c // 2), (cx, cy), (cx, cy + r_c), (cx + int(r_c * 0.866), cy + r_c // 2)]
            draw.polygon(pts_top, outline=fg_col, width=stroke_w)
            draw.polygon(pts_left, outline=fg_col, width=stroke_w)
            draw.polygon(pts_right, outline=fg_col, width=stroke_w)
            draw.ellipse([cx - int(S*0.1), cy - int(S*0.1), cx + int(S*0.1), cy + int(S*0.1)], fill=fg_col)

        elif name in ("home", "house"):
            pts = [
                (cx, pad),
                (pad, int(cy * 0.95)),
                (pad + int(S * 0.14), int(cy * 0.95)),
                (pad + int(S * 0.14), canvas_size - pad),
                (canvas_size - pad - int(S * 0.14), canvas_size - pad),
                (canvas_size - pad - int(S * 0.14), int(cy * 0.95)),
                (canvas_size - pad, int(cy * 0.95))
            ]
            draw.polygon(pts, outline=fg_col, width=stroke_w)
            door_w = int(S * 0.22)
            door_h = int(S * 0.34)
            dx1 = cx - door_w // 2
            dy1 = canvas_size - pad - door_h
            draw.rectangle([dx1, dy1, dx1 + door_w, canvas_size - pad], fill=fg_col)

        elif name in ("vision", "eye"):
            draw.arc([pad, cy - int(h * 0.32), canvas_size - pad, cy + int(h * 0.32)], start=0, end=180, fill=fg_col, width=stroke_w)
            draw.arc([pad, cy - int(h * 0.32), canvas_size - pad, cy + int(h * 0.32)], start=180, end=360, fill=fg_col, width=stroke_w)
            r_iris = int(S * 0.24)
            draw.ellipse([cx - r_iris, cy - r_iris, cx + r_iris, cy + r_iris], outline=fg_col, width=stroke_w)
            r_pupil = int(S * 0.11)
            draw.ellipse([cx - r_pupil, cy - r_pupil, cx + r_pupil, cy + r_pupil], fill=fg_col)

        elif name in ("memory", "brain"):
            cw = int(S * 0.52)
            ch = int(S * 0.52)
            draw.rounded_rectangle([cx - cw // 2, cy - ch // 2, cx + cw // 2, cy + ch // 2], radius=int(S * 0.1), outline=fg_col, width=stroke_w)
            draw.rectangle([cx - int(cw * 0.3), cy - int(ch * 0.3), cx + int(cw * 0.3), cy + int(ch * 0.3)], fill=fg_col)
            pin_len = int(S * 0.18)
            for offset in (-int(cw * 0.24), int(cw * 0.24)):
                draw.line([(cx + offset, cy - ch // 2), (cx + offset, cy - ch // 2 - pin_len)], fill=fg_col, width=stroke_w)
                draw.line([(cx + offset, cy + ch // 2), (cx + offset, cy + ch // 2 + pin_len)], fill=fg_col, width=stroke_w)
                draw.line([(cx - cw // 2, cy + offset), (cx - cw // 2 - pin_len, cy + offset)], fill=fg_col, width=stroke_w)
                draw.line([(cx + cw // 2, cy + offset), (cx + cw // 2 + pin_len, cy + offset)], fill=fg_col, width=stroke_w)

        elif name in ("history", "clock"):
            r = int(S * 0.42)
            draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=fg_col, width=stroke_w)
            draw.line([(cx, cy), (cx, cy - int(r * 0.65))], fill=fg_col, width=stroke_w)
            draw.line([(cx, cy), (cx + int(r * 0.55), cy)], fill=fg_col, width=stroke_w)
            draw.ellipse([cx - int(S * 0.05), cy - int(S * 0.05), cx + int(S * 0.05), cy + int(S * 0.05)], fill=fg_col)

        elif name in ("people", "users"):
            cx1 = cx + int(S * 0.12)
            r_head = int(S * 0.16)
            draw.ellipse([cx1 - r_head, cy - int(S * 0.34), cx1 + r_head, cy - int(S * 0.34) + 2 * r_head], fill=fg_col)
            draw.arc([cx1 - int(S * 0.32), cy - int(S * 0.02), cx1 + int(S * 0.32), cy + int(S * 0.44)], start=180, end=360, fill=fg_col, width=stroke_w)
            cx2 = cx - int(S * 0.18)
            r_head2 = int(S * 0.13)
            draw.ellipse([cx2 - r_head2, cy - int(S * 0.26), cx2 + r_head2, cy - int(S * 0.26) + 2 * r_head2], outline=fg_col, width=stroke_w)
            draw.arc([cx2 - int(S * 0.26), cy + int(S * 0.04), cx2 + int(S * 0.26), cy + int(S * 0.42)], start=180, end=270, fill=fg_col, width=stroke_w)

        elif name in ("glasses", "meta_glass"):
            gw = int(S * 0.36)
            gh = int(S * 0.30)
            draw.rounded_rectangle([pad, cy - gh // 2, pad + gw, cy + gh // 2], radius=int(S * 0.08), outline=fg_col, width=stroke_w)
            draw.rounded_rectangle([canvas_size - pad - gw, cy - gh // 2, canvas_size - pad, cy + gh // 2], radius=int(S * 0.08), outline=fg_col, width=stroke_w)
            draw.line([(pad + gw, cy - int(gh * 0.15)), (canvas_size - pad - gw, cy - int(gh * 0.15))], fill=fg_col, width=stroke_w)
            draw.line([(pad, cy - int(gh * 0.2)), (pad - int(S * 0.06), cy - int(gh * 0.4))], fill=fg_col, width=stroke_w)
            draw.line([(canvas_size - pad, cy - int(gh * 0.2)), (canvas_size - pad + int(S * 0.06), cy - int(gh * 0.4))], fill=fg_col, width=stroke_w)

        elif name in ("gear", "settings"):
            r_out = int(S * 0.40)
            r_in = int(S * 0.26)
            r_hole = int(S * 0.13)
            num_teeth = 8
            pts = []
            rot_rad = math.radians(rotation)
            for i in range(num_teeth * 2):
                angle = i * math.pi / num_teeth + rot_rad
                r = r_out if (i % 2 == 0) else r_in
                pts.append((cx + int(r * math.cos(angle)), cy + int(r * math.sin(angle))))
            draw.polygon(pts, outline=fg_col, width=stroke_w)
            draw.ellipse([cx - r_hole, cy - r_hole, cx + r_hole, cy + r_hole], fill=fg_col)

        elif name in ("mic", "speak"):
            mw = int(S * 0.28)
            mh = int(S * 0.44)
            draw.rounded_rectangle([cx - mw // 2, pad, cx + mw // 2, pad + mh], radius=mw // 2, fill=fg_col)
            cw = int(S * 0.48)
            draw.arc([cx - cw // 2, pad + int(mh * 0.3), cx + cw // 2, pad + mh + int(S * 0.16)], start=0, end=180, fill=fg_col, width=stroke_w)
            sy1 = pad + mh + int(S * 0.16)
            sy2 = canvas_size - pad - stroke_w
            draw.line([(cx, sy1), (cx, sy2)], fill=fg_col, width=stroke_w)
            draw.line([(cx - int(S * 0.22), sy2), (cx + int(S * 0.22), sy2)], fill=fg_col, width=stroke_w)

        elif name in ("describe", "scan"):
            corner_l = int(S * 0.22)
            draw.line([(pad, pad + corner_l), (pad, pad), (pad + corner_l, pad)], fill=fg_col, width=stroke_w)
            draw.line([(canvas_size - pad - corner_l, pad), (canvas_size - pad, pad), (canvas_size - pad, pad + corner_l)], fill=fg_col, width=stroke_w)
            draw.line([(pad, canvas_size - pad - corner_l), (pad, canvas_size - pad), (pad + corner_l, canvas_size - pad)], fill=fg_col, width=stroke_w)
            draw.line([(canvas_size - pad - corner_l, canvas_size - pad), (canvas_size - pad, canvas_size - pad), (canvas_size - pad, canvas_size - pad - corner_l)], fill=fg_col, width=stroke_w)
            draw.arc([pad + int(S*0.08), cy - int(S*0.2), canvas_size - pad - int(S*0.08), cy + int(S*0.2)], start=0, end=180, fill=fg_col, width=stroke_w)
            draw.arc([pad + int(S*0.08), cy - int(S*0.2), canvas_size - pad - int(S*0.08), cy + int(S*0.2)], start=180, end=360, fill=fg_col, width=stroke_w)
            r_pupil = int(S * 0.10)
            draw.ellipse([cx - r_pupil, cy - r_pupil, cx + r_pupil, cy + r_pupil], fill=fg_col)

        elif name in ("recognize", "person"):
            r_head = int(S * 0.20)
            draw.ellipse([cx - r_head, pad, cx + r_head, pad + 2 * r_head], outline=fg_col, width=stroke_w)
            draw.arc([pad, cy, canvas_size - pad, canvas_size - pad + int(S * 0.15)], start=180, end=360, fill=fg_col, width=stroke_w)
            b_len = int(S * 0.15)
            draw.line([(pad - 2, pad + b_len), (pad - 2, pad - 2), (pad + b_len, pad - 2)], fill=fg_col, width=stroke_w)
            draw.line([(canvas_size - pad + 2 - b_len, pad - 2), (canvas_size - pad + 2, pad - 2), (canvas_size - pad + 2, pad + b_len)], fill=fg_col, width=stroke_w)

        elif name in ("ocr", "read_text"):
            dw = int(S * 0.60)
            dh = int(S * 0.74)
            x1 = (canvas_size - dw) // 2
            y1 = (canvas_size - dh) // 2
            draw.rounded_rectangle([x1, y1, x1 + dw, y1 + dh], radius=int(S * 0.07), outline=fg_col, width=stroke_w)
            draw.line([(x1 + int(dw * 0.20), y1 + int(dh * 0.28)), (x1 + int(dw * 0.80), y1 + int(dh * 0.28))], fill=fg_col, width=stroke_w)
            draw.line([(x1 + int(dw * 0.20), y1 + int(dh * 0.50)), (x1 + int(dw * 0.80), y1 + int(dh * 0.50))], fill=fg_col, width=stroke_w)
            draw.line([(x1 + int(dw * 0.20), y1 + int(dh * 0.72)), (x1 + int(dw * 0.55), y1 + int(dh * 0.72))], fill=fg_col, width=stroke_w)

        elif name in ("currency", "rupee"):
            bw = int(S * 0.80)
            bh = int(S * 0.54)
            bx1 = (canvas_size - bw) // 2
            by1 = (canvas_size - bh) // 2
            draw.rounded_rectangle([bx1, by1, bx1 + bw, by1 + bh], radius=int(S * 0.08), outline=fg_col, width=stroke_w)
            rw = int(bw * 0.38)
            draw.line([(cx - rw // 2, cy - int(bh * 0.26)), (cx + rw // 2, cy - int(bh * 0.26))], fill=fg_col, width=stroke_w)
            draw.line([(cx - rw // 2, cy - int(bh * 0.08)), (cx + int(rw * 0.4), cy - int(bh * 0.08))], fill=fg_col, width=stroke_w)
            draw.arc([cx - rw // 2, cy - int(bh * 0.26), cx + rw // 2, cy + int(bh * 0.06)], start=270, end=90, fill=fg_col, width=stroke_w)
            draw.line([(cx - int(rw * 0.1), cy - int(bh * 0.02)), (cx + int(rw * 0.4), cy + int(bh * 0.28))], fill=fg_col, width=stroke_w)

        elif name in ("find_object", "search"):
            cx_lens = cx - int(S * 0.08)
            cy_lens = cy - int(S * 0.08)
            r_lens = int(S * 0.28)
            draw.ellipse([cx_lens - r_lens, cy_lens - r_lens, cx_lens + r_lens, cy_lens + r_lens], outline=fg_col, width=stroke_w)
            hx1 = cx_lens + int(r_lens * 0.707)
            hy1 = cy_lens + int(r_lens * 0.707)
            hx2 = canvas_size - pad
            hy2 = canvas_size - pad
            draw.line([(hx1, hy1), (hx2, hy2)], fill=fg_col, width=int(stroke_w * 1.5))

        elif name in ("safety", "shield"):
            top_y = pad
            mid_y = cy + int(S * 0.1)
            bot_y = canvas_size - pad
            half_w = int(S * 0.40)
            pts = [(cx - half_w, top_y), (cx + half_w, top_y),
                   (cx + half_w, mid_y), (cx, bot_y),
                   (cx - half_w, mid_y)]
            draw.polygon(pts, outline=fg_col, width=stroke_w)
            draw.line([(cx, top_y + int(S * 0.14)), (cx, top_y + int(S * 0.38))], fill=fg_col, width=stroke_w)
            draw.ellipse([cx - stroke_w // 2, top_y + int(S * 0.48), cx + stroke_w // 2, top_y + int(S * 0.48) + stroke_w], fill=fg_col)
            
        else:
            draw.ellipse([pad, pad, canvas_size - pad, canvas_size - pad], outline=fg_col, width=stroke_w)

        alpha_mask = icon_canvas.split()[3]
        blur_radius = (4.5 * scale) if hover else (3.0 * scale)
        glow_alpha = alpha_mask.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        glow_opacity = 0.95 if hover else 0.65
        glow_alpha = glow_alpha.point(lambda p: int(p * glow_opacity))
        
        glow_layer = Image.new("RGBA", (canvas_size, canvas_size), (*rgb, 0))
        glow_layer.putalpha(glow_alpha)
        
        composite = Image.alpha_composite(glow_layer, icon_canvas)
        
        target_size = size
        if hover:
            target_size = int(size * 1.10)
            
        return composite.resize((target_size, target_size), Image.Resampling.LANCZOS)

HUDIcon = GlowingHUDIcon

class HUDTooltip:
    """
    Lightweight floating dark glass HUD tooltip for navigation tabs and action buttons with 150ms responsive display.
    """
    def __init__(self, widget, text, delay=150):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.tip_window = None
        self.after_id = None
        self.widget.bind("<Enter>", self._on_enter, add="+")
        self.widget.bind("<Leave>", self._on_leave, add="+")
        self.widget.bind("<ButtonPress>", self._on_leave, add="+")

    def _on_enter(self, event=None):
        self._cancel()
        self.after_id = self.widget.after(self.delay, self._show)

    def _on_leave(self, event=None):
        self._cancel()
        self._hide()

    def _cancel(self):
        if self.after_id:
            try:
                self.widget.after_cancel(self.after_id)
            except Exception:
                pass
            self.after_id = None

    def _show(self):
        if self.tip_window or not self.text:
            return
        try:
            x = self.widget.winfo_rootx() + (self.widget.winfo_width() // 2)
            y = self.widget.winfo_rooty() + self.widget.winfo_height() + 6
            self.tip_window = tw = tk.Toplevel(self.widget)
            tw.wm_overrideredirect(True)
            tw.wm_geometry(f"+{x}+{y}")
            tw.configure(bg=COLOR_CYAN_PRIMARY)

            frame = tk.Frame(tw, bg=COLOR_PANEL_DEEP, highlightbackground=COLOR_CYAN_PRIMARY, highlightthickness=1, padx=8, pady=3)
            frame.pack(fill=tk.BOTH)

            lbl = tk.Label(frame, text=self.text, font=("Segoe UI", 8, "bold"), fg=COLOR_TEXT_PRIMARY, bg=COLOR_PANEL_DEEP)
            lbl.pack()
        except Exception:
            pass

    def _hide(self):
        if self.tip_window:
            try:
                self.tip_window.destroy()
            except Exception:
                pass
            self.tip_window = None

class FuturisticAudioCoreRenderer:
    """
    Clean Futuristic Voice AI Core Renderer:
    - Main center audio diaphragm (70-85px diameter) with dark glass body (#030303) and glowing cyan core (#00EDFF)
    - EXACTLY 3 concentric audio-energy rings (Ring 1 Cyan #00EDFF, Ring 2 Mint #4DF7C4, Ring 3 Purple #B377F7)
    - Symmetrical Left & Right audio waveforms (55-70px width each, mirroring each other)
    - Clean, voice-oriented, non-radar, non-crosshair aesthetics
    - Multi-state responsiveness: IDLE calm, LISTENING pulse, SPEAKING dynamic energy, THINKING purple flow, SLEEPING dim
    - Smooth hover interaction (1.05 scale, +20% glow)
    """
    def __init__(self, canvas_width=280, canvas_height=140):
        self.width = canvas_width
        self.height = canvas_height
        self.cx = canvas_width // 2
        self.cy = canvas_height // 2

    def draw_audio_core(self, canvas, time_val=0.0, state="IDLE", pulse=1.0, hover=False, mouse_tilt=(0.0, 0.0)):
        tx, ty = mouse_tilt
        cx = self.cx + int(ty * 10)
        cy = self.cy + int(tx * 10)

        if hover:
            pulse *= 1.05

        # Colors, Waveform Amplitudes & Speeds mapped to backend state
        if state == "AI_SPEAKING":
            core_col = "#00EDFF"
            diaph_col = "#4DF7C4"
            ring1_col = "#00EDFF"
            ring2_col = "#4DF7C4"
            ring3_col = "#B377F7"
            glow_col = "#08333e"
            wave_amp = 22.0 if not hover else 26.0
            wave_speed = 8.5
            glow_r = int(76 * pulse)
        elif state == "USER_SPEAKING":
            core_col = "#B377F7"
            diaph_col = "#00EDFF"
            ring1_col = "#00EDFF"
            ring2_col = "#B377F7"
            ring3_col = "#4DF7C4"
            glow_col = "#1c0d28"
            wave_amp = 18.0 if not hover else 22.0
            wave_speed = 7.5
            glow_r = int(74 * pulse)
        elif state == "AI_THINKING":
            core_col = "#00EDFF"
            diaph_col = "#B377F7"
            ring1_col = "#00EDFF"
            ring2_col = "#B377F7"
            ring3_col = "#B377F7"
            glow_col = "#160a22"
            wave_amp = 6.0 if not hover else 8.5
            wave_speed = 4.0
            glow_r = int(68 * pulse)
        elif state == "SAFETY_ALERT":
            core_col = "#FF4757"
            diaph_col = "#F678AB"
            ring1_col = "#FF4757"
            ring2_col = "#F678AB"
            ring3_col = "#FF4757"
            glow_col = "#260812"
            wave_amp = 24.0 if not hover else 28.0
            wave_speed = 9.0
            glow_r = int(78 * pulse)
        elif state == "SLEEPING":
            core_col = "#242e37"
            diaph_col = "#181f25"
            ring1_col = "#161616"
            ring2_col = "#111111"
            ring3_col = "#0c0c0c"
            glow_col = "#040404"
            wave_amp = 0.0
            wave_speed = 0.0
            glow_r = int(42 * pulse)
        else: # IDLE / LISTENING
            core_col = "#00EDFF"
            diaph_col = "#4DF7C4"
            ring1_col = "#00EDFF"
            ring2_col = "#4DF7C4"
            ring3_col = "#B377F7" if state == "LISTENING" else "#0c2b36"
            glow_col = "#041e26" if state == "LISTENING" else "#021419"
            wave_amp = 9.0 if state == "LISTENING" else 2.5
            wave_speed = 6.0 if state == "LISTENING" else 2.5
            if hover:
                wave_amp += 3.0
            glow_r = int((70 if state == "LISTENING" else 60) * pulse)

        # 1. Soft Ambient Radial Aura Glow
        canvas.create_oval(cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r, fill=glow_col, outline="", width=0)

        # 2. EXACTLY THREE Concentric Audio-Energy Rings
        # Ring 3: Outer Ring (Purple / Cyan)
        r3 = int(67 * pulse)
        canvas.create_oval(cx - r3, cy - r3, cx + r3, cy + r3, outline=ring3_col, width=1)

        # Ring 2: Middle Ring (Mint)
        r2 = int(55 * pulse)
        canvas.create_oval(cx - r2, cy - r2, cx + r2, cy + r2, outline=ring2_col, width=1)

        # Ring 1: Inner Ring (Cyan)
        r1 = int(44 * pulse)
        canvas.create_oval(cx - r1, cy - r1, cx + r1, cy + r1, outline=ring1_col, width=1)

        # 3. Symmetrical Left & Right Audio Waveforms (55-70px width each, mirrored)
        if state != "SLEEPING":
            # Primary Left Waveform (x: 12 to 70)
            pts_left_pri = []
            pts_left_sec = []
            for x in range(12, 71, 2):
                rel = (x - 12) / 58.0
                env = math.sin(rel * math.pi)
                wy1 = cy + int(wave_amp * env * math.sin((x - 12) * 0.16 - time_val * wave_speed))
                wy2 = cy + int((wave_amp * 0.6) * env * math.sin((x - 12) * 0.16 - time_val * wave_speed + 1.0))
                pts_left_pri.extend([x, wy1])
                pts_left_sec.extend([x, wy2])

            if len(pts_left_pri) >= 4:
                canvas.create_line(pts_left_pri, fill=ring1_col, width=2 if state in ("AI_SPEAKING", "USER_SPEAKING") else 1.5, smooth=True)
                if state in ("AI_SPEAKING", "USER_SPEAKING", "LISTENING"):
                    canvas.create_line(pts_left_sec, fill=ring2_col, width=1, smooth=True)

            # Symmetrical Right Waveform (x: 210 to 268, exact mirror)
            pts_right_pri = []
            pts_right_sec = []
            for x in range(210, 269, 2):
                rel = (268 - x) / 58.0 # mirrored relative progress
                env = math.sin(rel * math.pi)
                wy1 = cy + int(wave_amp * env * math.sin((268 - x) * 0.16 - time_val * wave_speed))
                wy2 = cy + int((wave_amp * 0.6) * env * math.sin((268 - x) * 0.16 - time_val * wave_speed + 1.0))
                pts_right_pri.extend([x, wy1])
                pts_right_sec.extend([x, wy2])

            if len(pts_right_pri) >= 4:
                canvas.create_line(pts_right_pri, fill=ring1_col, width=2 if state in ("AI_SPEAKING", "USER_SPEAKING") else 1.5, smooth=True)
                if state in ("AI_SPEAKING", "USER_SPEAKING", "LISTENING"):
                    canvas.create_line(pts_right_sec, fill=ring2_col, width=1, smooth=True)

        # 4. Main Center Speaker Diaphragm (70-85px diameter, #030303 dark glass body)
        body_r = int(38 * pulse) + (2 if hover else 0)
        canvas.create_oval(cx - body_r, cy - body_r, cx + body_r, cy + body_r,
                           fill="#030303", outline=ring1_col if state != "SLEEPING" else "#1c1c1c", width=2)

        # Stepped acoustic depth ring
        step_r = body_r - 5
        canvas.create_oval(cx - step_r, cy - step_r, cx + step_r, cy + step_r,
                           fill="#050d12" if state != "SLEEPING" else "#070707",
                           outline="#0a2530" if state != "SLEEPING" else "#101010", width=1)

        # 5. Central Acoustic Diaphragm Cone
        diaph_r = int(24 * pulse)
        canvas.create_oval(cx - diaph_r, cy - diaph_r, cx + diaph_r, cy + diaph_r,
                           fill="#031620" if state != "SLEEPING" else "#060606",
                           outline=ring2_col if state != "SLEEPING" else "#141414", width=1.5)

        # Subtle acoustic diaphragm groove
        if state != "SLEEPING":
            gr_r = int(diaph_r * 0.60)
            canvas.create_oval(cx - gr_r, cy - gr_r, cx + gr_r, cy + gr_r, outline="#0b3846", width=1)

        # 6. Center Glowing Core Dome
        cap_r = int(9 * pulse)
        canvas.create_oval(cx - cap_r, cy - cap_r, cx + cap_r, cy + cap_r,
                           fill=core_col if state != "SLEEPING" else "#242e37",
                           outline="#ffffff" if state != "SLEEPING" else "#333333", width=1)
        if state != "SLEEPING":
            canvas.create_oval(cx - 3, cy - 3, cx + 3, cy + 3, fill="#ffffff", outline="")

    def draw_cube(self, canvas, *args, **kwargs):
        """ Backward-compatibility alias for draw_audio_core """
        time_val = kwargs.get("time_val", 0.0)
        state = kwargs.get("state", "IDLE")
        pulse = kwargs.get("pulse", 1.0)
        hover = kwargs.get("hover", False)
        mouse_tilt = kwargs.get("mouse_tilt", (0.0, 0.0))
        self.draw_audio_core(canvas, time_val=time_val, state=state, pulse=pulse, hover=hover, mouse_tilt=mouse_tilt)

Real3DCubeRenderer = FuturisticAudioCoreRenderer

def animate_dialog_open(dialog, target_alpha=0.98, duration_ms=220):
    """ Smooth opacity opening transition for secondary modal dialogs (220ms ease-out) """
    try:
        dialog.attributes("-alpha", 0.0)
        steps = 8
        interval = max(10, duration_ms // steps)
        def step(i):
            if not dialog or not dialog.winfo_exists():
                return
            # Ease-out curve
            t = i / steps
            ease_t = 1.0 - (1.0 - t) * (1.0 - t)
            a = ease_t * target_alpha
            try:
                dialog.attributes("-alpha", a)
            except Exception:
                pass
            if i < steps:
                dialog.after(interval, lambda: step(i + 1))
        dialog.after(10, lambda: step(1))
    except Exception:
        try:
            dialog.attributes("-alpha", 1.0)
        except Exception:
            pass

def animate_dialog_close(dialog, duration_ms=180, callback=None):
    """ Smooth opacity closing transition for secondary modal dialogs (180ms ease-in) """
    try:
        if not dialog or not dialog.winfo_exists():
            if callback:
                callback()
            return
        steps = 6
        interval = max(10, duration_ms // steps)
        try:
            current_alpha = float(dialog.attributes("-alpha") or 1.0)
        except Exception:
            current_alpha = 1.0
        def step(i):
            if not dialog or not dialog.winfo_exists():
                if callback:
                    callback()
                return
            t = i / steps
            ease_t = t * t
            a = current_alpha * (1.0 - ease_t)
            try:
                dialog.attributes("-alpha", max(0.0, a))
            except Exception:
                pass
            if i < steps:
                dialog.after(interval, lambda: step(i + 1))
            else:
                try:
                    dialog.destroy()
                except Exception:
                    pass
                if callback:
                    callback()
        step(1)
    except Exception:
        try:
            dialog.destroy()
        except Exception:
            pass
        if callback:
            callback()

class SGCubeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SG CUBE — Personal AI Companion")
        self.root.geometry("1080x820")
        self.root.minsize(960, 720)
        self.root.configure(bg=COLOR_BG_PRIMARY)
        self.root.option_add("*Font", ("Segoe UI", 10))

        # Windows Taskbar Icon & AppUserModelID setup
        if os.name == 'nt':
            try:
                import ctypes
                ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("SGCUBE.Assistant.2.4.6")
            except Exception:
                pass

        # Set Window Icon
        _app_dir = os.path.abspath(os.path.dirname(__file__))
        _icon_candidates = [
            os.path.join(_app_dir, "assets", "SG-CUBE.ico"),
            os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "SG-CUBE", "assets", "SG-CUBE.ico")
        ]
        for _ic in _icon_candidates:
            if os.path.exists(_ic):
                try:
                    self.root.iconbitmap(_ic)
                    break
                except Exception:
                    pass

        # Instantiate Assistive Vision Engine
        self.engine = VisionEngine(data_dir="data")

        # State variables and locks
        self.camera_running = False
        self.ai_running = False
        self.camera_thread = None
        self.ai_thread = None
        self.latest_jpeg = None
        self.latest_raw_frame = None
        self.measured_fps = 20.0
        self.camera_lock = threading.Lock()
        self.frame_lock = threading.Lock()
        self.gui_queue = queue.Queue()

        # Session State Machine: IDLE, LISTENING, USER_SPEAKING, AI_THINKING, AI_SPEAKING, RECONNECTING, STOPPED, SLEEPING, SAFETY_ALERT
        self.state_lock = threading.Lock()
        self.current_state = "IDLE"

        # Developer / Debug Overlay Flag
        self.dev_mode = self.engine.store.get_setting("developer_mode", False)

        # Animation state variables for SG CUBE Floating Core & 3D Rotating Cube
        self.anim_angle = 0.0
        self.anim_time = 0.0
        self.orbit_angle = 0.0
        self.context_banner_timer = 0
        self.cube_3d = Real3DCubeRenderer(canvas_width=280, canvas_height=140)
        self.cube_rot_y = 0.0
        self.cube_rot_x = 0.26
        self.cube_rot_z = 0.0
        self.cube_hover = False
        self.cube_mouse_tilt = (0.0, 0.0)

        # Audio Queues & Authoritative Session Control
        self.mic_queue = queue.Queue()
        self.playback_queue = queue.Queue()

        self.session_lock = threading.Lock()
        self.active_session_id = None
        self.current_response_id = 0
        self.playback_thread = None
        self.playback_thread_started = False
        self.playback_stop_evt = threading.Event()
        self.wake_greeting_pending = False
        self.wake_greeting_timer = 0.0
        self.pending_speech_prompt = None
        self.first_valid_frame_received = False
        self.active_history_session_id = self.engine.history.create_session()
        self.action_busy = False
        self.user_transcript_buffer = ""
        self.last_user_speech_time = 0.0

        # Enforce Single-Instance Application Lock
        self._enforce_single_instance()

        self._build_ui()
        self._bind_shortcuts()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        # Start IPC Server for Wake Listener commands
        self._start_ipc_server()

        # Start periodic GUI queue polling (~20ms) and SG CUBE floating orb animation (~35ms)
        self.root.after(20, self._process_gui_queue)
        self.root.after(35, self._animate_sg_cube_core)

        # Auto-start services in background after UI renders
        self.root.after(50, self._notify_wake_listener_pause)
        self.root.after(100, self._update_info_strip)
        self.root.after(300, self._auto_start_services)
        self.root.after(400, self._check_first_run_onboarding)

    def _enforce_single_instance(self):
        """ Enforces single instance. If SG CUBE is already running, sends WAKE IPC signal to active instance and exits. """
        try:
            self.lock_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.lock_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.lock_socket.bind(('127.0.0.1', IPC_PORT_GUI))
            self.lock_socket.listen(5)
            print(f"[SINGLE-INSTANCE] Acquired single-instance socket lock on port {IPC_PORT_GUI}.")
        except Exception:
            print("[SINGLE-INSTANCE] SG CUBE is ALREADY RUNNING. Sending WAKE IPC signal to active instance and exiting...")
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(1.0)
                sock.connect(('127.0.0.1', IPC_PORT_GUI))
                sock.sendall(b"WAKE")
                sock.close()
            except Exception:
                pass
            sys.exit(0)

    def _start_ipc_server(self):
        """ Starts background IPC socket server listening for commands from wake_listener.py """
        def ipc_loop():
            try:
                print(f"[IPC-SERVER] SG CUBE IPC Server listening on port {IPC_PORT_GUI}...")
                while True:
                    conn, addr = self.lock_socket.accept()
                    data = conn.recv(1024).decode('utf-8').strip()
                    if data == "WAKE":
                        print("[IPC-SERVER] Received WAKE signal! Bringing SG CUBE to foreground...")
                        self.gui_queue.put(("ACTION", "WAKE_FOREGROUND"))
                        try:
                            conn.sendall(b"OK\n")
                        except Exception:
                            pass
                    elif data == "STATUS":
                        try:
                            status = "SLEEPING" if self.current_state == "SLEEPING" else "ACTIVE"
                            conn.sendall(f"{status}\n".encode('utf-8'))
                        except Exception:
                            pass
                    conn.close()
            except Exception as e:
                print(f"[IPC-SERVER] Error: {e}")

        t = threading.Thread(target=ipc_loop, daemon=True)
        t.start()

    def set_state(self, new_state: str):
        with self.state_lock:
            if self.current_state == "SLEEPING" and new_state not in ("OPENING", "ACTIVE", "LISTENING"):
                return
            if self.current_state != new_state:
                old_state = self.current_state
                self.current_state = new_state
                print(f"[STATE] {old_state} -> {new_state}")
                self.gui_queue.put(("STATE", new_state))

    def _clear_playback_queue(self):
        """ Clears unplayed audio chunks and advances response_id for barge-in / speech interruption """
        with self.state_lock:
            self.current_response_id += 1
            new_id = self.current_response_id

        self.playback_stop_evt.set()

        cleared_count = 0
        while not self.playback_queue.empty():
            try:
                self.playback_queue.get_nowait()
                cleared_count += 1
            except queue.Empty:
                break
        if cleared_count > 0:
            print(f"[BARGE-IN] Advanced response_id to {new_id}. Purged {cleared_count} stale audio chunks.")

    def _generate_time_greeting(self) -> str:
        hour = time.localtime().tm_hour
        user_name = self.engine.store.get_setting("user_display_name") or self.engine.store.get_setting("user_name", "")
        name_str = f", {user_name}" if user_name and user_name != "User" else ""
        if 5 <= hour < 12:
            return f"Good morning{name_str}! I'm here. What do you need?"
        elif 12 <= hour < 17:
            return f"Good afternoon{name_str}! I'm here. What do you need?"
        elif 17 <= hour < 22:
            return f"Good evening{name_str}! I'm here. What do you need?"
        else:
            return f"Good night{name_str}! I'm here. What do you need?"

    # --- Premium Ultra-Dark HUD UI Construction ---
    # --- Premium Ultra-Dark HUD UI Construction (Reference Locked 1536x1024 Base) ---
    def _build_ui(self):
        # 1. Header Navigation & Status Bar (#000000 Background, Height 62px)
        header = tk.Frame(self.root, bg=COLOR_BG_PRIMARY, height=62, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        header.pack(fill=tk.X, side=tk.TOP)
        header.pack_propagate(False)

        # Brand Container (32x32px 3D Glowing Futuristic Cube Logo + 24px SG CUBE Title + Subtitle)
        brand_frame = tk.Frame(header, bg=COLOR_BG_PRIMARY)
        brand_frame.pack(side=tk.LEFT, padx=(18, 10))

        logo_img = GlowingHUDIcon.get_photo_image("logo", size=32, color_hex=COLOR_CYAN_PRIMARY, hover=False, master=brand_frame)
        self.brand_icon_lbl = tk.Label(brand_frame, image=logo_img, bg=COLOR_BG_PRIMARY)
        self.brand_icon_lbl.image = logo_img
        self.brand_icon_lbl.pack(side=tk.LEFT, padx=(0, 8))

        brand_logo = tk.Label(brand_frame, text="SG CUBE", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 16, "bold"))
        brand_logo.pack(side=tk.LEFT)

        brand_sub = tk.Label(brand_frame, text="Personal AI Companion", bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9))
        brand_sub.pack(side=tk.LEFT, padx=(8, 0))

        # Centered Navigation Group (20px Category-Colored Vector Icons, 14px Text, 8px Gap, 14px H-Pad, 9px V-Pad)
        nav_frame = tk.Frame(header, bg=COLOR_BG_PRIMARY)
        nav_frame.pack(side=tk.LEFT, expand=True)

        self.nav_buttons = {}
        self.nav_buttons["home"] = self._create_nav_btn(nav_frame, "home", "Home", self._on_nav_home, active=True, accent=COLOR_CYAN_PRIMARY, tooltip="Home dashboard")
        self.nav_buttons["vision"] = self._create_nav_btn(nav_frame, "vision", "Vision", self.open_vision_dialog, accent=COLOR_TEAL_MINT, tooltip="Live vision")
        self.nav_buttons["memory"] = self._create_nav_btn(nav_frame, "memory", "Memory", self.open_memory_dialog, accent=COLOR_PURPLE, tooltip="Personal memory")
        self.nav_buttons["history"] = self._create_nav_btn(nav_frame, "history", "History", self.open_history_dialog, accent=COLOR_PURPLE, tooltip="Conversation history")
        self.nav_buttons["people"] = self._create_nav_btn(nav_frame, "people", "People", self.open_people_dialog, accent=COLOR_ORANGE, tooltip="Face profiles")
        self.nav_buttons["glasses"] = self._create_nav_btn(nav_frame, "glasses", "Meta Glass", self.open_meta_glass_dialog, accent=COLOR_CYAN_PRIMARY, tooltip="Meta Glass")

        # Right Header Action Icons (~105px Status Pill + 24px Settings Gear)
        actions_frame = tk.Frame(header, bg=COLOR_BG_PRIMARY)
        actions_frame.pack(side=tk.RIGHT, padx=18)

        self.sys_status_label = tk.Label(
            actions_frame,
            text="● Online",
            bg=COLOR_BG_SECONDARY,
            fg=COLOR_STATUS_GREEN,
            font=("Segoe UI", 9, "bold"),
            padx=14,
            pady=4,
            highlightbackground="#06383D",
            highlightthickness=1
        )
        self.sys_status_label.pack(side=tk.LEFT, padx=(0, 12))

        # Settings Gear Icon Button (24px with 10° Smooth Animated Rotation on Hover)
        gear_img_normal = GlowingHUDIcon.get_photo_image("gear", size=24, color_hex=COLOR_CYAN_PRIMARY, hover=False, master=actions_frame, rotation=0.0)
        gear_imgs_enter = [
            GlowingHUDIcon.get_photo_image("gear", size=26, color_hex=COLOR_CYAN_PRIMARY, hover=True, master=actions_frame, rotation=3.5),
            GlowingHUDIcon.get_photo_image("gear", size=26, color_hex=COLOR_CYAN_PRIMARY, hover=True, master=actions_frame, rotation=7.0),
            GlowingHUDIcon.get_photo_image("gear", size=26, color_hex=COLOR_CYAN_PRIMARY, hover=True, master=actions_frame, rotation=10.0),
        ]
        gear_imgs_leave = [
            GlowingHUDIcon.get_photo_image("gear", size=25, color_hex=COLOR_CYAN_PRIMARY, hover=True, master=actions_frame, rotation=7.0),
            GlowingHUDIcon.get_photo_image("gear", size=24, color_hex=COLOR_CYAN_PRIMARY, hover=False, master=actions_frame, rotation=3.5),
            gear_img_normal,
        ]

        self.btn_settings = tk.Button(
            actions_frame,
            image=gear_img_normal,
            bg=COLOR_BG_SECONDARY,
            activebackground="#0A0A0A",
            relief=tk.FLAT,
            bd=0,
            padx=8,
            pady=3,
            cursor="hand2",
            highlightbackground=COLOR_BORDER_SUBTLE,
            highlightthickness=1,
            command=self.open_settings_dialog
        )
        btn_settings = self.btn_settings
        btn_settings.image_normal = gear_img_normal
        btn_settings.image = gear_img_normal
        btn_settings.anim_job = None

        def on_gear_enter(e):
            if btn_settings.anim_job:
                self.root.after_cancel(btn_settings.anim_job)
            btn_settings.config(bg="#0A0A0A", image=gear_imgs_enter[0], highlightbackground=COLOR_CYAN_PRIMARY)
            def frame1():
                btn_settings.config(image=gear_imgs_enter[1])
            def frame2():
                btn_settings.config(image=gear_imgs_enter[2])
            self.root.after(45, frame1)
            btn_settings.anim_job = self.root.after(90, frame2)

        def on_gear_leave(e):
            if btn_settings.anim_job:
                self.root.after_cancel(btn_settings.anim_job)
            btn_settings.config(image=gear_imgs_leave[0])
            def frame1():
                btn_settings.config(image=gear_imgs_leave[1])
            def frame2():
                btn_settings.config(bg=COLOR_BG_SECONDARY, image=gear_imgs_leave[2], highlightbackground=COLOR_BORDER_SUBTLE)
            self.root.after(45, frame1)
            btn_settings.anim_job = self.root.after(90, frame2)

        btn_settings.bind("<Enter>", on_gear_enter, add="+")
        btn_settings.bind("<Leave>", on_gear_leave, add="+")

        btn_settings.pack(side=tk.LEFT)
        HUDTooltip(btn_settings, "Settings")

        # 2. Main Stage (Pure Black #000000)
        main_stage = tk.Frame(self.root, bg=COLOR_BG_PRIMARY)
        main_stage.pack(fill=tk.BOTH, expand=True, padx=20, pady=(10, 8))

        # Main Vision Panel (#030303 Panel with 1px #00EDFF Cyan Border and Subtle Outer Glow)
        cam_outer_card = tk.Frame(main_stage, bg=COLOR_BG_SECONDARY, highlightbackground=COLOR_CYAN_PRIMARY, highlightthickness=1)
        cam_outer_card.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        cam_top_bar = tk.Frame(cam_outer_card, bg=COLOR_BG_SECONDARY)
        cam_top_bar.pack(fill=tk.X, padx=15, pady=8)

        cam_title_frame = tk.Frame(cam_top_bar, bg=COLOR_BG_SECONDARY)
        cam_title_frame.pack(side=tk.LEFT)
        self.cam_live_dot = tk.Label(cam_title_frame, text="●", bg=COLOR_BG_SECONDARY, fg=COLOR_ALERT_RED, font=("Segoe UI", 9, "bold"))
        self.cam_live_dot.pack(side=tk.LEFT, padx=(0, 4))
        cam_title = tk.Label(cam_title_frame, text="LIVE VISION", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 10, "bold"))
        cam_title.pack(side=tk.LEFT)

        self.cam_badge = tk.Label(
            cam_top_bar,
            text="● 20 FPS",
            bg=COLOR_BG_SECONDARY,
            fg=COLOR_TEAL_MINT,
            font=("Segoe UI", 8, "bold"),
            padx=10,
            pady=2,
            highlightbackground="#06383D",
            highlightthickness=1
        )
        self.cam_badge.pack(side=tk.RIGHT)

        # Main Camera Viewport Frame (Holds Left Environment HUD, Center Camera Preview, Right Objects HUD)
        cam_viewport_frame = tk.Frame(cam_outer_card, bg=COLOR_BG_SECONDARY)
        cam_viewport_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # --- LEFT SIDE: ENVIRONMENT HUD CARD (Width 165px) ---
        self.hud_env_card = tk.Frame(
            cam_viewport_frame,
            bg="#050505",
            highlightbackground="#102530",
            highlightthickness=1,
            width=165,
            padx=10,
            pady=8
        )
        self.hud_env_card.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 12), pady=2)
        self.hud_env_card.pack_propagate(False)

        self.hud_env_title = tk.Label(
            self.hud_env_card,
            text="ENVIRONMENT",
            bg="#050505",
            fg=COLOR_CYAN_PRIMARY,
            font=("Segoe UI", 8, "bold"),
            anchor="w"
        )
        self.hud_env_title.pack(fill=tk.X, pady=(0, 6))

        self.hud_env_person_lbl = tk.Label(self.hud_env_card, text="● Person: None", bg="#050505", fg="#F1F5F9", font=("Segoe UI", 8), anchor="w")
        self.hud_env_person_lbl.pack(fill=tk.X, pady=2)

        self.hud_env_room_lbl = tk.Label(self.hud_env_card, text="▣ Room: Clear space", bg="#050505", fg="#8B96A5", font=("Segoe UI", 8), anchor="w", wraplength=145, justify=tk.LEFT)
        self.hud_env_room_lbl.pack(fill=tk.X, pady=2)

        self.hud_env_light_lbl = tk.Label(self.hud_env_card, text="☼ Light: Normal", bg="#050505", fg="#4DF7C4", font=("Segoe UI", 8), anchor="w")
        self.hud_env_light_lbl.pack(fill=tk.X, pady=2)

        self.hud_env_safety_lbl = tk.Label(self.hud_env_card, text="⚠ Safety: Clear", bg="#050505", fg=COLOR_STATUS_GREEN, font=("Segoe UI", 8, "bold"), anchor="w", wraplength=145, justify=tk.LEFT)
        self.hud_env_safety_lbl.pack(fill=tk.X, pady=2)

        # Subtle hover on environment card
        def _on_env_enter(e):
            self.hud_env_card.config(bg="#0A0A0A", highlightbackground=COLOR_CYAN_PRIMARY)
            self.hud_env_title.config(bg="#0A0A0A")
            for w in (self.hud_env_person_lbl, self.hud_env_room_lbl, self.hud_env_light_lbl, self.hud_env_safety_lbl):
                w.config(bg="#0A0A0A")
        def _on_env_leave(e):
            self.hud_env_card.config(bg="#050505", highlightbackground="#102530")
            self.hud_env_title.config(bg="#050505")
            for w in (self.hud_env_person_lbl, self.hud_env_room_lbl, self.hud_env_light_lbl, self.hud_env_safety_lbl):
                w.config(bg="#050505")
        self.hud_env_card.bind("<Enter>", _on_env_enter, add="+")
        self.hud_env_card.bind("<Leave>", _on_env_leave, add="+")

        # --- CENTER: REAL CAMERA PREVIEW ---
        self.preview_label = tk.Label(
            cam_viewport_frame,
            text="Initializing SG CUBE Vision Window...",
            bg=COLOR_BG_PRIMARY,
            fg=COLOR_TEXT_SECONDARY,
            font=("Segoe UI", 11)
        )
        self.preview_label.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, pady=2)

        # --- RIGHT SIDE: OBJECTS HUD CARD (Width 165px) ---
        self.hud_obj_card = tk.Frame(
            cam_viewport_frame,
            bg="#050505",
            highlightbackground="#102530",
            highlightthickness=1,
            width=165,
            padx=10,
            pady=8
        )
        self.hud_obj_card.pack(side=tk.RIGHT, fill=tk.Y, padx=(12, 0), pady=2)
        self.hud_obj_card.pack_propagate(False)

        self.hud_obj_title = tk.Label(
            self.hud_obj_card,
            text="OBJECTS",
            bg="#050505",
            fg=COLOR_CYAN_PRIMARY,
            font=("Segoe UI", 8, "bold"),
            anchor="w"
        )
        self.hud_obj_title.pack(fill=tk.X, pady=(0, 6))

        self.hud_obj_labels = []
        for _ in range(5):
            lbl = tk.Label(self.hud_obj_card, text="", bg="#050505", fg="#F1F5F9", font=("Segoe UI", 8), anchor="w")
            lbl.pack(fill=tk.X, pady=1)
            self.hud_obj_labels.append(lbl)

        # Subtle hover on objects card
        def _on_obj_enter(e):
            self.hud_obj_card.config(bg="#0A0A0A", highlightbackground=COLOR_CYAN_PRIMARY)
            self.hud_obj_title.config(bg="#0A0A0A")
            for w in self.hud_obj_labels:
                w.config(bg="#0A0A0A")
        def _on_obj_leave(e):
            self.hud_obj_card.config(bg="#050505", highlightbackground="#102530")
            self.hud_obj_title.config(bg="#050505")
            for w in self.hud_obj_labels:
                w.config(bg="#050505")
        self.hud_obj_card.bind("<Enter>", _on_obj_enter, add="+")
        self.hud_obj_card.bind("<Leave>", _on_obj_leave, add="+")

        # 2.5 Real Information Strip (Height 95-105px: Recent History | System Status | User/Time | Quick Memory)
        self._build_info_strip(main_stage)

        # 3. Lower Stage: SG CUBE AI Control Deck (#000000 Background, 20-25% height)
        lower_stage = tk.Frame(main_stage, bg=COLOR_BG_PRIMARY, height=215)
        lower_stage.pack(fill=tk.X, side=tk.BOTTOM)
        lower_stage.pack_propagate(False)

        # Center Status Pill (Centered directly ABOVE 3D Cube, ~150-170px x 32-36px, bg #030303, border 1px #06383D, text 13px #00EDFF)
        self.context_banner = tk.Label(
            lower_stage,
            text="● SG CUBE Listening",
            bg=COLOR_BG_SECONDARY,
            fg=COLOR_CYAN_PRIMARY,
            font=("Segoe UI", 9, "bold"),
            padx=16,
            pady=3,
            highlightbackground="#06383D",
            highlightthickness=1
        )
        self.context_banner.pack(anchor="n", pady=(0, 2))

        # Centered AI Core Control Deck: Left 4 Buttons | Center Large 3D Cube | Right 4 Buttons
        deck_frame = tk.Frame(lower_stage, bg=COLOR_BG_PRIMARY)
        deck_frame.pack(anchor="center", pady=(0, 2))

        # Left Action Group (4 Buttons: Speak, Describe, Recognize, Memory — compact horizontal row)
        left_card = tk.Frame(deck_frame, bg=COLOR_BG_PRIMARY)
        left_card.pack(side=tk.LEFT, padx=(0, 8))

        self.action_buttons = {}
        self.action_buttons["speak"] = self._create_floating_action_btn(left_card, "mic", "Speak", lambda: self._on_space_shortcut(), accent=COLOR_PINK, tooltip="Speak to SG CUBE")
        self.action_buttons["describe"] = self._create_floating_action_btn(left_card, "describe", "Describe", lambda: self._trigger_action_async("describe", "Scene Description", "What is around me?"), accent=COLOR_TEAL_MINT, tooltip="Describe what the camera sees")
        self.action_buttons["recognize"] = self._create_floating_action_btn(left_card, "recognize", "Recognize", lambda: self._trigger_action_async("recognize", "Face Recognition", "Who is in front of me?"), accent=COLOR_ORANGE, tooltip="Recognize people")
        self.action_buttons["memory"] = self._create_floating_action_btn(left_card, "memory", "Memory", self.open_memory_dialog, accent=COLOR_PURPLE, tooltip="Save or recall memory")

        # Center AI Audio Core Canvas (280px x 140px: Circular AI Speaker Core body 95-115px, footprint 150-180px)
        self.orb_canvas = tk.Canvas(
            deck_frame,
            width=280,
            height=140,
            bg=COLOR_BG_PRIMARY,
            highlightthickness=0,
            cursor="hand2"
        )
        self.orb_canvas.pack(side=tk.LEFT, padx=8)

        def on_cube_motion(e):
            self.cube_mouse_tilt = ((e.y - 70) * 0.003, (e.x - 140) * 0.003)
            self.cube_hover = True

        def on_cube_enter(e):
            self.cube_hover = True

        def on_cube_leave(e):
            self.cube_hover = False
            self.cube_mouse_tilt = (0.0, 0.0)

        self.orb_canvas.bind("<Motion>", on_cube_motion, add="+")
        self.orb_canvas.bind("<Enter>", on_cube_enter, add="+")
        self.orb_canvas.bind("<Leave>", on_cube_leave, add="+")
        self.orb_canvas.bind("<Button-1>", lambda e: self._on_space_shortcut(), add="+")
        HUDTooltip(self.orb_canvas, "SG CUBE AI Audio Core")

        # Right Action Group (4 Buttons: Read Text, Currency, Find Object, Safety Alert — compact horizontal row)
        right_card = tk.Frame(deck_frame, bg=COLOR_BG_PRIMARY)
        right_card.pack(side=tk.LEFT, padx=(8, 0))

        self.action_buttons["ocr"] = self._create_floating_action_btn(right_card, "ocr", "Read Text", lambda: self._trigger_action_async("ocr", "Read Text", "Read this"), accent=COLOR_ORANGE, tooltip="Read visible text")
        self.action_buttons["currency"] = self._create_floating_action_btn(right_card, "currency", "Currency", lambda: self._trigger_action_async("currency", "Currency Recognition", "How much money is this?"), accent=COLOR_TEAL_MINT, tooltip="Recognize currency")
        self.action_buttons["find_object"] = self._create_floating_action_btn(right_card, "find_object", "Find Object", self.open_object_finder_dialog, accent=COLOR_CYAN_PRIMARY, tooltip="Find an object")
        self.action_buttons["safety"] = self._create_floating_action_btn(right_card, "safety", "Safety Alert", lambda: self._trigger_action_async("safety", "Safety Hazard Check", "Is it safe?"), accent=COLOR_PINK, tooltip="Check nearby hazards")

        # Bottom Live Dialogue Text Line (~420px x 34px Pill with Speaker Icon 🔊, Centered below Deck)
        dialogue_pill = tk.Frame(
            lower_stage,
            bg=COLOR_BG_SECONDARY,
            highlightbackground=COLOR_CYAN_PRIMARY,
            highlightthickness=1,
            padx=18,
            pady=3
        )
        dialogue_pill.pack(side=tk.BOTTOM, pady=(0, 4))

        self.dialogue_prefix = tk.Label(
            dialogue_pill,
            text="Assistive: ",
            bg=COLOR_BG_SECONDARY,
            fg=COLOR_WARNING_GOLD,
            font=("Segoe UI", 9, "bold")
        )
        self.dialogue_prefix.pack(side=tk.LEFT)

        self.dialogue_banner = tk.Label(
            dialogue_pill,
            text="\"I'm here. What do you need?\"",
            bg=COLOR_BG_SECONDARY,
            fg=COLOR_TEXT_PRIMARY,
            font=("Segoe UI", 9, "italic")
        )
        self.dialogue_banner.pack(side=tk.LEFT)

        self.dialogue_speaker = tk.Label(
            dialogue_pill,
            text=" 🔊",
            bg=COLOR_BG_SECONDARY,
            fg=COLOR_CYAN_PRIMARY,
            font=("Segoe UI", 9)
        )
        self.dialogue_speaker.pack(side=tk.LEFT)

    def _build_info_strip(self, parent):
        """
        Compact Real Information Strip (Height 95-105px) with 4 live cards:
        - Card 1: RECENT HISTORY (Last 3 real history entries from ConversationHistory)
        - Card 2: SYSTEM STATUS (Live Camera, Mic, Gemini Live, Speaker, Battery)
        - Card 3: CURRENT USER / TIME (Dynamic clock, greeting, user profile name)
        - Card 4: QUICK MEMORY (Top 3 stored facts from MemoryManager)
        """
        info_strip = tk.Frame(parent, bg=COLOR_BG_PRIMARY, height=102)
        info_strip.pack(fill=tk.X, side=tk.TOP, pady=(0, 6))
        info_strip.pack_propagate(False)

        # Helper to create a stylish HUD card with hover glow
        def create_card_frame(title_text, is_clickable=False, on_click_cmd=None):
            card = tk.Frame(
                info_strip,
                bg="#050505",
                highlightbackground="#102530",
                highlightthickness=1,
                padx=10,
                pady=6,
                cursor="hand2" if is_clickable else "arrow"
            )
            card.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)

            title_row = tk.Frame(card, bg="#050505")
            title_row.pack(fill=tk.X, pady=(0, 3))

            title_lbl = tk.Label(
                title_row,
                text=title_text,
                bg="#050505",
                fg="#8B96A5",
                font=("Segoe UI", 8, "bold")
            )
            title_lbl.pack(side=tk.LEFT)

            if is_clickable:
                arrow_lbl = tk.Label(
                    title_row,
                    text="↗",
                    bg="#050505",
                    fg="#00EDFF",
                    font=("Segoe UI", 8, "bold")
                )
                arrow_lbl.pack(side=tk.RIGHT)

            body = tk.Frame(card, bg="#050505")
            body.pack(fill=tk.BOTH, expand=True)

            if is_clickable:
                def on_enter(e):
                    card.config(bg="#0A0A0A", highlightbackground=COLOR_CYAN_PRIMARY)
                    title_row.config(bg="#0A0A0A")
                    title_lbl.config(bg="#0A0A0A", fg="#00EDFF")
                    body.config(bg="#0A0A0A")
                    if 'arrow_lbl' in locals():
                        arrow_lbl.config(bg="#0A0A0A")
                    for child in body.winfo_children():
                        try:
                            child.config(bg="#0A0A0A")
                            for sub in child.winfo_children():
                                try:
                                    sub.config(bg="#0A0A0A")
                                except Exception:
                                    pass
                        except Exception:
                            pass

                def on_leave(e):
                    card.config(bg="#050505", highlightbackground="#102530")
                    title_row.config(bg="#050505")
                    title_lbl.config(bg="#050505", fg="#8B96A5")
                    body.config(bg="#050505")
                    if 'arrow_lbl' in locals():
                        arrow_lbl.config(bg="#050505")
                    for child in body.winfo_children():
                        try:
                            child.config(bg="#050505")
                            for sub in child.winfo_children():
                                try:
                                    sub.config(bg="#050505")
                                except Exception:
                                    pass
                        except Exception:
                            pass

                card.bind("<Enter>", on_enter, add="+")
                card.bind("<Leave>", on_leave, add="+")
                if on_click_cmd:
                    card.bind("<Button-1>", lambda e: on_click_cmd(), add="+")
                    title_lbl.bind("<Button-1>", lambda e: on_click_cmd(), add="+")
                    body.bind("<Button-1>", lambda e: on_click_cmd(), add="+")

            return card, body

        # --- CARD 1: RECENT HISTORY ---
        card1, body1 = create_card_frame("RECENT HISTORY", is_clickable=True, on_click_cmd=self.open_history_dialog)
        HUDTooltip(card1, "Open conversation history")
        self.info_history_labels = []
        for _ in range(3):
            lbl = tk.Label(body1, text="", bg="#050505", fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 8), anchor="w")
            lbl.pack(fill=tk.X, pady=0)
            self.info_history_labels.append(lbl)

        # --- CARD 2: SYSTEM STATUS ---
        card2, body2 = create_card_frame("SYSTEM STATUS", is_clickable=False)
        self.info_status_labels = {}
        
        status_grid = tk.Frame(body2, bg="#050505")
        status_grid.pack(fill=tk.BOTH, expand=True)

        col_l = tk.Frame(status_grid, bg="#050505")
        col_l.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        row_cam = tk.Frame(col_l, bg="#050505")
        row_cam.pack(fill=tk.X)
        tk.Label(row_cam, text="Camera:", bg="#050505", fg="#8B96A5", font=("Segoe UI", 7), width=8, anchor="w").pack(side=tk.LEFT)
        self.info_status_labels["cam"] = tk.Label(row_cam, text="● Active", bg="#050505", fg=COLOR_STATUS_GREEN, font=("Segoe UI", 7, "bold"), anchor="w")
        self.info_status_labels["cam"].pack(side=tk.LEFT)

        row_mic = tk.Frame(col_l, bg="#050505")
        row_mic.pack(fill=tk.X)
        tk.Label(row_mic, text="Mic:", bg="#050505", fg="#8B96A5", font=("Segoe UI", 7), width=8, anchor="w").pack(side=tk.LEFT)
        self.info_status_labels["mic"] = tk.Label(row_mic, text="● Active", bg="#050505", fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 7, "bold"), anchor="w")
        self.info_status_labels["mic"].pack(side=tk.LEFT)

        row_spk = tk.Frame(col_l, bg="#050505")
        row_spk.pack(fill=tk.X)
        tk.Label(row_spk, text="Speaker:", bg="#050505", fg="#8B96A5", font=("Segoe UI", 7), width=8, anchor="w").pack(side=tk.LEFT)
        self.info_status_labels["speaker"] = tk.Label(row_spk, text="● Active", bg="#050505", fg=COLOR_STATUS_GREEN, font=("Segoe UI", 7, "bold"), anchor="w")
        self.info_status_labels["speaker"].pack(side=tk.LEFT)

        col_r = tk.Frame(status_grid, bg="#050505")
        col_r.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        row_gem = tk.Frame(col_r, bg="#050505")
        row_gem.pack(fill=tk.X)
        tk.Label(row_gem, text="Gemini:", bg="#050505", fg="#8B96A5", font=("Segoe UI", 7), width=8, anchor="w").pack(side=tk.LEFT)
        self.info_status_labels["gemini"] = tk.Label(row_gem, text="● Ready", bg="#050505", fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 7, "bold"), anchor="w")
        self.info_status_labels["gemini"].pack(side=tk.LEFT)

        row_bat = tk.Frame(col_r, bg="#050505")
        row_bat.pack(fill=tk.X)
        tk.Label(row_bat, text="Battery:", bg="#050505", fg="#8B96A5", font=("Segoe UI", 7), width=8, anchor="w").pack(side=tk.LEFT)
        self.info_status_labels["battery"] = tk.Label(row_bat, text="100%", bg="#050505", fg=COLOR_STATUS_GREEN, font=("Segoe UI", 7, "bold"), anchor="w")
        self.info_status_labels["battery"].pack(side=tk.LEFT)

        # --- CARD 3: CURRENT USER / TIME ---
        card3, body3 = create_card_frame("USER & TIME", is_clickable=False)
        self.info_user_title_lbl = tk.Label(body3, text="GOOD MORNING", bg="#050505", fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 8, "bold"), anchor="w")
        self.info_user_title_lbl.pack(fill=tk.X)

        self.info_user_name_lbl = tk.Label(body3, text="User", bg="#050505", fg="#F1F5F9", font=("Segoe UI", 9, "bold"), anchor="w")
        self.info_user_name_lbl.pack(fill=tk.X, pady=(0, 1))

        time_row = tk.Frame(body3, bg="#050505")
        time_row.pack(fill=tk.X)
        self.info_user_time_lbl = tk.Label(time_row, text="--:--", bg="#050505", fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 8, "bold"))
        self.info_user_time_lbl.pack(side=tk.LEFT)
        self.info_user_day_lbl = tk.Label(time_row, text="", bg="#050505", fg="#8B96A5", font=("Segoe UI", 7))
        self.info_user_day_lbl.pack(side=tk.LEFT, padx=(4, 0))

        # --- CARD 4: QUICK MEMORY ---
        card4, body4 = create_card_frame("QUICK MEMORY", is_clickable=True, on_click_cmd=self.open_memory_dialog)
        HUDTooltip(card4, "Open personal memories")
        self.info_memory_labels = []
        for _ in range(3):
            lbl = tk.Label(body4, text="", bg="#050505", fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 8), anchor="w")
            lbl.pack(fill=tk.X, pady=0)
            self.info_memory_labels.append(lbl)

    def _update_info_strip(self):
        """ Periodically and reactively updates the 4 Real Information Strip cards """
        if not hasattr(self, 'root') or not self.root:
            return

        # 1. Update Card 1: RECENT HISTORY
        try:
            if hasattr(self, 'info_history_labels') and self.info_history_labels and hasattr(self, 'engine') and self.engine.history:
                with self.engine.history._get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.execute("SELECT timestamp, sender, text FROM messages ORDER BY id DESC LIMIT 3")
                    rows = [dict(r) for r in cursor.fetchall()]

                for i, lbl in enumerate(self.info_history_labels):
                    if i < len(rows):
                        r = rows[i]
                        ts = r.get("timestamp", 0)
                        t_str = time.strftime("%H:%M", time.localtime(ts)) if ts else "--:--"
                        txt = r.get("text", "")
                        clean_txt = (txt[:20] + "...") if len(txt) > 20 else txt
                        prefix = "You: " if r.get("sender") == "user" else "AI: "
                        lbl.config(text=f"{t_str} {prefix}{clean_txt}", fg=COLOR_TEXT_PRIMARY)
                    else:
                        if i == 0 and not rows:
                            lbl.config(text="No recent history", fg=COLOR_TEXT_MUTED)
                        else:
                            lbl.config(text="", fg=COLOR_TEXT_MUTED)
        except Exception:
            pass

        # 2. Update Card 2: SYSTEM STATUS
        try:
            if hasattr(self, 'info_status_labels') and self.info_status_labels:
                cam_active = getattr(self, 'camera_running', True) and not getattr(self, 'camera_paused', False)
                self.info_status_labels["cam"].config(
                    text="● Active" if cam_active else "● Paused",
                    fg=COLOR_STATUS_GREEN if cam_active else COLOR_WARNING_GOLD
                )

                mic_active = getattr(self, 'mic_running', True)
                is_listening = getattr(self, 'current_state', 'IDLE') in ("LISTENING", "USER_SPEAKING")
                self.info_status_labels["mic"].config(
                    text="● Listening" if is_listening else ("● Active" if mic_active else "● Off"),
                    fg=COLOR_CYAN_PRIMARY if is_listening else (COLOR_STATUS_GREEN if mic_active else COLOR_ALERT_RED)
                )

                ai_active = getattr(self, 'ai_running', False)
                self.info_status_labels["gemini"].config(
                    text="● Connected" if ai_active else "● Ready",
                    fg=COLOR_STATUS_GREEN if ai_active else COLOR_CYAN_PRIMARY
                )

                is_speaking = getattr(self, 'current_state', 'IDLE') == "AI_SPEAKING"
                self.info_status_labels["speaker"].config(
                    text="● Speaking" if is_speaking else "● Active",
                    fg=COLOR_CYAN_PRIMARY if is_speaking else COLOR_STATUS_GREEN
                )

                try:
                    import psutil
                    bat = psutil.sensors_battery()
                    bat_txt = f"{int(bat.percent)}%" if bat else "AC Power"
                except Exception:
                    bat_txt = "100%"
                self.info_status_labels["battery"].config(text=bat_txt, fg=COLOR_STATUS_GREEN)
        except Exception:
            pass

        # 3. Update Card 3: CURRENT USER / TIME
        try:
            now = time.localtime()
            hour = now.tm_hour
            if 5 <= hour < 12:
                greeting_title = "GOOD MORNING"
            elif 12 <= hour < 17:
                greeting_title = "GOOD AFTERNOON"
            elif 17 <= hour < 22:
                greeting_title = "GOOD EVENING"
            else:
                greeting_title = "GOOD NIGHT"

            user_name = "User"
            if hasattr(self, 'user_profile') and self.user_profile and self.user_profile.get("user_name"):
                user_name = self.user_profile.get("user_name")
            elif hasattr(self, 'engine') and self.engine.store:
                user_name = self.engine.store.get_setting("user_display_name") or self.engine.store.get_setting("user_name") or "User"

            time_str = time.strftime("%I:%M %p", now).lstrip("0")
            day_str = time.strftime("%A, %b %d", now)

            if hasattr(self, 'info_user_title_lbl') and self.info_user_title_lbl:
                self.info_user_title_lbl.config(text=greeting_title)
            if hasattr(self, 'info_user_name_lbl') and self.info_user_name_lbl:
                self.info_user_name_lbl.config(text=user_name)
            if hasattr(self, 'info_user_time_lbl') and self.info_user_time_lbl:
                self.info_user_time_lbl.config(text=time_str)
            if hasattr(self, 'info_user_day_lbl') and self.info_user_day_lbl:
                self.info_user_day_lbl.config(text=day_str)
        except Exception:
            pass

        # 4. Update Card 4: QUICK MEMORY
        try:
            if hasattr(self, 'info_memory_labels') and self.info_memory_labels:
                mems = self.engine.memory.list_all_memories() if (hasattr(self, 'engine') and self.engine.memory) else []
                for i, lbl in enumerate(self.info_memory_labels):
                    if i < len(mems):
                        m = mems[i]
                        k = m.get("key_phrase", "").replace("_", " ").title()
                        v = m.get("fact_value", "")
                        clean_v = (v[:15] + "...") if len(v) > 15 else v
                        lbl.config(text=f"• {k}: {clean_v}", fg=COLOR_TEXT_PRIMARY)
                    else:
                        if i == 0 and not mems:
                            lbl.config(text="No saved memories", fg=COLOR_TEXT_MUTED)
                        else:
                            lbl.config(text="", fg=COLOR_TEXT_MUTED)
        except Exception:
            pass

        try:
            self.root.after(1000, self._update_info_strip)
        except tk.TclError:
            pass

    def _create_nav_btn(self, parent, icon_name, text, command, active=False, accent=COLOR_CYAN_PRIMARY, tooltip=""):
        if active:
            fg_col = COLOR_TEXT_PRIMARY
            bg_col = "#071114"
            border_col = COLOR_CYAN_PRIMARY
            icon_img = GlowingHUDIcon.get_photo_image(icon_name, size=21, color_hex=accent, hover=True, master=parent)
        else:
            fg_col = COLOR_TEXT_SECONDARY
            bg_col = COLOR_BG_PRIMARY
            border_col = COLOR_BG_PRIMARY
            icon_img = GlowingHUDIcon.get_photo_image(icon_name, size=20, color_hex=accent, hover=False, master=parent)

        hover_icon = GlowingHUDIcon.get_photo_image(icon_name, size=21, color_hex=accent, hover=True, master=parent)

        btn = tk.Button(
            parent,
            text=f" {text}",
            image=icon_img,
            compound=tk.LEFT,
            font=("Segoe UI", 10, "bold" if active else "normal"),
            bg=bg_col,
            fg=fg_col,
            activebackground="#0A0A0A",
            activeforeground=accent,
            relief=tk.FLAT,
            bd=0,
            padx=14,
            pady=7,
            cursor="hand2",
            highlightbackground=border_col,
            highlightthickness=1,
            command=command
        )
        btn.image_normal = icon_img
        btn.image_hover = hover_icon
        btn.image = icon_img
        btn.pack(side=tk.LEFT, padx=4)

        if not active:
            def on_enter(e):
                btn.config(bg="#0A0A0A", fg="#ffffff", image=btn.image_hover, highlightbackground=accent)
            def on_leave(e):
                btn.config(bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_SECONDARY, image=btn.image_normal, highlightbackground=COLOR_BG_PRIMARY)
            btn.bind("<Enter>", on_enter, add="+")
            btn.bind("<Leave>", on_leave, add="+")

        if tooltip:
            HUDTooltip(btn, tooltip)
        return btn

    def _create_floating_action_btn(self, parent, icon_name, text, command, accent=COLOR_CYAN_PRIMARY, tooltip=""):
        normal_img = GlowingHUDIcon.get_photo_image(icon_name, size=24, color_hex=accent, hover=False, master=parent)
        hover_img = GlowingHUDIcon.get_photo_image(icon_name, size=26, color_hex=accent, hover=True, master=parent)

        btn = tk.Button(
            parent,
            text=text,
            image=normal_img,
            compound=tk.TOP,
            font=("Segoe UI", 8, "bold"),
            bg=COLOR_PANEL_DEEP,
            fg=COLOR_TEXT_PRIMARY,
            activebackground="#0A0A0A",
            activeforeground=accent,
            relief=tk.FLAT,
            bd=0,
            padx=10,
            pady=6,
            cursor="hand2",
            highlightbackground=COLOR_BORDER_SUBTLE,
            highlightthickness=1,
            command=command
        )
        btn.image_normal = normal_img
        btn.image_hover = hover_img
        btn.image = normal_img
        btn.pack(side=tk.LEFT, padx=4, pady=(2, 0))

        def on_enter(e):
            btn.config(bg="#0A0A0A", fg="#ffffff", image=btn.image_hover, highlightbackground=accent)
            btn.pack_configure(pady=(0, 2))
        def on_leave(e):
            btn.config(bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, image=btn.image_normal, highlightbackground=COLOR_BORDER_SUBTLE)
            btn.pack_configure(pady=(2, 0))
        def on_press(e):
            btn.config(bg=COLOR_BG_PRIMARY, highlightbackground=COLOR_BG_PRIMARY)
        def on_release(e):
            btn.config(bg="#0A0A0A", highlightbackground=accent)

        btn.bind("<Enter>", on_enter, add="+")
        btn.bind("<Leave>", on_leave, add="+")
        btn.bind("<Button-1>", on_press, add="+")
        btn.bind("<ButtonRelease-1>", on_release, add="+")

        if tooltip:
            HUDTooltip(btn, tooltip)
        return btn

    def _bind_shortcuts(self):
        self.root.bind("<space>", lambda e: self._on_space_shortcut())
        self.root.bind("<Escape>", lambda e: self._on_esc_shortcut())
        self.root.bind("<Control-Shift-S>", lambda e: self.open_settings_dialog())
        self.root.bind("<Control-s>", lambda e: self.open_settings_dialog())
        self.root.bind("<Control-m>", lambda e: self.open_memory_dialog())
        self.root.bind("<Control-h>", lambda e: self.open_history_dialog())
        self.root.bind("<Control-v>", lambda e: self.open_vision_dialog())
        self.root.bind("<Control-p>", lambda e: self.open_people_dialog())
        self.root.bind("<Control-q>", lambda e: self.on_close())

    def _on_nav_home(self):
        """ Home navigation tab: ensures primary companion dashboard is active """
        self.show_context_alert("SG CUBE Home Dashboard Active", color=COLOR_CYAN_PRIMARY)

    def _on_space_shortcut(self):
        """ Spacebar key shortcut: interrupts speech / triggers immediate attention """
        self._clear_playback_queue()
        self.set_state("LISTENING")
        self.show_context_alert("Listening...", color=COLOR_CYAN_PRIMARY)

    def _on_esc_shortcut(self):
        """ Escape key shortcut: immediately stops current AI speech """
        self._clear_playback_queue()
        self.set_state("LISTENING")
        self.show_context_alert("Speech stopped.", color=COLOR_TEXT_MUTED)

    def _trigger_action_async(self, action_key: str, display_label: str, intent_query: str, specific_handler=None):
        """ Debounced, non-blocking execution of floating action buttons with loading and audio feedback """
        with self.state_lock:
            if hasattr(self, 'action_busy') and self.action_busy:
                print(f"[ACTION-BTN] Operation '{display_label}' already in progress. Debounce active.")
                self.show_context_alert("Processing in progress...", color=COLOR_ORANGE)
                return
            self.action_busy = True

        print(f"[ACTION-BTN] Triggering action '{display_label}' ('{intent_query}')")
        self._clear_playback_queue()
        self.set_state("AI_THINKING")
        self.gui_queue.put(("TRANSCRIPT_USER", intent_query))
        self.show_context_alert(f"Processing {display_label}...", color=COLOR_CYAN_PRIMARY)

        def worker():
            try:
                if specific_handler:
                    resp = specific_handler()
                else:
                    resp = self.engine.process_user_speech_query(intent_query)

                if resp:
                    print(f"[ACTION-RESULT] {display_label}: '{resp}'")
                    self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", resp))
                    if hasattr(self, 'active_history_session_id') and self.active_history_session_id:
                        self.engine.history.add_message(self.active_history_session_id, "user", intent_query)
                        self.engine.history.add_message(self.active_history_session_id, "assistant", resp)
                    self.engine.response_manager.add_response(resp, priority=1, force=True)
                else:
                    err_msg = f"No result returned for {display_label}."
                    self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", err_msg))
            except Exception as e:
                print(f"[ACTION-ERR] Error in {display_label}: {e}")
                self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", f"Error processing {display_label}: {e}"))
            finally:
                with self.state_lock:
                    self.action_busy = False
                if self.current_state not in ("SLEEPING", "STOPPED", "CLOSED"):
                    self.set_state("LISTENING")

        threading.Thread(target=worker, daemon=True).start()

    def _trigger_voice_intent(self, text_query: str):
        """ Programmatically triggers a voice query from floating action buttons """
        self._trigger_action_async("voice_intent", "Voice Intent", text_query)

    # --- Real 3D Cross-Axis Rotating SG CUBE Assistant Rendering ---
    def _animate_sg_cube_core(self):
        if not self.root or not self.orb_canvas:
            return

        self.anim_time += 0.035
        self.anim_angle += 0.08
        if self.anim_angle > 2 * math.pi:
            self.anim_angle = 0.0

        state = self.current_state

        # Multi-axis / Cross-axis rotation speeds mapped from backend states
        rot_y_delta = 0.022
        pulse = 1.0
        if state == "AI_SPEAKING":
            rot_y_delta = 0.040
            pulse = 1.0 + 0.09 * math.sin(self.anim_angle * 3.0)
            status_txt = "● SG CUBE Speaking..."
            sys_txt = "● Speaking"
            sys_col = COLOR_TEAL_MINT
        elif state == "USER_SPEAKING":
            rot_y_delta = 0.032
            pulse = 1.0 + 0.06 * math.sin(self.anim_angle * 2.5)
            status_txt = "● Listening to you..."
            sys_txt = "● User Speaking"
            sys_col = COLOR_PURPLE
        elif state == "AI_THINKING":
            rot_y_delta = 0.018
            pulse = 1.0 + 0.04 * math.sin(self.anim_angle * 1.5)
            status_txt = "● Thinking..."
            sys_txt = "● Thinking"
            sys_col = COLOR_WARNING_GOLD
        elif state == "SAFETY_ALERT":
            rot_y_delta = 0.048
            pulse = 1.0 + 0.10 * math.sin(self.anim_angle * 4.0)
            status_txt = "⚠ Physical Hazard Alert"
            sys_txt = "⚠ Hazard Alert"
            sys_col = COLOR_ALERT_RED
        elif state == "RECONNECTING":
            rot_y_delta = 0.025
            status_txt = "● Reconnecting..."
            sys_txt = "● Reconnecting"
            sys_col = COLOR_ORANGE
        elif state == "SLEEPING":
            rot_y_delta = 0.0
            pulse = 0.90
            status_txt = "● Sleeping (Say 'Hey SG CUBE')"
            sys_txt = "● Sleeping"
            sys_col = COLOR_TEXT_MUTED
        else:  # LISTENING / IDLE
            rot_y_delta = 0.022 if state == "IDLE" else 0.030
            pulse = 1.0 + 0.04 * math.sin(self.anim_angle)
            status_txt = "● SG CUBE Listening"
            sys_txt = "● Listening" if state == "LISTENING" else "● Online"
            sys_col = COLOR_CYAN_PRIMARY if state == "LISTENING" else COLOR_STATUS_GREEN

        # Multi-axis floating motion: Y=11s continuous, X=6s oscillation -8°..+8°, Z=8.5s subtle -2°..+2°
        if state != "SLEEPING":
            self.cube_rot_y += (2 * math.pi) / (11.0 / 0.035)  # 11s full rotation period
            if self.cube_rot_y > 2 * math.pi:
                self.cube_rot_y -= 2 * math.pi

        rot_x = 0.1396 * math.sin(self.anim_time * (2 * math.pi / 6.0))  # -8° to +8° over 6s
        rot_z = 0.0349 * math.cos(self.anim_time * (2 * math.pi / 8.5))  # -2° to +2° over 8.5s

        # Update Header Status Label
        if hasattr(self, 'sys_status_label') and self.sys_status_label:
            self.sys_status_label.config(text=sys_txt, fg=sys_col)

        # Update Context Banner text if no active alert
        if self.context_banner_timer <= 0:
            self.context_banner.config(text=status_txt, fg=COLOR_CYAN_PRIMARY if state != "SAFETY_ALERT" else COLOR_ALERT_RED)
        else:
            self.context_banner_timer -= 1

        # Clear canvas
        self.orb_canvas.delete("all")

        # Render Futuristic AI Audio / Speaker Core
        self.cube_3d.draw_audio_core(
            self.orb_canvas,
            time_val=self.anim_time,
            state=state,
            pulse=pulse,
            hover=self.cube_hover,
            mouse_tilt=self.cube_mouse_tilt
        )

        try:
            self.root.after(35, self._animate_sg_cube_core)
        except tk.TclError:
            pass

    # --- GUI Queue Processing (Main Thread) ---
    def _process_gui_queue(self):
        while not self.gui_queue.empty():
            try:
                msg_type, payload = self.gui_queue.get_nowait()
                if msg_type == "FRAME":
                    photo_img = ImageTk.PhotoImage(image=payload)
                    self.preview_label.config(image=photo_img, text="")
                    self.preview_label.image = photo_img
                elif msg_type == "ACTION":
                    if payload == "WAKE_FOREGROUND":
                        self.bring_to_foreground()
                elif msg_type == "STATE":
                    pass
                elif msg_type == "PERCEPTION":
                    if "Hazards:" in payload and "Clear" not in payload:
                        self.show_context_alert(payload, color=COLOR_ALERT_RED)
                elif msg_type == "TRANSCRIPT_USER":
                    self.dialogue_banner.config(text=f"You: \"{payload}\"", fg=COLOR_CYAN_PRIMARY)
                elif msg_type == "TRANSCRIPT_AI":
                    self.dialogue_banner.config(text=f"SG CUBE: \"{payload}\"", fg=COLOR_TEAL_MINT)
                elif msg_type == "TRANSCRIPT_ASSISTIVE":
                    self.dialogue_banner.config(text=f"Assistive: \"{payload}\"", fg=COLOR_WARNING_GOLD)
                    self.show_context_alert(payload, color=COLOR_CYAN_PRIMARY)
                elif msg_type == "CAMERA_STATUS":
                    if payload and payload.startswith("● LIVE"):
                        badge_text = payload.replace("● LIVE ", "● ")
                        self.cam_badge.config(text=badge_text, bg=COLOR_PANEL_DEEP, fg=COLOR_TEAL_MINT)
                    elif payload == "● META GLASS":
                        self.cam_badge.config(text="● META GLASS", bg=COLOR_CYAN_PRIMARY, fg="#000000")
                    elif payload == "● RECONNECTING":
                        self.cam_badge.config(text="● RECONNECTING", bg=COLOR_ALERT_RED, fg="#ffffff")
                elif msg_type == "HUD_UPDATE":
                    self._update_hud_display(payload)
                elif msg_type == "CAMERA_STOPPED":
                    self.cam_badge.config(text="OFFLINE", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_MUTED)
                    self.preview_label.config(image="", text="Camera View Offline")
            except queue.Empty:
                break
        if self.root:
            try:
                self.root.after(20, self._process_gui_queue)
            except tk.TclError:
                pass

    def _update_hud_display(self, data: dict):
        """ Updates the Live Vision left (Environment) and right (Objects) HUD cards with real backend data """
        if not hasattr(self, 'hud_env_person_lbl') or not self.hud_env_person_lbl:
            return

        try:
            # 1. Update Environment Card
            env = data.get("environment", {})
            faces = data.get("faces", [])
            safety = data.get("safety", {})

            # Person
            if faces:
                names = [f.get("name") or "Person" for f in faces]
                self.hud_env_person_lbl.config(text=f"● Person: {', '.join(names[:2])}", fg="#F1F5F9")
            else:
                self.hud_env_person_lbl.config(text="● Person: None", fg="#8B96A5")

            # Room / Scene
            scene_text = env.get("scene_summary", "Clear space")
            clean_scene = (scene_text[:22] + "...") if len(scene_text) > 22 else scene_text
            self.hud_env_room_lbl.config(text=f"▣ Room: {clean_scene}", fg="#8B96A5")

            # Light
            light_lvl = env.get("light_level", "NORMAL")
            self.hud_env_light_lbl.config(text=f"☼ Light: {light_lvl.title()}", fg="#4DF7C4")

            # Safety
            if safety.get("hazard_detected"):
                warn_txt = safety.get("warning_text", "Hazard detected")
                clean_warn = (warn_txt[:20] + "...") if len(warn_txt) > 20 else warn_txt
                self.hud_env_safety_lbl.config(text=f"⚠ {clean_warn}", fg=COLOR_ALERT_RED)
            else:
                self.hud_env_safety_lbl.config(text="⚠ Safety: Clear", fg=COLOR_STATUS_GREEN)

            # 2. Update Objects Card
            objects = data.get("objects", [])
            if hasattr(self, 'hud_obj_labels') and self.hud_obj_labels:
                for idx, lbl in enumerate(self.hud_obj_labels):
                    if idx < len(objects):
                        obj = objects[idx]
                        sp = obj.get("spatial", {})
                        dist_verb = sp.get("distance_verbal", "near")
                        h_verb = sp.get("h_zone", "center")
                        lbl.config(text=f"• Item {idx+1}: {h_verb.title()} ({dist_verb})", fg="#F1F5F9")
                    else:
                        if idx == 0 and not objects:
                            lbl.config(text="No objects in view", fg="#8B96A5")
                        else:
                            lbl.config(text="", fg="#8B96A5")
        except Exception:
            pass

    def show_context_alert(self, text: str, color: str = COLOR_CYAN_PRIMARY):
        """ Temporarily displays a contextual detection banner near the companion core """
        self.context_banner.config(text=f"● {text}", fg=color)
        self.context_banner_timer = 120  # ~4 seconds duration

    # --- Seamless Auto-Start Lifecycle ---
    def _auto_start_services(self):
        """ Auto-starts camera, audio stream, and Gemini Live in the background """
        print("[AUTO-START] Beginning seamless background initialization...")
        self._notify_wake_listener_pause()
        self.set_state("INITIALIZING")

        self.start_camera()

        api_key = self.engine.key_manager.load_api_key()
        if not api_key:
            self.dialogue_banner.config(text="Gemini API key is not configured. Please open Settings (Ctrl+Shift+S).", fg="#ff4757")
            print("[AUTO-START] Gemini API key not configured. Open Settings to configure.")
            return

        self.start_ai(api_key)

        with self.state_lock:
            self.wake_greeting_pending = True
            self.wake_greeting_timer = time.time()

    def start_camera(self):
        """ Starts or restores the continuous camera capture worker thread """
        with self.session_lock:
            if self.camera_running and self.camera_thread and self.camera_thread.is_alive():
                print("[CAMERA] Camera already active.")
                return
            print("[CAMERA] initialization started")
            self.camera_running = True
            self.first_valid_frame_received = False
            self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
            self.camera_thread.start()
            print("[CAMERA] worker started")

    def stop_camera(self):
        """ Safe teardown of camera capture thread (called on app exit or sleep) """
        print("[CAMERA] Stopping camera service worker thread...")
        self.camera_running = False
        if self.camera_thread and self.camera_thread.is_alive():
            self.camera_thread.join(timeout=1.5)
        self.camera_thread = None
        self.gui_queue.put(("CAMERA_STOPPED", None))
        print("[CAMERA] hardware released")
        print("[CAMERA] camera state = SLEEPING / OFF")

    def start_ai(self, api_key=None):
        if self.ai_running:
            return
        if not api_key:
            key_info = self.engine.key_manager.get_active_key()
            if not key_info:
                print("[API-KEY-FAILOVER] No available valid Gemini API keys found.")
                self.set_state("DISCONNECTED")
                return
            _, api_key = key_info

        self.ai_running = True
        print(f"[MAIN] MICROPHONE STARTED ({self.engine.key_manager.get_active_key_label()})")
        self.ai_thread = threading.Thread(target=self._ai_worker_thread, args=(api_key,), daemon=True)
        self.ai_thread.start()

    def stop_ai(self):
        if not self.ai_running:
            return
        self.ai_running = False
        print("[MAIN] MICROPHONE STOPPED")
        with self.session_lock:
            self.active_session_id = None
        self.playback_stop_evt.set()
        self._clear_playback_queue()
        self.set_state("STOPPED")

    def enter_sleep_mode(self):
        """ Puts SG CUBE to sleep after playing one short farewell greeting """
        print("[SLEEP] requested")

        # Generate personalized or generic sleep farewell greeting
        user_name = self.engine.store.get_setting("user_display_name") or self.engine.store.get_setting("user_name", "")
        name_str = f" {user_name}" if user_name and user_name != "User" else ""
        farewell_msg = f"Okay{name_str}, I'm going to sleep."
        self.last_farewell_msg = farewell_msg

        print(f"[SLEEP-GREETING] Farewell message generated: '{farewell_msg}'")
        self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", farewell_msg))

        # Queue farewell response in ResponseManager & playback
        self.engine.response_manager.add_response(farewell_msg, priority=2, force=True)

        # Wait up to 2.5s for audio playback queue to drain before shutting down services
        start_wait = time.time()
        while not self.playback_queue.empty() and (time.time() - start_wait < 2.5):
            time.sleep(0.05)

        time.sleep(0.3)

        print("[SLEEP] stopping speech")
        self._clear_playback_queue()
        print("[SLEEP] audio cleared")

        print("[SLEEP] Gemini session closed")
        print("[SLEEP] releasing microphone")
        self.stop_ai()

        print("[SLEEP] releasing camera")
        if hasattr(self.engine, 'meta_glass'):
            self.engine.meta_glass.on_sleep()
        self.stop_camera()

        self.set_state("SLEEPING")
        print("[STATE] SLEEPING")
        print("[HANDOFF] MAIN -> HOTWORD")
        print("[MIC_OWNER] HOTWORD")
        print("[HOTWORD] MICROPHONE ACTIVE")
        print("[HOTWORD] READY")
        print("[SLEEP] SG CUBE inactive")

        try:
            self.root.withdraw()
            print("[SLEEP] SG CUBE main window hidden/withdrawn.")
        except Exception as e:
            print(f"[SLEEP] Error withdrawing GUI window: {e}")

        self.root.after(400, self._notify_wake_listener_resume)

    def _ensure_wake_listener_running(self):
        """ Ensures background wake listener is running when main app sleeps or exits. """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.4)
            res = sock.connect_ex(('127.0.0.1', 49154))  # listener lock socket
            if res == 0:
                return  # already running
        except Exception:
            pass
        finally:
            if sock:
                try: sock.close()
                except Exception: pass

        try:
            app_dir = os.path.abspath(os.path.dirname(__file__))
            python_candidates = [
                os.path.join(app_dir, "runtime", "Scripts", "python.exe"),
                os.path.join(app_dir, "runtime", "python.exe"),
                sys.executable,
                os.path.join(app_dir, ".venv", "Scripts", "python.exe"),
            ]
            python_exe = sys.executable
            for p in python_candidates:
                if os.path.exists(p):
                    python_exe = p
                    break

            listener_script = os.path.join(app_dir, "wake_listener.py")
            if os.path.exists(listener_script):
                DETACHED_PROCESS = 0x00000008
                CREATE_NEW_PROCESS_GROUP = 0x00000200
                subprocess.Popen(
                    [python_exe, listener_script],
                    cwd=app_dir,
                    creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
                    close_fds=True
                )
                print("[LIFECYCLE] Background wake listener started.")
        except Exception as e:
            print(f"[LIFECYCLE] Error starting wake listener: {e}")

    def _notify_wake_listener_resume(self):
        """ Sends IPC signal over port 49153 to instruct the background wake listener to resume microphone standby """
        sock = None
        notified = False
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.8)
            sock.connect(('127.0.0.1', IPC_PORT_WAKE_LISTENER))
            sock.sendall(b"RESUME_WAKE_LISTENING\n")
            notified = True
            print("[IPC] Successfully notified background wake listener to resume.")
        except Exception:
            pass
        finally:
            if sock:
                try: sock.close()
                except Exception: pass

        if not notified:
            self._ensure_wake_listener_running()

    def _notify_wake_listener_pause(self):
        """ Sends IPC signal over port 49153 to instruct the background wake listener to pause microphone standby """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(0.8)
            sock.connect(('127.0.0.1', IPC_PORT_WAKE_LISTENER))
            sock.sendall(b"PAUSE_WAKE_LISTENING\n")
            print("[IPC] Successfully notified background wake listener to pause.")
        except Exception:
            pass
        finally:
            if sock:
                try: sock.close()
                except Exception: pass

    def bring_to_foreground(self):
        """ Restores window from sleep/hidden state, auto-starts camera & AI services, and triggers person-aware greeting """
        self._notify_wake_listener_pause()
        self.set_state("OPENING")
        print("[WAKE] MATCH")
        print("[HANDOFF] HOTWORD -> MAIN")
        print("[MIC_OWNER] HOTWORD RELEASED")
        print("[STATE] OPENING")
        print("[WAKE] detected")
        print("[WAKE] activation started")
        try:
            self.root.deiconify()
            self.root.lift()
            self.root.focus_force()

            if os.name == 'nt':
                try:
                    import win32gui, win32con
                    hwnd = int(self.root.wm_frame(), 16)
                    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
                    win32gui.SetForegroundWindow(hwnd)
                except Exception:
                    pass
            print("[WAKE] application visible")
        except Exception as e:
            print(f"[WAKE] Error bringing to foreground: {e}")

        if hasattr(self.engine, 'meta_glass'):
            self.engine.meta_glass.on_wake()

        if not self.camera_running:
            print("[START] camera initializing")
            self.start_camera()

        api_key = self.engine.key_manager.load_api_key()
        if api_key and not self.ai_running:
            print("[START] microphone initializing")
            print("[START] Gemini connecting")
            self.start_ai(api_key)

        print("[MIC_OWNER] MAIN ACTIVE")
        print("[WAKE] APPLICATION READY")
        self.set_state("LISTENING")
        print("[STATE] ACTIVE / LISTENING")
        print("[START] ready")
        print("[START] listening")

        with self.state_lock:
            self.wake_greeting_pending = True
            self.wake_greeting_timer = time.time()
            self.first_valid_frame_received = False

    def _evaluate_wake_greeting(self, faces: list):
        """ Evaluates camera faces upon wake/startup and queues exactly one person-aware greeting """
        if not self.wake_greeting_pending:
            return

        if not self.engine.store.get_setting("greeting_enabled", True):
            self.wake_greeting_pending = False
            return

        now = time.time()
        elapsed = now - self.wake_greeting_timer

        # Wait for first valid frame from camera loop unless timeout reached
        if not self.first_valid_frame_received and elapsed < 2.5:
            return

        print("[GREETING 01] application startup detected")

        owner_name = self.engine.store.get_setting("user_display_name") or self.engine.store.get_setting("user_name", "")
        print("[GREETING 02] profile loaded")
        safe_name = owner_name if owner_name else "Neutral"
        print(f"[GREETING 03] user display name = {safe_name}")

        hour = time.localtime().tm_hour
        time_period = "morning" if 5 <= hour < 12 else ("afternoon" if 12 <= hour < 17 else ("evening" if 17 <= hour < 22 else "night"))
        print(f"[GREETING 04] time-of-day calculated ({time_period})")

        print(f"[FACE] frame received (faces_count={len(faces)})")
        print(f"[FACE] detection count = {len(faces)}")
        print("[FACE] recognition started")

        chosen_greeting = None

        if faces:
            recognized_name = None
            recognized_conf = 0.0
            for face in faces:
                name = face.get("name")
                if name and name != "Unknown":
                    recognized_name = name
                    recognized_conf = face.get("confidence", 1.0)
                    break

            if recognized_name:
                print(f"[FACE] best match = {recognized_name}")
                print(f"[FACE] confidence = {recognized_conf:.2f}")
                print(f"[FACE] stored identity = {recognized_name}")
                if owner_name and recognized_name.lower() == owner_name.lower():
                    chosen_greeting = f"Good {time_period}, {owner_name}. I'm ready to help."
                else:
                    chosen_greeting = f"Hello, {recognized_name}."
            else:
                print("[FACE] best match = Unknown")
                print("[FACE] confidence = 0.00")
                print("[FACE] stored identity = Unknown")
                chosen_greeting = "Hello. I'm SG CUBE. How can I help?"

        elif elapsed > 2.5:
            print("[FACE] no face detected within 2.5s timeout")
            if owner_name and owner_name != "User":
                chosen_greeting = f"Good {time_period}, {owner_name}. I'm ready to help."
            else:
                chosen_greeting = "Hello. I'm SG CUBE. How can I help?"

        if chosen_greeting:
            self.wake_greeting_pending = False
            print(f"[GREETING 05] greeting text generated: '{chosen_greeting}'")
            print("[GREETING 06] greeting queued")
            self.engine.response_manager.add_response(chosen_greeting, priority=2, force=True)
            self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", chosen_greeting))

            with self.session_lock:
                self.pending_speech_prompt = f"Speak this exact startup greeting out loud in a warm, natural, friendly, confident voice: '{chosen_greeting}'"

    def _on_ai_stopped(self, reason=None):
        self.ai_running = False
        self._clear_playback_queue()
        self.set_state("IDLE")

    # --- Background Camera Worker Thread ---
    def _camera_loop(self):
        """
        Continuous, non-blocking camera capture worker with bounded latest-frame strategy.
        Decouples hardware capture from AI inference so preview rendering and capture remain smooth.
        Includes automatic retry & reconnect backoff if physical camera read fails.
        """
        print("[SENSOR] camera initialization started")
        cam_idx_setting = self.engine.store.get_setting("camera_index", 0)
        target_fps = 30.0
        frame_interval = 1.0 / target_fps
        consecutive_errors = 0

        # Perception cadence (10-12 Hz) to ensure continuous, smooth camera capture
        perception_interval = 1.0 / 12.0
        last_perception_time = 0.0
        cached_frame_info = {
            "faces": [],
            "safety": {},
            "environment": getattr(self.engine, 'last_environment', {}),
            "objects": getattr(self.engine, 'last_objects', [])
        }

        # Rolling FPS tracking
        frame_times = []
        last_status_update_time = 0.0

        while self.camera_running:
            cap = None
            for cam_idx in (cam_idx_setting, 0, 1):
                temp_cap = cv2.VideoCapture(cam_idx, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
                if temp_cap.isOpened():
                    cap = temp_cap
                    print(f"[CAMERA] device opened (index {cam_idx})")
                    print(f"[SENSOR] camera opened (index {cam_idx})")
                    break

            if not cap or not cap.isOpened():
                consecutive_errors += 1
                print(f"[CAMERA-WARN] Unable to open camera device (attempt {consecutive_errors}). Retrying in 2.0s...")
                self.gui_queue.put(("CAMERA_STATUS", "● RECONNECTING"))
                time.sleep(2.0)
                if consecutive_errors > 10:
                    self.gui_queue.put(("CAMERA_STOPPED", None))
                continue

            print("[CAMERA] Camera capture stream opened successfully.")
            print("[CAMERA] worker started")
            print("[SENSOR] camera worker started")
            self.gui_queue.put(("CAMERA_STATUS", "● LIVE 20 FPS"))
            consecutive_errors = 0

            try:
                while self.camera_running:
                    start_time = time.time()

                    # Real Camera Source Selection (Laptop Camera vs Meta Glass)
                    is_glass_active = hasattr(self.engine, 'meta_glass') and self.engine.meta_glass.active_source == "META_GLASS"
                    if is_glass_active and self.engine.meta_glass.is_streaming() and self.engine.meta_glass.latest_frame is not None:
                        frame = self.engine.meta_glass.latest_frame
                        ret = True
                        self.gui_queue.put(("CAMERA_STATUS", "● META GLASS"))
                    else:
                        try:
                            ret, frame = cap.read()
                        except Exception:
                            ret, frame = False, None
                        if is_glass_active:
                            # Glass selected but not streaming -> fallback to laptop
                            self.engine.meta_glass.active_source = "LAPTOP"

                    if not ret or frame is None or frame.size == 0:
                        consecutive_errors += 1
                        if consecutive_errors == 1:
                            print("[CAMERA] FRAME_ERROR")
                        print(f"[CAMERA-WARN] Frame read failed ({consecutive_errors}/5 consecutive failures).")
                        if consecutive_errors >= 5:
                            print("[CAMERA] RECOVERING")
                            print("[CAMERA-RECOVER] Consecutive read failures exceeded threshold. Re-opening camera device...")
                            break
                        time.sleep(0.02)
                        continue

                    if not self.first_valid_frame_received:
                        self.first_valid_frame_received = True
                        print(f"[CAMERA] first valid frame received")
                        print(f"[CAMERA] ACTIVE")
                        print(f"[SENSOR] first valid frame received (shape={frame.shape}, ts={time.time():.3f})")

                    consecutive_errors = 0
                    now = time.time()

                    # Update Rolling FPS Tracking
                    frame_times.append(now)
                    if len(frame_times) > 20:
                        frame_times.pop(0)

                    if len(frame_times) > 1:
                        dt = frame_times[-1] - frame_times[0]
                        if dt > 0:
                            self.measured_fps = (len(frame_times) - 1) / dt

                    # Update Telemetry Status at ~2 Hz
                    if now - last_status_update_time >= 0.5:
                        last_status_update_time = now
                        fps_display = int(round(self.measured_fps)) if self.measured_fps > 0 else 20
                        if not is_glass_active:
                            self.gui_queue.put(("CAMERA_STATUS", f"● LIVE {fps_display} FPS"))

                    # Store freshest raw frame for explicit user actions (OCR, Object Search, etc.)
                    with self.frame_lock:
                        self.latest_raw_frame = frame

                    # Cadenced Background Perception (Face Detection, Safety, Scene, Objects)
                    if now - last_perception_time >= perception_interval:
                        last_perception_time = now
                        try:
                            cached_frame_info = self.engine.process_frame(frame)
                        except Exception as frame_e:
                            print(f"[CAMERA-WARN] Frame processing exception caught: {frame_e}")
                            cached_frame_info = {"faces": [], "safety": {}, "scene": {}}

                    # Process in-flight assistant speech
                    next_speech = self.engine.response_manager.get_next_response()
                    if next_speech:
                        self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", next_speech))

                    # Encode JPEG for Gemini Live WebSocket stream
                    ret_jpg, jpg_buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                    if ret_jpg:
                        with self.frame_lock:
                            self.latest_jpeg = jpg_buffer.tobytes()

                    # Prepare HUD preview rendering
                    preview_frame = cv2.resize(frame, (640, 420))

                    faces = cached_frame_info.get("faces", [])
                    face_summary = []
                    for face in faces:
                        x, y, w, h = face["bbox"]
                        name = face.get("name") or "Unknown"
                        face_summary.append(name)
                        if self.dev_mode:
                            scale_x = 640 / float(frame.shape[1])
                            scale_y = 420 / float(frame.shape[0])
                            px, py, pw, ph = int(x * scale_x), int(y * scale_y), int(w * scale_x), int(h * scale_y)
                            is_known = bool(face.get("name") and face.get("name") != "Unknown")
                            # Mint outline for known (#4DF7C4 -> BGR: 196, 247, 77), Amber outline for unknown (#FDAB72 -> BGR: 114, 171, 253)
                            color = (196, 247, 77) if is_known else (114, 171, 253)
                            cv2.rectangle(preview_frame, (px, py), (px + pw, py + ph), color, 2)
                            # Name tag on #050505 background with matching border & text
                            tag_text = f" {name} "
                            (tw, th), _ = cv2.getTextSize(tag_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                            tag_y1 = max(0, py - th - 8)
                            tag_y2 = py
                            cv2.rectangle(preview_frame, (px, tag_y1), (px + tw + 6, tag_y2), (5, 5, 5), -1)
                            cv2.rectangle(preview_frame, (px, tag_y1), (px + tw + 6, tag_y2), color, 1)
                            cv2.putText(preview_frame, tag_text, (px + 2, py - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

                    self._evaluate_wake_greeting(faces)

                    safety = cached_frame_info.get("safety", {})
                    hazard_desc = safety.get("warning_text", "Clear") if safety.get("hazard_detected") else "Clear"

                    perception_text = f"Faces: {', '.join(face_summary) if face_summary else 'None'} | Hazards: {hazard_desc}"
                    self.gui_queue.put(("PERCEPTION", perception_text))

                    # Send live vision HUD telemetry (Environment & Objects)
                    hud_payload = {
                        "faces": faces,
                        "safety": safety,
                        "environment": cached_frame_info.get("environment", {}),
                        "objects": cached_frame_info.get("objects", [])
                    }
                    self.gui_queue.put(("HUD_UPDATE", hud_payload))

                    cv2_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(cv2_rgb)

                    self.gui_queue.put(("FRAME", pil_img))

                    elapsed = time.time() - start_time
                    sleep_time = max(0.0, frame_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            finally:
                if cap:
                    try:
                        cap.release()
                    except Exception:
                        pass

        print("[CAMERA] Camera capture loop terminated cleanly.")
        with self.frame_lock:
            self.latest_jpeg = None
            self.latest_raw_frame = None
        self.gui_queue.put(("CAMERA_STOPPED", None))

    # --- Background AI Worker Thread & Single Authoritative Audio Loop ---
    def _ensure_single_playback_thread(self):
        """ Spawns exactly ONE dedicated speaker playback worker thread if not already running """
        with self.session_lock:
            if not self.playback_thread_started or self.playback_thread is None or not self.playback_thread.is_alive():
                self.playback_stop_evt.clear()
                self.playback_thread = threading.Thread(
                    target=self._audio_playback_loop,
                    args=(self.playback_stop_evt,),
                    daemon=True
                )
                self.playback_thread.start()
                self.playback_thread_started = True
                print("[PLAYBACK] Single authoritative speaker worker thread active.")

    def _ai_worker_thread(self, api_key):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while not self.mic_queue.empty():
            try: self.mic_queue.get_nowait()
            except queue.Empty: break
        self._clear_playback_queue()

        self._ensure_single_playback_thread()

        mic_stream = None
        try:
            mic_cb_logged = False
            def mic_callback(indata, frames, time_info, status):
                nonlocal mic_cb_logged
                if not mic_cb_logged:
                    print("[SENSOR] microphone callback active")
                    mic_cb_logged = True
                if self.ai_running:
                    self.mic_queue.put(bytes(indata))

            print("[SENSOR] microphone initialization started")
            print("[MAIN-MIC] application acquired input device")
            mic_stream = sd.RawInputStream(
                samplerate=16000,
                channels=1,
                dtype='int16',
                blocksize=1024,
                callback=mic_callback
            )
            mic_stream.start()
            print("[MIC] Single 16kHz microphone stream opened.")
            print("[SENSOR] microphone opened")

            loop.run_until_complete(self._run_live_session(api_key))

        except Exception as e:
            print(f"[ERROR] AI worker thread exception: {e}")
            curr_key_num = self.engine.key_manager.active_key_num or 1
            next_key = self.engine.key_manager.get_next_failover_key(curr_key_num, error=e)
            if next_key and self.current_state not in ("SLEEPING", "STOPPED", "CLOSED"):
                next_num, next_val = next_key
                print(f"[FAILOVER] Key {curr_key_num} failed -> Failing over to Key {next_num}")
                self.stop_ai()
                try:
                    self.root.after(500, lambda: self.start_ai(next_val))
                except Exception:
                    threading.Timer(0.5, lambda: self.start_ai(next_val)).start()
            else:
                print("[FAILOVER] All configured Gemini API keys failed or on cooldown.")
                self.set_state("DISCONNECTED")
            self.gui_queue.put(("AI_STOPPED", str(e)))
        finally:
            if mic_stream:
                try:
                    mic_stream.stop()
                    mic_stream.close()
                except Exception:
                    pass

            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()
            self.gui_queue.put(("AI_STOPPED", None))

    def _audio_playback_loop(self, stop_evt):
        """ Single authoritative 24kHz PCM audio playback thread """
        out_stream = None
        try:
            out_stream = sd.RawOutputStream(
                samplerate=24000,
                channels=1,
                dtype='int16'
            )
            out_stream.start()

            # Warm-up silence buffer to prevent initial pop/clipping
            out_stream.write(bytes(4800))
            print("[PLAYBACK] Speaker output stream active (24kHz 16-bit PCM).")

            while self.ai_running:
                if stop_evt.is_set():
                    stop_evt.clear()
                    while not self.playback_queue.empty():
                        try: self.playback_queue.get_nowait()
                        except queue.Empty: break
                    continue

                try:
                    item = self.playback_queue.get(timeout=0.01)
                    if item and self.ai_running:
                        if isinstance(item, tuple):
                            resp_id, pcm_chunk = item
                        else:
                            resp_id = self.current_response_id
                            pcm_chunk = item

                        # DROPPING STALE CHUNKS FROM PREVIOUS TURN / INTERRUPTIONS
                        if resp_id != self.current_response_id:
                            print(f"[BARGE-IN] Dropping stale audio chunk from response_id={resp_id} (active={self.current_response_id})")
                            continue

                        gain = self.engine.store.get_setting("assistant_volume", 1.25)
                        if gain != 1.0:
                            audio_np = np.frombuffer(pcm_chunk, dtype=np.int16).astype(np.float32)
                            audio_np = np.clip(audio_np * gain, -32768.0, 32767.0).astype(np.int16)
                            pcm_chunk = audio_np.tobytes()

                        self.set_state("AI_SPEAKING")
                        out_stream.write(pcm_chunk)
                        if self.playback_queue.empty() and self.ai_running:
                            self.set_state("LISTENING")
                except queue.Empty:
                    continue
        except Exception as e:
            print("[AUDIO-PLAYBACK] Error:", e)
        finally:
            if out_stream:
                try:
                    out_stream.stop()
                    out_stream.close()
                except Exception:
                    pass
            with self.session_lock:
                self.playback_thread_started = False
            print("[PLAYBACK] Speaker output worker terminated cleanly.")

    async def _run_live_session(self, api_key):
        import uuid
        session_id = uuid.uuid4().hex[:8]
        with self.session_lock:
            self.active_session_id = session_id

        print(f"[SESSION] [START] Initializing Gemini Live session {session_id}...")

        client = genai.Client(api_key=api_key)

        user_mem_context = self.engine.memory.get_relevant_user_context()
        full_instruction = SYSTEM_INSTRUCTION
        if user_mem_context:
            full_instruction += f"\nStored User Context: {user_mem_context}"

        voice_name = self.engine.store.get_setting("assistant_voice", "Puck")
        print(f"[VOICE] [VOICE_CONFIG] Authoritative single voice selected: '{voice_name}'")

        tools = [
            types.Tool(
                function_declarations=[
                    types.FunctionDeclaration(
                        name="get_ambient_status",
                        description="Checks ambient room lighting, color of objects/clothing in view, faces, and immediate physical hazards.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "query_type": types.Schema(type="STRING", description="What to check: 'light', 'color', 'all'")
                            }
                        )
                    ),
                    types.FunctionDeclaration(
                        name="scan_product_details",
                        description="Scans packaging, QR codes, barcodes, medicine bottle text, and expiration dates in view.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={}
                        )
                    ),
                    types.FunctionDeclaration(
                        name="enroll_person_face",
                        description="Enrolls the face currently visible in the camera under the specified name.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "person_name": types.Schema(type="STRING", description="The name of the person to remember.")
                            },
                            required=["person_name"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="save_reminder_note",
                        description="Saves a personal fact or reminder for the user in long-term memory.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "fact_text": types.Schema(type="STRING", description="The fact or detail to remember.")
                            },
                            required=["fact_text"]
                        )
                    ),
                    types.FunctionDeclaration(
                        name="recall_user_memory",
                        description="Retrieves persistent personal memories, facts, preferences, or relationships saved for the user.",
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                "search_query": types.Schema(type="STRING", description="The search query or keyword to look up.")
                            },
                            required=["search_query"]
                        )
                    )
                ]
            )
        ]

        config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=voice_name
                    )
                )
            ),
            system_instruction=types.Content(parts=[types.Part.from_text(text=full_instruction)]),
            input_audio_transcription=types.AudioTranscriptionConfig(),
            output_audio_transcription=types.AudioTranscriptionConfig(),
            tools=tools
        )

        model_name = "gemini-3.1-flash-live-preview"
        session_established = False

        try:
            async with client.aio.live.connect(model=model_name, config=config) as session:
                session_established = True
                print(f"[SESSION] Gemini Live WebSocket connected (session_id={session_id}).")
                if self.ai_running and self.current_state not in ("SLEEPING", "STOPPED", "CLOSED"):
                    self.set_state("LISTENING")

                if self.engine.store.should_trigger_startup_greeting():
                    greeting = self._generate_time_greeting()
                    print(f"[GREETING] Triggering startup greeting: '{greeting}'")
                    self.gui_queue.put(("TRANSCRIPT_AI", greeting))
                    try:
                        await session.send_client_content(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text=f"Speak this exact startup greeting out loud with a warm, natural, friendly, confident voice: '{greeting}'")]
                                )
                            ],
                            turn_complete=True
                        )
                    except Exception as ge:
                        print(f"[GREETING] Error sending greeting prompt: {ge}")

                mic_task = asyncio.create_task(self._send_mic_loop(session, session_id))
                video_task = asyncio.create_task(self._send_video_loop(session, session_id))
                receive_task = asyncio.create_task(self._receive_loop(session, session_id))

                while self.ai_running and self.active_session_id == session_id:
                    done, pending = await asyncio.wait(
                        [mic_task, video_task, receive_task],
                        timeout=0.2,
                        return_when=asyncio.FIRST_EXCEPTION
                    )
                    for task in done:
                        if task.exception():
                            print(f"[SESSION] Subtask exception in session {session_id}: {task.exception()}")
                            raise task.exception()

                mic_task.cancel()
                video_task.cancel()
                receive_task.cancel()
                await asyncio.gather(mic_task, video_task, receive_task, return_exceptions=True)

        except Exception as e:
            if not self.ai_running or self.current_state in ("SLEEPING", "STOPPED", "CLOSED"):
                return

            print(f"[SESSION] Live session exception (session_id={session_id}, established={session_established}): {e}")

            # If the session was NEVER established, or if the failure is classified as invalid key / quota exhaustion,
            # raise the exception to let _ai_worker_thread trigger multi-key failover!
            reason, _ = self.engine.key_manager.classify_failure(e)
            if not session_established or reason in ("INVALID_KEY", "DAILY_QUOTA_EXHAUSTED"):
                raise e

            # For established sessions experiencing a transient network disconnect, attempt reconnect
            self.set_state("RECONNECTING")
            await asyncio.sleep(1.5)
            if self.ai_running and self.current_state not in ("SLEEPING", "STOPPED", "CLOSED"):
                with self.session_lock:
                    if self.active_session_id == session_id:
                        self.active_session_id = None
                active_key = self.engine.key_manager.get_active_api_key() or api_key
                return await self._run_live_session(active_key)
        finally:
            try:
                await client.aio.aclose()
            except Exception:
                pass

    async def _send_mic_loop(self, session, session_id):
        while self.ai_running and self.active_session_id == session_id:
            try:
                # Check for pending speech prompt (e.g. person-aware startup/wake greeting)
                prompt_to_send = None
                with self.session_lock:
                    if self.pending_speech_prompt:
                        prompt_to_send = self.pending_speech_prompt
                        self.pending_speech_prompt = None

                if prompt_to_send:
                    print("[GREETING 07] playback started")
                    try:
                        await session.send_client_content(
                            turns=[
                                types.Content(
                                    role="user",
                                    parts=[types.Part.from_text(text=prompt_to_send)]
                                )
                            ],
                            turn_complete=True
                        )
                    except Exception as pe:
                        print(f"[GREETING-ERR] Error sending speech prompt: {pe}")

                pcm_data = None
                try:
                    pcm_data = self.mic_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.001)
                    continue

                if pcm_data and self.ai_running and self.active_session_id == session_id:
                    blob = types.Blob(data=pcm_data, mime_type="audio/pcm;rate=16000")
                    await session.send_realtime_input(audio=blob)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.ai_running or self.active_session_id != session_id or self.current_state in ("SLEEPING", "STOPPED", "CLOSED"):
                    break
                err_str = str(e).lower()
                if any(x in err_str for x in ["1011", "closed", "deadline", "broken", "eof", "connection", "reset", "abort"]):
                    print(f"[SESSION] Live mic stream closed ({e}).")
                    raise
                print(f"[MIC] Warning sending audio chunk: {e}")
                await asyncio.sleep(0.05)

    async def _send_video_loop(self, session, session_id):
        while self.ai_running and self.active_session_id == session_id:
            try:
                await asyncio.sleep(1.0)
                jpeg_data = None
                with self.frame_lock:
                    jpeg_data = self.latest_jpeg

                if jpeg_data and self.ai_running and self.active_session_id == session_id:
                    blob = types.Blob(data=jpeg_data, mime_type="image/jpeg")
                    await session.send_realtime_input(video=blob)
            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.ai_running or self.active_session_id != session_id or self.current_state in ("SLEEPING", "STOPPED", "CLOSED"):
                    break
                err_str = str(e).lower()
                if any(x in err_str for x in ["1011", "closed", "deadline", "broken", "eof", "connection", "reset", "abort"]):
                    print(f"[SESSION] Live video stream closed ({e}).")
                    raise
                print(f"[CAMERA] Warning sending video frame: {e}")
                await asyncio.sleep(0.5)

    def _finalize_user_speech_turn(self):
        """ Finalizes the accumulated user speech utterance and dispatches local intent handling. """
        user_text = (self.user_transcript_buffer or "").strip()
        self.user_transcript_buffer = ""
        if not user_text:
            return False

        print(f"[VOICE] user_speech_turn_finalized: '{user_text}'")
        # Store full user message in persistent conversation history
        self.engine.history.add_message(self.active_history_session_id, "user", user_text)

        local_response = self.engine.process_user_speech_query(user_text, session_id=self.active_history_session_id)
        if local_response:
            # Cancel any pending wake/startup greeting so it never interrupts or overlaps user command
            self.wake_greeting_pending = False

            # Clear any preliminary/stale streaming audio from playback queue and advance response_id
            self._clear_playback_queue()

            self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", local_response))
            self.engine.history.add_message(self.active_history_session_id, "assistant", local_response)
            
            # Immediately update Quick Memory card in GUI HUD
            try:
                self._update_info_strip()
            except Exception:
                pass

            if "sleep" in user_text.lower() or "stop listening" in user_text.lower():
                self.enter_sleep_mode()
            else:
                print(f"[SAVE] voice queued: '{local_response}'")
                with self.session_lock:
                    self.pending_speech_prompt = f"Speak this exact response out loud in a warm, natural, friendly, confident voice: '{local_response}'"
                try:
                    self.root.after(1500, lambda: self.set_state("LISTENING") if self.current_state in ("AI_THINKING", "USER_SPEAKING") and self.playback_queue.empty() else None)
                except Exception:
                    pass
            return True
        else:
            self.set_state("AI_THINKING")
            return False

    async def _receive_loop(self, session, session_id):
        print(f"[RECEIVE] Starting receive loop for session_id={session_id}")
        t_speech_start = 0.0
        first_audio_logged = False

        while self.ai_running and self.active_session_id == session_id:
            try:
                async for response in session.receive():
                    if not self.ai_running or self.active_session_id != session_id:
                        print(f"[RECEIVE] Exiting receive loop for inactive session {session_id}")
                        break

                    server_content = response.server_content
                    if server_content is not None:
                        if server_content.input_transcription and server_content.input_transcription.text:
                            text_chunk = server_content.input_transcription.text
                            t_speech_start = time.time()
                            self.last_user_speech_time = t_speech_start
                            first_audio_logged = False
                            print(f"[VOICE] speech_chunk: '{text_chunk}'")

                            self.set_state("USER_SPEAKING")
                            # BARGE-IN: Clear old audio queue and advance response_id immediately
                            self._clear_playback_queue()

                            # Accumulate streaming transcription chunks into full user utterance
                            if not self.user_transcript_buffer:
                                self.user_transcript_buffer = text_chunk
                            else:
                                if text_chunk.startswith(self.user_transcript_buffer):
                                    self.user_transcript_buffer = text_chunk
                                else:
                                    sep = "" if (self.user_transcript_buffer.endswith(" ") or text_chunk.startswith(" ")) else " "
                                    self.user_transcript_buffer += sep + text_chunk

                            self.gui_queue.put(("TRANSCRIPT_USER", self.user_transcript_buffer))

                        if server_content.output_transcription and server_content.output_transcription.text:
                            # Finalize pending user speech turn before processing assistant output
                            intercepted = self._finalize_user_speech_turn()
                            text_chunk = server_content.output_transcription.text
                            if not intercepted:
                                self.gui_queue.put(("TRANSCRIPT_AI", text_chunk))
                                self.engine.history.accumulate_assistant_chunk(text_chunk)

                        if server_content.model_turn:
                            # Finalize pending user speech turn when AI model starts speaking
                            intercepted = self._finalize_user_speech_turn()
                            if not intercepted:
                                curr_resp_id = self.current_response_id
                                for part in server_content.model_turn.parts:
                                    if part.inline_data and part.inline_data.data:
                                        if not first_audio_logged:
                                            t_first_chunk = time.time()
                                            first_audio_logged = True
                                            latency = t_first_chunk - t_speech_start if t_speech_start > 0 else 0.0
                                            print(f"[VOICE] response_first_chunk (time_to_first_audio={latency:.3f}s)")
                                            print("[VOICE] playback_started")
                                        pcm_bytes = part.inline_data.data
                                        # Tag audio chunk with current response_id
                                        self.playback_queue.put((curr_resp_id, pcm_bytes))

                        if server_content.turn_complete:
                            # Finalize any remaining user speech turn
                            self._finalize_user_speech_turn()
                            t_complete = time.time()
                            total_time = t_complete - t_speech_start if t_speech_start > 0 else 0.0
                            print(f"[VOICE] response_complete (total_response_time={total_time:.3f}s)")
                            print("[GREETING 08] playback completed")
                            # Merge accumulated streaming chunks into ONE assistant message and save to SQLite
                            self.engine.history.finalize_assistant_turn(self.active_history_session_id)
                            if self.playback_queue.empty():
                                self.set_state("LISTENING")

                    tool_call = getattr(response, "tool_call", None)
                    if tool_call is not None:
                        # Finalize any pending user speech turn when tool call arrives
                        self._finalize_user_speech_turn()
                        for call in getattr(tool_call, "function_calls", []):
                            fn_name = call.name
                            fn_args = call.args or {}
                            call_id = call.id

                            result_content = {}
                            if fn_name == "get_ambient_status":
                                frame = self.engine.current_frame
                                light = self.engine.color_detector.check_ambient_light(frame)
                                color = self.engine.color_detector.detect_dominant_color(frame)
                                result_content = {"light": light, "color": color}
                            elif fn_name == "scan_product_details":
                                frame = self.engine.current_frame
                                ocr_res = self.engine.ocr_engine.process_ocr(frame)
                                prod_res = self.engine.product_scanner.scan_product_label(frame, ocr_text=ocr_res.get("text", ""))
                                result_content = prod_res
                            elif fn_name == "enroll_person_face":
                                name = fn_args.get("person_name", "Friend")
                                res = self.engine.face_recognizer.enroll_active_face(self.engine.current_frame, name)
                                if res.get("success"):
                                    self.engine.memory.save_memory("relationship", name.lower(), f"{name} is saved in face memory.")
                                    try:
                                        self._update_info_strip()
                                    except Exception:
                                        pass
                                result_content = res
                            elif fn_name == "save_reminder_note":
                                fact = fn_args.get("fact_text", "")
                                print(f"[SAVE] tool_call 'save_reminder_note' with fact: '{fact}'")
                                key, fact_val = self.engine.router.extract_memory_key_and_fact(fact)
                                if not key or key == "contextual":
                                    key_words = [w for w in re.sub(r'[^\w\s]', '', (fact_val or fact).lower()).split() if w not in ["that", "my", "the", "a", "an", "this", "it", "is", "are", "user", "users"]]
                                    key = " ".join(key_words[:2]) if key_words else "personal_fact"
                                success = self.engine.memory.save_memory("personal", key, fact_val or fact)
                                try:
                                    self._update_info_strip()
                                except Exception:
                                    pass
                                result_content = {"status": "saved" if success else "failed", "fact": fact_val or fact}
                            elif fn_name == "recall_user_memory":
                                query = fn_args.get("search_query", "")
                                memory_fact = self.engine.memory.recall_memory(query)
                                result_content = {"recalled_memory": memory_fact or "No memory saved matching query."}

                            try:
                                await session.send_tool_response(
                                    function_responses=[
                                        types.FunctionResponse(
                                            name=fn_name,
                                            response={"result": result_content},
                                            id=call_id
                                        )
                                    ]
                                )
                            except Exception as te:
                                print(f"[TOOL-RESPONSE] Error sending tool response: {te}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self.ai_running or self.active_session_id != session_id or self.current_state in ("SLEEPING", "STOPPED", "CLOSED"):
                    break
                err_str = str(e).lower()
                if any(x in err_str for x in ["1011", "closed", "deadline", "broken", "eof", "connection", "reset", "abort"]):
                    print(f"[SESSION] Live receive stream closed ({e}).")
                    raise
                print(f"[RECEIVE] Exception in session {session_id}: {e}")
                await asyncio.sleep(0.5)

    # --- Interactive Secondary Modal Dialogs ---
    def open_history_dialog(self):
        """ Persistent Conversation History Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE — Persistent Conversation History")
        dialog.geometry("780x620")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        # Title Bar
        title_lbl = tk.Label(dialog, text="📜 Conversation History", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        # Search Bar
        search_frame = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        search_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        search_entry = tk.Entry(search_frame, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 8))

        # Main Split Frame: Left Sessions List, Right Transcript View
        main_split = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        main_split.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        # Left Sessions Listbox Frame
        sessions_frame = tk.Frame(main_split, bg=COLOR_PANEL_SECONDARY, width=280, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        sessions_frame.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        sessions_frame.pack_propagate(False)

        sess_lbl = tk.Label(sessions_frame, text="SAVED SESSIONS", bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 8, "bold"))
        sess_lbl.pack(anchor="w", padx=10, pady=8)

        sess_listbox = tk.Listbox(sessions_frame, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, selectbackground=COLOR_BORDER_ACTIVE, selectforeground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightthickness=0)
        sess_listbox.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0, 8))

        # Right Transcript Display Box
        transcript_frame = tk.Frame(main_split, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        transcript_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        trans_box = scrolledtext.ScrolledText(transcript_frame, wrap=tk.WORD, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0)
        trans_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        active_sessions_data = []

        def load_sessions():
            nonlocal active_sessions_data
            sess_listbox.delete(0, tk.END)
            query = search_entry.get().strip()
            if query:
                sessions = self.engine.history.search_history(query)
            else:
                sessions = self.engine.history.list_all_sessions()

            active_sessions_data = sessions
            if sessions:
                for s in sessions:
                    sess_listbox.insert(tk.END, f"● {s['title']}")
                sess_listbox.select_set(0)
                show_transcript(0)
            else:
                trans_box.configure(state=tk.NORMAL)
                trans_box.delete("1.0", tk.END)
                trans_box.insert(tk.END, "No matching stored conversation sessions found.\n")
                trans_box.configure(state=tk.DISABLED)

        def show_transcript(idx):
            if idx < 0 or idx >= len(active_sessions_data):
                return
            sess = active_sessions_data[idx]
            sid = sess["session_id"]
            messages = self.engine.history.get_session_messages(sid)

            trans_box.configure(state=tk.NORMAL)
            trans_box.delete("1.0", tk.END)
            trans_box.insert(tk.END, f"=== {sess['title']} ===\n\n")

            if messages:
                for m in messages:
                    t_str = time.strftime('%I:%M %p', time.localtime(m['timestamp']))
                    sender_label = "You" if m['sender'].lower() == "user" else "SG CUBE"
                    trans_box.insert(tk.END, f"[{sender_label} - {t_str}]\n{m['text']}\n\n")
            else:
                trans_box.insert(tk.END, "No messages in this session.\n")
            trans_box.configure(state=tk.DISABLED)

        def on_session_select(evt):
            sel = sess_listbox.curselection()
            if sel:
                show_transcript(sel[0])

        sess_listbox.bind("<<ListboxSelect>>", on_session_select)

        btn_search = tk.Button(search_frame, text="Search", bg=COLOR_PANEL_SECONDARY, fg=COLOR_CYAN_PRIMARY, activebackground=COLOR_BORDER_ACTIVE, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=12, pady=4, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=load_sessions)
        btn_search.pack(side=tk.RIGHT)

        # Bottom Actions Bar
        bottom_bar = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        bottom_bar.pack(fill=tk.X, padx=20, pady=(0, 15))

        def delete_selected():
            sel = sess_listbox.curselection()
            if sel and sel[0] < len(active_sessions_data):
                sid = active_sessions_data[sel[0]]["session_id"]
                self.engine.history.delete_session(sid)
                load_sessions()

        def clear_all():
            if messagebox.askyesno("Confirm Clear History", "Are you sure you want to permanently delete all conversation history?\n\n(Personal memory and face memory will NOT be deleted.)", parent=dialog):
                self.engine.history.clear_all_history()
                load_sessions()

        btn_del = tk.Button(bottom_bar, text="Delete Selected", bg=COLOR_PANEL_DEEP, fg=COLOR_ALERT_RED, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_ALERT_RED, relief=tk.FLAT, bd=0, padx=12, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=delete_selected)
        btn_del.pack(side=tk.LEFT, padx=(0, 8))

        btn_clear_all = tk.Button(bottom_bar, text="Clear All History", bg=COLOR_ALERT_RED, fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, padx=14, pady=5, cursor="hand2", command=clear_all)
        btn_clear_all.pack(side=tk.LEFT)

        btn_close = tk.Button(bottom_bar, text="Close", font=("Segoe UI", 9, "bold"), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_close.pack(side=tk.RIGHT)

        load_sessions()

    def open_memory_dialog(self):
        """ Persistent Memory Management Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE — Persistent Memory Database")
        dialog.geometry("560x580")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        title_lbl = tk.Label(dialog, text="✦ Stored Long-Term Memories", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        # Search Bar
        search_frame = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        search_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        search_entry = tk.Entry(search_frame, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        search_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 8))

        mem_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        mem_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        mem_box = scrolledtext.ScrolledText(mem_card, wrap=tk.WORD, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, height=14)
        mem_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def refresh_memories():
            mem_box.configure(state=tk.NORMAL)
            mem_box.delete("1.0", tk.END)
            kw = search_entry.get().strip()
            if kw:
                mems = self.engine.memory.search_memories(kw)
            else:
                mems = self.engine.memory.list_all_memories()

            if mems:
                for idx, m in enumerate(mems, 1):
                    mem_box.insert(tk.END, f"{idx}. [{m['category'].upper()}] {m['fact_value']}\n   (Key: '{m['key_phrase']}')\n\n")
            else:
                mem_box.insert(tk.END, "No matching stored memories found.\nClick 'Add Fact' or say 'Remember that...' to save facts.\n")
            mem_box.configure(state=tk.DISABLED)

        btn_search = tk.Button(search_frame, text="Search", bg=COLOR_PANEL_SECONDARY, fg=COLOR_CYAN_PRIMARY, activebackground=COLOR_BORDER_ACTIVE, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=12, pady=4, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=refresh_memories)
        btn_search.pack(side=tk.RIGHT)

        # Bottom Actions Bar
        actions_bar = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        actions_bar.pack(fill=tk.X, padx=20, pady=(0, 15))

        def add_fact_prompt():
            import tkinter.simpledialog as sd
            fact = sd.askstring("Add Personal Memory", "Enter new fact/detail to remember:", parent=dialog)
            if fact and fact.strip():
                ok = self.engine.memory.save_memory("personal", "fact", fact.strip())
                if ok:
                    messagebox.showinfo("Memory Saved", f"Saved: '{fact.strip()}'", parent=dialog)
                else:
                    messagebox.showwarning("Notice", "Could not save memory.", parent=dialog)
                refresh_memories()

        btn_add = tk.Button(actions_bar, text="+ Add Fact", bg=COLOR_PANEL_DEEP, fg=COLOR_CYAN_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=12, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=add_fact_prompt)
        btn_add.pack(side=tk.LEFT, padx=(0, 8))

        btn_clear = tk.Button(actions_bar, text="Clear All Memories", bg=COLOR_ALERT_RED, fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, padx=12, pady=5, cursor="hand2", command=lambda: [messagebox.askyesno("Confirm Clear Memories", "Are you sure you want to delete all stored personal memories?", parent=dialog) and self.engine.memory.clear_all_memories(), refresh_memories()])
        btn_clear.pack(side=tk.LEFT)

        btn_close = tk.Button(actions_bar, text="Close", font=("Segoe UI", 9, "bold"), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_close.pack(side=tk.RIGHT)

        refresh_memories()

    def open_vision_dialog(self):
        """ Visual Perception Overview Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE — Assistive Vision Dashboard")
        dialog.geometry("540x500")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        title_lbl = tk.Label(dialog, text="◉ Assistive Vision Overview", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        info_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        info_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        info_box = scrolledtext.ScrolledText(info_card, wrap=tk.WORD, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, height=12)
        info_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def refresh_vision():
            info_box.configure(state=tk.NORMAL)
            info_box.delete("1.0", tk.END)
            faces = [f.get("name") or "Unknown" for f in self.engine.last_faces]
            safety = self.engine.last_safety.get("warning_text", "Clear") if self.engine.last_safety.get("hazard_detected") else "No immediate hazards detected."

            info_box.insert(tk.END, f"● Camera Capture Stream: {'Active (~20 FPS)' if self.camera_running else 'Offline'}\n\n")
            info_box.insert(tk.END, f"● Detected Faces in View: {', '.join(faces) if faces else 'None'}\n\n")
            info_box.insert(tk.END, f"● Safety Hazard Status: {safety}\n\n")
            info_box.insert(tk.END, f"● Continuous Perception: {'ON' if self.engine.monitor.is_continuous() else 'OFF'}\n\n")
            info_box.configure(state=tk.DISABLED)

        # Quick Actions Bar
        actions_bar = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        actions_bar.pack(fill=tk.X, padx=20, pady=(0, 15))

        def check_color():
            frame = self.engine.current_frame
            col = self.engine.color_detector.detect_dominant_color(frame)
            messagebox.showinfo("Dominant Color", f"Dominant color in view: {col}", parent=dialog)

        def check_light():
            frame = self.engine.current_frame
            light = self.engine.color_detector.check_ambient_light(frame)
            messagebox.showinfo("Ambient Lighting", f"Current lighting condition: {light}", parent=dialog)

        def describe_scene():
            frame = self.engine.current_frame
            sc = self.engine.scene.analyze_scene(frame)
            desc = sc.get("summary", "No description available.")
            messagebox.showinfo("Scene Description", desc, parent=dialog)

        btn_color = tk.Button(actions_bar, text="Detect Color", bg=COLOR_PANEL_DEEP, fg=COLOR_TEAL_MINT, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=check_color)
        btn_color.pack(side=tk.LEFT, padx=(0, 6))

        btn_light = tk.Button(actions_bar, text="Check Light", bg=COLOR_PANEL_DEEP, fg=COLOR_WARNING_GOLD, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=check_light)
        btn_light.pack(side=tk.LEFT, padx=(0, 6))

        btn_scene = tk.Button(actions_bar, text="Describe Scene", bg=COLOR_PANEL_DEEP, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=describe_scene)
        btn_scene.pack(side=tk.LEFT)

        btn_close = tk.Button(actions_bar, text="Close", font=("Segoe UI", 9, "bold"), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_close.pack(side=tk.RIGHT)

        refresh_vision()

    def open_people_dialog(self):
        """ People & Face Memory Registry Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE — Face Memory Registry")
        dialog.geometry("540x500")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        title_lbl = tk.Label(dialog, text="👥 Enrolled People & Face Registry", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        list_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        list_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        list_box = scrolledtext.ScrolledText(list_card, wrap=tk.WORD, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, height=12)
        list_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        def refresh_people():
            list_box.configure(state=tk.NORMAL)
            list_box.delete("1.0", tk.END)
            people = self.engine.face_memory.list_people()
            if people:
                for idx, p in enumerate(people, 1):
                    list_box.insert(tk.END, f"{idx}. {p}\n")
            else:
                list_box.insert(tk.END, "No saved face profiles enrolled yet.\nClick 'Enroll Face' or say 'Enroll face as [Name]'.\n")
            list_box.configure(state=tk.DISABLED)

        actions_frame = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        actions_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        def enroll_face_prompt():
            import tkinter.simpledialog as sd
            name = sd.askstring("Enroll Face", "Enter person's name to remember face:", parent=dialog)
            if name and name.strip():
                res = self.engine.face_recognizer.enroll_active_face(self.engine.current_frame, name.strip())
                if res.get("success"):
                    messagebox.showinfo("Success", f"Enrolled face profile for '{name.strip()}'.", parent=dialog)
                else:
                    messagebox.showwarning("Notice", res.get("message", "No clear face detected in frame."), parent=dialog)
                refresh_people()

        def test_recognition():
            print("[FACE-TEST] frame received")
            frame = self.engine.current_frame
            if frame is None or getattr(frame, 'size', 0) == 0:
                print("[FACE-TEST] no frame available")
                print("[FACE-TEST] decision = NO FRAME")
                print("[FACE-TEST] completed")
                messagebox.showwarning("Notice", "Camera feed is not ready or active. Please ensure camera is connected.", parent=dialog)
                self.set_state("LISTENING")
                return

            boxes = self.engine.face_recognizer.detect_faces(frame)
            num_faces = len(boxes)
            print(f"[FACE-TEST] faces detected = {num_faces}")

            if num_faces == 0:
                print("[FACE-TEST] crop created = None")
                print("[FACE-TEST] decision = NO FACE")
                print("[FACE-TEST] completed")
                messagebox.showinfo("Recognition Result", "No face detected in current frame.\nPlease look directly at the camera.", parent=dialog)
                self.set_state("LISTENING")
                return

            faces = self.engine.face_recognizer.process_frame(frame)
            num_stored = len(self.engine.face_memory.profiles)
            print(f"[FACE-TEST] stored profiles = {num_stored}")

            summary_lines = []
            voice_names = []
            threshold = self.engine.face_recognizer.threshold

            for idx, face_info in enumerate(faces, 1):
                crop = face_info.get("crop")
                name = face_info.get("name")
                conf = face_info.get("confidence", 0.0)

                print(f"[FACE-TEST] crop created: shape={crop.shape if crop is not None else 'None'}")
                if crop is not None and crop.size > 0:
                    emb = self.engine.face_memory.compute_face_embedding(crop)
                    print(f"[FACE-TEST] embedding created")
                    print(f"[FACE-TEST] embedding dimension = {len(emb)}")
                else:
                    print(f"[FACE-TEST] embedding created: 0-D")

                print(f"[FACE-TEST] best profile = {name or 'Unknown'}")
                print(f"[FACE-TEST] similarity = {conf:.4f}")
                print(f"[FACE-TEST] threshold = {threshold:.2f}")

                if name:
                    decision = "MATCH"
                    pct = conf * 100.0
                    summary_lines.append(f"Face {idx}: MATCH — {name} ({pct:.1f}% confidence)")
                    voice_names.append(name)
                else:
                    decision = "UNKNOWN"
                    pct = conf * 100.0
                    summary_lines.append(f"Face {idx}: UNKNOWN — (best similarity: {pct:.1f}%, threshold: {threshold*100:.0f}%)")

                print(f"[FACE-TEST] decision = {decision}")

            print("[FACE-TEST] completed")

            result_text = "\n".join(summary_lines)
            if voice_names:
                voice_msg = f"I recognize {', '.join(voice_names)}."
            else:
                voice_msg = "I detect a face, but it is not enrolled in face memory."

            self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", voice_msg))
            self.set_state("LISTENING")

            messagebox.showinfo(
                "Face Recognition Result",
                f"Face Recognition Pipeline Results ({len(faces)} face(s) found):\n\n{result_text}",
                parent=dialog
            )

        def clear_faces():
            if messagebox.askyesno("Confirm Clear Face Profiles", "Are you sure you want to delete all stored face profiles?", parent=dialog):
                self.engine.face_memory.clear_all_profiles()
                refresh_people()

        btn_enroll = tk.Button(actions_frame, text="Enroll Face", bg=COLOR_PANEL_DEEP, fg=COLOR_CYAN_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=enroll_face_prompt)
        btn_enroll.pack(side=tk.LEFT, padx=(0, 6))

        btn_test = tk.Button(actions_frame, text="Test Recognition", bg=COLOR_PANEL_DEEP, fg=COLOR_ORANGE, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=test_recognition)
        btn_test.pack(side=tk.LEFT, padx=(0, 6))

        btn_clear = tk.Button(actions_frame, text="Clear All", bg=COLOR_ALERT_RED, fg="#ffffff", font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, padx=10, pady=5, cursor="hand2", command=clear_faces)
        btn_clear.pack(side=tk.LEFT)

        btn_close = tk.Button(actions_frame, text="Close", font=("Segoe UI", 9, "bold"), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_close.pack(side=tk.RIGHT)

        refresh_people()

    def open_meta_glass_dialog(self):
        """ Meta Ray-Ban Glass Bridge & Wearable Status Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("👓 Meta Glass / VisionClaw Bridge")
        dialog.geometry("540x500")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        title_lbl = tk.Label(dialog, text="👓 Meta Glass Wearable Integration", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        status_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        status_card.pack(fill=tk.X, padx=20, pady=(0, 10), ipady=8)

        lbl_state = tk.Label(status_card, text="Bridge Status: DISCONNECTED", bg=COLOR_PANEL_SECONDARY, fg=COLOR_ORANGE, font=("Segoe UI", 10, "bold"))
        lbl_state.pack(anchor="w", padx=12, pady=(4, 2))

        lbl_detail = tk.Label(status_card, text="", bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 8))
        lbl_detail.pack(anchor="w", padx=12, pady=(0, 4))

        info_grid = tk.Frame(status_card, bg=COLOR_PANEL_DEEP, padx=10, pady=8, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        info_grid.pack(fill=tk.X, padx=12, pady=4)

        lbl_source = tk.Label(info_grid, text="• Active Camera Source: LAPTOP CAMERA", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 9))
        lbl_source.pack(anchor="w", pady=1)

        lbl_stream = tk.Label(info_grid, text="• Optical Video Stream: INACTIVE", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9))
        lbl_stream.pack(anchor="w", pady=1)

        lbl_audio = tk.Label(info_grid, text="• Conversational Audio: SG CUBE Voice ('Puck')", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9))
        lbl_audio.pack(anchor="w", pady=1)

        def refresh_dialog_ui():
            if not hasattr(self.engine, 'meta_glass') or not self.engine.meta_glass:
                return
            mg = self.engine.meta_glass
            st = mg.current_state
            detail = mg.state_detail
            col_map = {
                "CONNECTED": COLOR_STATUS_GREEN,
                "STREAMING": COLOR_STATUS_GREEN,
                "CONNECTING": COLOR_CYAN_PRIMARY,
                "DISCONNECTED": COLOR_ORANGE,
                "ERROR": COLOR_ALERT_RED,
                "NOT AVAILABLE": COLOR_TEXT_MUTED
            }
            lbl_state.config(text=f"Bridge Status: {st}", fg=col_map.get(st, COLOR_TEXT_PRIMARY))
            lbl_detail.config(text=f"Detail: {detail}")
            lbl_source.config(text=f"• Active Camera Source: {mg.active_source}")
            is_stream = mg.is_streaming()
            lbl_stream.config(
                text=f"• Optical Video Stream: {'ACTIVE (~20 FPS)' if is_stream else 'INACTIVE'}",
                fg=COLOR_STATUS_GREEN if is_stream else COLOR_TEXT_SECONDARY
            )

        # Source Selection Buttons
        source_frame = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        source_frame.pack(fill=tk.X, padx=20, pady=(0, 10))

        def switch_to_glass():
            ok, msg = self.engine.meta_glass.set_camera_source("META_GLASS")
            if ok:
                messagebox.showinfo("Camera Source", msg, parent=dialog)
            else:
                messagebox.showwarning("Notice", msg, parent=dialog)
            refresh_dialog_ui()

        def switch_to_laptop():
            ok, msg = self.engine.meta_glass.set_camera_source("LAPTOP")
            messagebox.showinfo("Camera Source", msg, parent=dialog)
            refresh_dialog_ui()

        btn_use_glass = tk.Button(source_frame, text="Use Glass Camera", bg=COLOR_PANEL_DEEP, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=switch_to_glass)
        btn_use_glass.pack(side=tk.LEFT, padx=(0, 6))

        btn_use_laptop = tk.Button(source_frame, text="Use Laptop Camera", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=10, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=switch_to_laptop)
        btn_use_laptop.pack(side=tk.LEFT)

        # Connection & Snapshot Actions Bar
        actions_bar = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        actions_bar.pack(fill=tk.X, padx=20, pady=(0, 15))

        def do_connect():
            def on_done(ok, msg):
                def _ui():
                    if ok:
                        messagebox.showinfo("Meta Glass Bridge", msg, parent=dialog)
                    else:
                        messagebox.showwarning("Meta Glass Bridge", msg, parent=dialog)
                    refresh_dialog_ui()
                dialog.after(0, _ui)
            self.engine.meta_glass.connect_async(on_complete=on_done)
            refresh_dialog_ui()

        def do_disconnect():
            self.engine.meta_glass.disconnect()
            messagebox.showinfo("Meta Glass Bridge", "Disconnected Meta Glass bridge.", parent=dialog)
            refresh_dialog_ui()

        def do_snapshot():
            ok, path, msg = self.engine.meta_glass.capture_snapshot()
            if ok:
                messagebox.showinfo("Snapshot Saved", f"{msg}\nPath: {path}", parent=dialog)
            else:
                messagebox.showwarning("Snapshot", msg, parent=dialog)

        btn_conn = tk.Button(actions_bar, text="Connect", bg=COLOR_PANEL_DEEP, fg=COLOR_STATUS_GREEN, font=("Segoe UI", 9, "bold"), relief=tk.FLAT, bd=0, padx=12, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=do_connect)
        btn_conn.pack(side=tk.LEFT, padx=(0, 6))

        btn_disconn = tk.Button(actions_bar, text="Disconnect", bg=COLOR_PANEL_DEEP, fg=COLOR_ORANGE, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=12, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=do_disconnect)
        btn_disconn.pack(side=tk.LEFT, padx=(0, 6))

        btn_snap = tk.Button(actions_bar, text="Snapshot", bg=COLOR_PANEL_DEEP, fg=COLOR_TEAL_MINT, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, padx=12, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=do_snapshot)
        btn_snap.pack(side=tk.LEFT)

        btn_close = tk.Button(actions_bar, text="Close", font=("Segoe UI", 9, "bold"), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_close.pack(side=tk.RIGHT)

        refresh_dialog_ui()

    def open_object_finder_dialog(self):
        """ Interactive Object Finder Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE — Object Finder")
        dialog.geometry("520x460")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        title_lbl = tk.Label(dialog, text="🔍 Assistive Object Finder", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        chips_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, padx=12, pady=8)
        chips_card.pack(fill=tk.X, padx=20, pady=(0, 10))

        tk.Label(chips_card, text="Quick Search Presets:", bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        chips_frame = tk.Frame(chips_card, bg=COLOR_PANEL_SECONDARY)
        chips_frame.pack(fill=tk.X)

        search_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, padx=12, pady=8)
        search_card.pack(fill=tk.X, padx=20, pady=(0, 10))

        tk.Label(search_card, text="Target Object:", bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9, "bold")).pack(anchor="w", pady=(0, 4))
        search_row = tk.Frame(search_card, bg=COLOR_PANEL_SECONDARY)
        search_row.pack(fill=tk.X)

        ent_target = tk.Entry(search_row, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        ent_target.insert(0, "phone")
        ent_target.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=4, padx=(0, 8))

        res_card = tk.Frame(dialog, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        res_card.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 10))

        res_box = scrolledtext.ScrolledText(res_card, wrap=tk.WORD, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10), relief=tk.FLAT, bd=0, height=6)
        res_box.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        res_box.insert(tk.END, "Enter or select an object name to find its spatial location in the camera view.\n")
        res_box.configure(state=tk.DISABLED)

        def do_find_object(target_str):
            target = target_str.strip()
            if not target:
                return
            res_box.configure(state=tk.NORMAL)
            res_box.delete("1.0", tk.END)
            res_box.insert(tk.END, f"Searching camera view for '{target}'...\n")
            res_box.configure(state=tk.DISABLED)

            frame = self.engine.current_frame
            obj_res = self.engine.object_detector.search_object(frame, target)
            res_text = obj_res.get("text", f"No object matching '{target}' was detected in view.")

            res_box.configure(state=tk.NORMAL)
            res_box.delete("1.0", tk.END)
            res_box.insert(tk.END, f"● Search Target: {target}\n\n{res_text}\n")
            res_box.configure(state=tk.DISABLED)

            self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", res_text))
            self.show_context_alert(res_text, color=COLOR_CYAN_PRIMARY)
            self.engine.response_manager.add_response(res_text, priority=1)

        btn_search = tk.Button(search_row, text="Find Object", font=("Segoe UI", 9, "bold"), bg=COLOR_CYAN_PRIMARY, fg=COLOR_BG_PRIMARY, activebackground=COLOR_TEAL_MINT, activeforeground=COLOR_BG_PRIMARY, relief=tk.FLAT, bd=0, padx=14, pady=4, cursor="hand2", command=lambda: do_find_object(ent_target.get()))
        btn_search.pack(side=tk.RIGHT)

        for chip_name in ("Phone", "Bottle", "Cup", "Keys", "Laptop", "Person"):
            def make_chip_cmd(c=chip_name):
                return lambda: [ent_target.delete(0, tk.END), ent_target.insert(0, c.lower()), do_find_object(c.lower())]
            c_btn = tk.Button(chips_frame, text=chip_name, bg=COLOR_PANEL_DEEP, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 8), relief=tk.FLAT, bd=0, padx=8, pady=3, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=make_chip_cmd())
            c_btn.pack(side=tk.LEFT, padx=3)

        btn_close = tk.Button(dialog, text="Close", font=("Segoe UI", 9, "bold"), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=5, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_close.pack(pady=(0, 12))

    def open_settings_dialog(self):
        """ Settings & Preferences Modal Dialog """
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE Settings & Preferences")
        dialog.geometry("520x640")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog))

        title_lbl = tk.Label(dialog, text="⚙ Settings & Preferences", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 13, "bold"))
        title_lbl.pack(anchor="w", padx=20, pady=(15, 10))

        container = tk.Frame(dialog, bg=COLOR_BG_PRIMARY, padx=20)
        container.pack(fill=tk.BOTH, expand=True)

        var_greetings = tk.BooleanVar(value=self.engine.store.get_setting("greeting_enabled", True))
        var_safety = tk.BooleanVar(value=self.engine.store.get_setting("safety_alerts_enabled", True))
        var_continuous = tk.BooleanVar(value=self.engine.store.get_setting("environment_monitor_enabled", False))
        var_dev = tk.BooleanVar(value=self.dev_mode)

        chk_greet = tk.Checkbutton(container, text="Enable Person Recognition Greetings", variable=var_greetings, bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, selectcolor=COLOR_PANEL_DEEP, activebackground=COLOR_BG_PRIMARY, activeforeground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9))
        chk_greet.pack(anchor="w", pady=4)

        chk_safe = tk.Checkbutton(container, text="Enable Safety Obstacle Warnings", variable=var_safety, bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, selectcolor=COLOR_PANEL_DEEP, activebackground=COLOR_BG_PRIMARY, activeforeground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9))
        chk_safe.pack(anchor="w", pady=4)

        chk_cont = tk.Checkbutton(container, text="Enable Continuous Environment Monitoring", variable=var_continuous, bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, selectcolor=COLOR_PANEL_DEEP, activebackground=COLOR_BG_PRIMARY, activeforeground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9))
        chk_cont.pack(anchor="w", pady=4)

        chk_dev = tk.Checkbutton(container, text="Developer Mode (Show Bounding Boxes & Tech Logs)", variable=var_dev, bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, selectcolor=COLOR_PANEL_DEEP, activebackground=COLOR_BG_PRIMARY, activeforeground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9))
        chk_dev.pack(anchor="w", pady=4)

        # 🔑 Multi-Gemini API Keys & Failover Configuration Card
        api_card = tk.Frame(container, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        api_card.pack(fill=tk.X, pady=(10, 10), ipady=6)

        api_title = tk.Label(api_card, text="🔑 Gemini API Keys (Multi-Key Failover)", bg=COLOR_PANEL_SECONDARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10, "bold"))
        api_title.pack(anchor="w", padx=12, pady=(6, 2))

        status_str = f"Active: {self.engine.key_manager.get_active_key_label()}"
        lbl_api_status = tk.Label(api_card, text=status_str, bg=COLOR_PANEL_SECONDARY, fg=COLOR_STATUS_GREEN if self.engine.key_manager.get_active_key() else COLOR_ALERT_RED, font=("Segoe UI", 9, "bold"))
        lbl_api_status.pack(anchor="w", padx=12, pady=(0, 6))

        # Render rows for Key 1, Key 2, Key 3
        key_entries = {}
        for kn in (1, 2, 3):
            k_frame = tk.Frame(api_card, bg=COLOR_PANEL_SECONDARY)
            k_frame.pack(fill=tk.X, padx=12, pady=3)

            label_txt = f"Key {kn} (Primary):" if kn == 1 else (f"Key {kn} (Secondary):" if kn == 2 else f"Key {kn} (Tertiary):")
            tk.Label(k_frame, text=label_txt, bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 8, "bold"), width=15, anchor="w").pack(side=tk.LEFT)

            ent = tk.Entry(k_frame, show="•", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
            ent.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=3, padx=(4, 4))
            
            stored_val = self.engine.key_manager.keys.get(kn, "")
            if stored_val:
                ent.insert(0, self.engine.key_manager.get_masked_key(kn))
            key_entries[kn] = ent

            showing_plain = [False]

            def make_paste_cmd(e=ent, knum=kn, sp=showing_plain):
                def _paste():
                    try:
                        clip = dialog.clipboard_get().strip()
                        if clip:
                            e.delete(0, tk.END)
                            e.insert(0, clip)
                            e.config(show="" if sp[0] else "•")
                    except Exception:
                        pass
                return _paste

            def make_toggle_show_cmd(e=ent, knum=kn, btn_ref=[None], sp=showing_plain):
                def _toggle():
                    sp[0] = not sp[0]
                    if sp[0]:
                        cur_text = e.get().strip()
                        if cur_text.startswith("••••") and not cur_text.startswith("AIza"):
                            actual_key = self.engine.key_manager.keys.get(knum, "")
                            if actual_key:
                                e.delete(0, tk.END)
                                e.insert(0, actual_key)
                        e.config(show="")
                        if btn_ref[0]:
                            btn_ref[0].config(text="Hide")
                    else:
                        cur_text = e.get().strip()
                        if cur_text == self.engine.key_manager.keys.get(knum, ""):
                            e.delete(0, tk.END)
                            e.insert(0, self.engine.key_manager.get_masked_key(knum))
                        e.config(show="•")
                        if btn_ref[0]:
                            btn_ref[0].config(text="Show")
                return _toggle

            def make_test_cmd(knum=kn, e=ent):
                def _test():
                    val = e.get().strip()
                    if val.startswith("••••"):
                        val = self.engine.key_manager.keys.get(knum, "")
                    if not val:
                        messagebox.showwarning(f"Test Key {knum}", f"Key {knum} is not entered.", parent=dialog)
                        return
                    ok, msg = self.engine.key_manager.test_connection(val)
                    if ok:
                        messagebox.showinfo(f"Test Key {knum}", f"Key {knum} is valid!\n\n{msg}", parent=dialog)
                    else:
                        messagebox.showwarning(f"Test Key {knum}", f"Key {knum} verification notice:\n\n{msg}", parent=dialog)
                return _test

            def make_save_cmd(knum=kn, e=ent, sp=showing_plain, btn_sh_ref=[None]):
                def _save():
                    val = e.get().strip()
                    if not val:
                        messagebox.showwarning(f"Save Key {knum}", f"Please enter a valid API key string for Key {knum}.", parent=dialog)
                        return
                    # If unchanged masked representation
                    if val.startswith("••••") and val == self.engine.key_manager.get_masked_key(knum):
                        messagebox.showinfo("Saved", f"Key {knum} is already saved and active.", parent=dialog)
                        return

                    # Persist key to user-data storage
                    self.engine.key_manager.set_key(knum, val)
                    lbl_api_status.config(text=f"Active: {self.engine.key_manager.get_active_key_label()}", fg=COLOR_STATUS_GREEN)

                    # Reset entry to masked format
                    e.delete(0, tk.END)
                    e.insert(0, self.engine.key_manager.get_masked_key(knum))
                    e.config(show="•")
                    sp[0] = False
                    if btn_sh_ref[0]:
                        btn_sh_ref[0].config(text="Show")

                    messagebox.showinfo("Success", f"Key {knum} securely saved to user data!\nActive: {self.engine.key_manager.get_active_key_label()}", parent=dialog)

                    # Hot-switch live AI session if running
                    if self.ai_running:
                        self.stop_ai()
                        self.start_ai(val)
                return _save

            def make_clear_cmd(knum=kn, e=ent, sp=showing_plain, btn_sh_ref=[None]):
                def _clear():
                    if messagebox.askyesno(f"Clear Key {knum}", f"Are you sure you want to clear Key {knum}?", parent=dialog):
                        self.engine.key_manager.clear_key(knum)
                        e.delete(0, tk.END)
                        e.config(show="•")
                        sp[0] = False
                        if btn_sh_ref[0]:
                            btn_sh_ref[0].config(text="Show")
                        lbl_api_status.config(text=f"Active: {self.engine.key_manager.get_active_key_label()}", fg=COLOR_STATUS_GREEN if self.engine.key_manager.get_active_key() else COLOR_ALERT_RED)
                return _clear

            # Buttons: Paste, Show/Hide, Save, Test, Clear
            btn_paste = tk.Button(k_frame, text="Paste", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_BORDER_ACTIVE, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=5, pady=2, font=("Segoe UI", 8), highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=make_paste_cmd(ent, kn, showing_plain))
            btn_paste.pack(side=tk.LEFT, padx=1)

            btn_sh_ref = [None]
            btn_toggle = tk.Button(k_frame, text="Show", bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_BORDER_ACTIVE, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=5, pady=2, font=("Segoe UI", 8), highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2")
            btn_toggle.config(command=make_toggle_show_cmd(ent, kn, btn_sh_ref, showing_plain))
            btn_sh_ref[0] = btn_toggle
            btn_toggle.pack(side=tk.LEFT, padx=1)

            btn_save = tk.Button(k_frame, text="Save", bg=COLOR_CYAN_PRIMARY, fg=COLOR_BG_PRIMARY, activebackground=COLOR_TEAL_MINT, activeforeground=COLOR_BG_PRIMARY, relief=tk.FLAT, bd=0, padx=6, pady=2, font=("Segoe UI", 8, "bold"), cursor="hand2", command=make_save_cmd(kn, ent, showing_plain, btn_sh_ref))
            btn_save.pack(side=tk.LEFT, padx=1)

            btn_test = tk.Button(k_frame, text="Test", bg=COLOR_PANEL_DEEP, fg=COLOR_CYAN_PRIMARY, activebackground=COLOR_BORDER_ACTIVE, activeforeground=COLOR_TEXT_PRIMARY, relief=tk.FLAT, bd=0, padx=5, pady=2, font=("Segoe UI", 8), highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=make_test_cmd(kn, ent))
            btn_test.pack(side=tk.LEFT, padx=1)

            btn_clear = tk.Button(k_frame, text="Clear", bg=COLOR_PANEL_DEEP, fg=COLOR_ALERT_RED, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_ALERT_RED, relief=tk.FLAT, bd=0, padx=5, pady=2, font=("Segoe UI", 8), highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=make_clear_cmd(kn, ent, showing_plain, btn_sh_ref))
            btn_clear.pack(side=tk.LEFT, padx=1)

        # 👤 User Profile Card
        prof_card = tk.Frame(container, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        prof_card.pack(fill=tk.X, pady=(10, 10), ipady=6)

        prof_title = tk.Label(prof_card, text="👤 User Profile Settings", bg=COLOR_PANEL_SECONDARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10, "bold"))
        prof_title.pack(anchor="w", padx=12, pady=(6, 2))

        prof_grid = tk.Frame(prof_card, bg=COLOR_PANEL_SECONDARY)
        prof_grid.pack(fill=tk.X, padx=12, pady=(0, 6))

        tk.Label(prof_grid, text="Full Name:", bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        entry_uname = tk.Entry(prof_grid, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        entry_uname.insert(0, self.engine.store.get_setting("user_name", ""))
        entry_uname.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)

        tk.Label(prof_grid, text="Display Name:", bg=COLOR_PANEL_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w")
        entry_dname = tk.Entry(prof_grid, bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        entry_dname.insert(0, self.engine.store.get_setting("user_display_name", ""))
        entry_dname.grid(row=1, column=1, sticky="ew", padx=(6, 0), pady=2)

        prof_grid.columnconfigure(1, weight=1)

        sc_card = tk.Frame(container, bg=COLOR_PANEL_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        sc_card.pack(fill=tk.X, pady=(10, 10), ipady=6)

        sc_title = tk.Label(sc_card, text="Keyboard Shortcuts:", bg=COLOR_PANEL_SECONDARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9, "bold"))
        sc_title.pack(anchor="w", padx=10, pady=(4, 2))

        sc_info = tk.Label(
            sc_card,
            text="Space : Interrupt / Start Listening\nEsc : Stop Current Speech\nCtrl+Shift+S : Settings | Ctrl+M : Memory\nCtrl+V : Vision | Ctrl+P : People | Ctrl+Q : Exit",
            bg=COLOR_PANEL_SECONDARY,
            fg=COLOR_TEXT_SECONDARY,
            font=("Segoe UI", 8),
            justify=tk.LEFT
        )
        sc_info.pack(anchor="w", padx=10)

        def save_and_close():
            self.engine.store.set_setting("greeting_enabled", var_greetings.get())
            self.engine.store.set_setting("safety_alerts_enabled", var_safety.get())
            self.engine.store.set_setting("environment_monitor_enabled", var_continuous.get())
            self.engine.store.set_setting("developer_mode", var_dev.get())

            self.engine.store.set_setting("user_name", entry_uname.get().strip())
            self.engine.store.set_setting("user_display_name", entry_dname.get().strip())

            self.engine.face_recognizer.set_greetings_enabled(var_greetings.get())
            self.engine.monitor.set_mode("continuous" if var_continuous.get() else "on_demand")
            self.dev_mode = var_dev.get()

            animate_dialog_close(dialog)

        btn_frame = tk.Frame(container, bg=COLOR_BG_PRIMARY)
        btn_frame.pack(fill=tk.X, pady=(12, 10))

        btn_save = tk.Button(btn_frame, text="Save Settings", font=("Segoe UI", 10, "bold"), bg=COLOR_CYAN_PRIMARY, fg=COLOR_BG_PRIMARY, activebackground=COLOR_TEAL_MINT, activeforeground=COLOR_BG_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=6, cursor="hand2", command=save_and_close)
        btn_save.pack(side=tk.LEFT)

        btn_cancel = tk.Button(btn_frame, text="Close", font=("Segoe UI", 10), bg=COLOR_PANEL_DEEP, fg=COLOR_TEXT_PRIMARY, activebackground=COLOR_PANEL_SECONDARY, activeforeground=COLOR_CYAN_PRIMARY, relief=tk.FLAT, bd=0, padx=16, pady=6, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1, cursor="hand2", command=lambda: animate_dialog_close(dialog))
        btn_cancel.pack(side=tk.RIGHT)

    def _check_first_run_onboarding(self):
        """ Checks if this is a fresh installation / first-run and pops up onboarding wizard """
        first_run_done = self.engine.store.get_setting("first_run_completed", False)
        user_name = self.engine.store.get_setting("user_name", "")
        if not first_run_done or not user_name:
            self._show_first_run_onboarding()

    def _show_first_run_onboarding(self):
        """ Production-Grade First-Run Installation & Onboarding Wizard Dialog """
        print("[ONBOARDING] Fresh installation detected. Launching First-Run Setup & Permissions Wizard...")
        dialog = tk.Toplevel(self.root)
        dialog.title("SG CUBE — First-Run Installation & Profile Setup")
        dialog.geometry("640x620")
        dialog.configure(bg=COLOR_BG_PRIMARY)
        dialog.transient(self.root)
        dialog.grab_set()
        animate_dialog_open(dialog, target_alpha=0.98, duration_ms=220)

        dialog.bind("<Escape>", lambda e: animate_dialog_close(dialog, duration_ms=180))

        header = tk.Frame(dialog, bg=COLOR_BG_PRIMARY, height=56, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        header.pack(fill=tk.X)
        header.pack_propagate(False)

        title = tk.Label(header, text="✨ Welcome to SG CUBE Setup", bg=COLOR_BG_PRIMARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 12, "bold"))
        title.pack(side=tk.LEFT, padx=16)

        container = tk.Frame(dialog, bg=COLOR_BG_PRIMARY)
        container.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)

        # 1. System Permissions Verification Card
        perm_card = tk.Frame(container, bg=COLOR_BG_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        perm_card.pack(fill=tk.X, pady=(0, 10), ipady=4)

        perm_title = tk.Label(perm_card, text="🔒 System Permissions & Hardware Verification", bg=COLOR_BG_SECONDARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10, "bold"))
        perm_title.pack(anchor="w", padx=12, pady=(4, 2))

        # Real Hardware / Permission Verification
        cam_ok = False
        try:
            test_cap = cv2.VideoCapture(0, cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            if test_cap.isOpened():
                cam_ok = True
                test_cap.release()
        except Exception:
            cam_ok = False

        audio_ok = False
        try:
            devs = sd.query_devices()
            audio_ok = len(devs) > 0
        except Exception:
            audio_ok = False

        cam_text = "● Camera Access: VERIFIED (Visual scene analysis, OCR, Face & Currency)" if cam_ok else "● Camera Access: STANDBY (Camera device not currently detected)"
        cam_col = COLOR_STATUS_GREEN if cam_ok else COLOR_WARNING_GOLD
        tk.Label(perm_card, text=cam_text, bg=COLOR_BG_SECONDARY, fg=cam_col, font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=1)

        audio_text = "● Audio & Microphone: VERIFIED (Voice conversation & 24kHz audio output)" if audio_ok else "● Audio & Microphone: STANDBY (Audio output device ready)"
        audio_col = COLOR_STATUS_GREEN if audio_ok else COLOR_WARNING_GOLD
        tk.Label(perm_card, text=audio_text, bg=COLOR_BG_SECONDARY, fg=audio_col, font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=1)

        tk.Label(perm_card, text="● Meta Glass Bridge: OPTIONAL (BLE hardware link on-demand)", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 8)).pack(anchor="w", padx=12, pady=1)

        # 2. Profile Setup Card
        prof_card = tk.Frame(container, bg=COLOR_BG_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        prof_card.pack(fill=tk.X, pady=(0, 10), ipady=6)

        prof_title = tk.Label(prof_card, text="👤 Personal Profile Setup", bg=COLOR_BG_SECONDARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10, "bold"))
        prof_title.pack(anchor="w", padx=12, pady=(4, 2))

        prof_prompt = tk.Label(prof_card, text="What should I call you?", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_PRIMARY, font=("Segoe UI", 9, "bold"))
        prof_prompt.pack(anchor="w", padx=12, pady=(0, 4))

        prof_grid = tk.Frame(prof_card, bg=COLOR_BG_SECONDARY)
        prof_grid.pack(fill=tk.X, padx=12, pady=(0, 4))

        tk.Label(prof_grid, text="Your Name:", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=0, column=0, sticky="w")
        entry_uname = tk.Entry(prof_grid, bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        entry_uname.grid(row=0, column=1, sticky="ew", padx=(8, 0), pady=3)

        tk.Label(prof_grid, text="Display / Greeting:", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9)).grid(row=1, column=0, sticky="w")
        entry_dname = tk.Entry(prof_grid, bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        entry_dname.grid(row=1, column=1, sticky="ew", padx=(8, 0), pady=3)
        prof_grid.columnconfigure(1, weight=1)

        # 3. Optional Gemini API Key Setup Card
        api_card = tk.Frame(container, bg=COLOR_BG_SECONDARY, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        api_card.pack(fill=tk.X, pady=(0, 10), ipady=6)

        api_title = tk.Label(api_card, text="🔑 Gemini AI Key Setup (Optional during install)", bg=COLOR_BG_SECONDARY, fg=COLOR_CYAN_PRIMARY, font=("Segoe UI", 10, "bold"))
        api_title.pack(anchor="w", padx=12, pady=(4, 2))

        api_desc = tk.Label(api_card, text="Enter your Google Gemini API Key for real-time vision conversation, or configure later in Settings.", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 8))
        api_desc.pack(anchor="w", padx=12, pady=(0, 4))

        api_input_frame = tk.Frame(api_card, bg=COLOR_BG_SECONDARY)
        api_input_frame.pack(fill=tk.X, padx=12, pady=(0, 4))

        tk.Label(api_input_frame, text="Gemini Key 1:", bg=COLOR_BG_SECONDARY, fg=COLOR_TEXT_SECONDARY, font=("Segoe UI", 9)).pack(side=tk.LEFT)
        entry_key1 = tk.Entry(api_input_frame, show="•", bg=COLOR_BG_PRIMARY, fg=COLOR_TEXT_PRIMARY, insertbackground=COLOR_CYAN_PRIMARY, font=("Segoe UI", 9), relief=tk.FLAT, bd=0, highlightbackground=COLOR_BORDER_SUBTLE, highlightthickness=1)
        existing_key = self.engine.key_manager.keys.get(1, "")
        if existing_key:
            entry_key1.insert(0, existing_key)
        entry_key1.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        lbl_onboarding_status = tk.Label(container, text="", bg=COLOR_BG_PRIMARY, fg=COLOR_ALERT_RED, font=("Segoe UI", 9))
        lbl_onboarding_status.pack(pady=(0, 4))

        def save_onboarding_and_start():
            uname = entry_uname.get().strip()
            if not uname:
                lbl_onboarding_status.config(text="Please enter your name to personalize SG CUBE.", fg=COLOR_ALERT_RED)
                entry_uname.focus_set()
                return

            dname = entry_dname.get().strip() or uname
            k1 = entry_key1.get().strip()

            self.engine.store.set_setting("user_name", uname)
            self.engine.store.set_setting("user_display_name", dname)
            self.engine.store.set_setting("first_run_completed", True)

            if k1:
                self.engine.key_manager.set_key(1, k1)

            print(f"[ONBOARDING] Profile saved: user_name='{uname}', display_name='{dname}'. Setup completed.")
            animate_dialog_close(dialog, duration_ms=180)

            # Trigger immediate startup greeting for configured user
            hour = time.localtime().tm_hour
            time_period = "morning" if 5 <= hour < 12 else ("afternoon" if 12 <= hour < 17 else ("evening" if 17 <= hour < 22 else "night"))
            greeting_msg = f"Good {time_period}, {uname}. I'm ready to help."
            self.gui_queue.put(("TRANSCRIPT_ASSISTIVE", greeting_msg))
            self.engine.response_manager.add_response(greeting_msg, priority=2, force=True)

        btn_finish = tk.Button(
            container,
            text="Complete Setup & Launch SG CUBE",
            font=("Segoe UI", 10, "bold"),
            bg=COLOR_CYAN_PRIMARY,
            fg=COLOR_BG_PRIMARY,
            activebackground=COLOR_TEAL_MINT,
            activeforeground=COLOR_BG_PRIMARY,
            relief=tk.FLAT,
            bd=0,
            padx=18,
            pady=8,
            cursor="hand2",
            command=save_onboarding_and_start
        )
        btn_finish.pack(pady=(4, 0))

    def on_close(self):
        print("[STATE] CLOSED")
        print("[HANDOFF] MAIN -> BACKGROUND")
        print("[SHUTDOWN] SG CUBE shutting down cleanly...")
        self.stop_ai()
        self.stop_camera()
        self._clear_playback_queue()
        self._notify_wake_listener_resume()
        time.sleep(0.15)
        try:
            self.root.destroy()
        except Exception:
            pass

def main():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(0.5)
        res = sock.connect_ex(('127.0.0.1', IPC_PORT_GUI))
        sock.close()
        if res == 0:
            print(f"[SINGLE-INSTANCE] Port {IPC_PORT_GUI} active. Sending WAKE IPC to existing instance and exiting before GUI creation.")
            try:
                s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                s.settimeout(1.0)
                s.connect(('127.0.0.1', IPC_PORT_GUI))
                s.sendall(b"WAKE")
                s.close()
            except Exception:
                pass
            sys.exit(0)
    except Exception:
        pass

    root = tk.Tk()
    app = SGCubeApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
