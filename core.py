# core.py
# ============================================
# ЯДРО СИСТЕМЫ - ВСЯ БАЗОВАЯ ЛОГИКА
# ============================================

import cv2
import numpy as np
import mss
import time
import pickle
import os
import random
import signal
import sys
import math
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set
from pynput.mouse import Button, Controller as MouseController, Listener as MouseListener
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyboardListener
import threading
from datetime import datetime

# ============================================
# КОНСТАНТЫ
# ============================================
GAME_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
MOUSE_IDLE_THRESHOLD = 1.5
LEARNING_INTERVAL = 10
RECORDING_BUFFER_SIZE = 500


# ============================================
# КЛАСС: SCAN CODE REGISTRY
# ============================================
class ScanCodeRegistry:
    KEY_CODES = {
        'a': 0x1E, 'b': 0x30, 'c': 0x2E, 'd': 0x20, 'e': 0x12,
        'f': 0x21, 'g': 0x22, 'h': 0x23, 'i': 0x17, 'j': 0x24,
        'k': 0x25, 'l': 0x26, 'm': 0x32, 'n': 0x31, 'o': 0x18,
        'p': 0x19, 'q': 0x10, 'r': 0x13, 's': 0x1F, 't': 0x14,
        'u': 0x16, 'v': 0x2F, 'w': 0x11, 'x': 0x2D, 'y': 0x15,
        'z': 0x2C,
        '0': 0x0B, '1': 0x02, '2': 0x03, '3': 0x04, '4': 0x05,
        '5': 0x06, '6': 0x07, '7': 0x08, '8': 0x09, '9': 0x0A,
        'f1': 0x3B, 'f2': 0x3C, 'f3': 0x3D, 'f4': 0x3E,
        'f5': 0x3F, 'f6': 0x40, 'f7': 0x41, 'f8': 0x42,
        'f9': 0x43, 'f10': 0x44, 'f11': 0x57, 'f12': 0x58,
        'space': 0x39,
        'enter': 0x1C,
        'escape': 0x01,
        'tab': 0x0F,
        'backspace': 0x0E,
        'shift': 0x2A,
        'ctrl': 0x1D,
        'alt': 0x38,
        'capslock': 0x3A,
        'numlock': 0x45,
        'scrolllock': 0x46,
        'pause': 0xC5,
        'insert': 0xD2,
        'delete': 0xD3,
        'home': 0xC7,
        'end': 0xCF,
        'page_up': 0xC9,
        'page_down': 0xD1,
        'print_screen': 0xE0 + 0x37,
        'sysrq': 0xE0 + 0x37,
        'up': 0xC8, 'down': 0xD0, 'left': 0xCB, 'right': 0xCD,
        'tilde': 0x29,
        'minus': 0x0C,
        'equals': 0x0D,
        'bracket_left': 0x1A,
        'bracket_right': 0x1B,
        'backslash': 0x2B,
        'semicolon': 0x27,
        'apostrophe': 0x28,
        'comma': 0x33,
        'period': 0x34,
        'slash': 0x35,
        'numpad_0': 0x52,
        'numpad_1': 0x4F,
        'numpad_2': 0x50,
        'numpad_3': 0x51,
        'numpad_4': 0x4B,
        'numpad_5': 0x4C,
        'numpad_6': 0x4D,
        'numpad_7': 0x47,
        'numpad_8': 0x48,
        'numpad_9': 0x49,
        'numpad_decimal': 0x53,
        'numpad_divide': 0xB5,
        'numpad_multiply': 0x37,
        'numpad_subtract': 0x4A,
        'numpad_add': 0x4E,
        'numpad_enter': 0x9C,
        'win_left': 0x5B,
        'win_right': 0x5C,
        'apps': 0x5D,
        'play_pause': 0xE0 + 0x22,
        'prev_track': 0xE0 + 0x10,
        'next_track': 0xE0 + 0x19,
        'media_select': 0xE0 + 0x6D,
        'calculator': 0xE0 + 0x21,
        'email': 0xE0 + 0x6C,
        'www': 0xE0 + 0x6F,
        'my_computer': 0xE0 + 0x6B,
    }

    CODE_TO_KEY = {v: k for k, v in KEY_CODES.items()}

    @classmethod
    def get_code(cls, key_name: str) -> int:
        return cls.KEY_CODES.get(key_name.lower(), 0)

    @classmethod
    def get_key_name(cls, code: int) -> str:
        return cls.CODE_TO_KEY.get(code, 'unknown')


# ============================================
# КЛАСС: SYSTEM PROTECTION
# ============================================
class SystemProtection:
    DANGEROUS_COMBOS = {
        'ctrl+alt+del': True,
        'ctrl+alt+delete': True,
        'alt+sysrq+b': True,
        'alt+sysrq+o': True,
        'alt+sysrq+s': True,
        'alt+sysrq+u': True,
        'alt+sysrq+e': True,
        'alt+sysrq+i': True,
        'alt+sysrq+r': True,
        'alt+sysrq+k': True,
        'alt+print_screen': True,
        'alt+sysrq': True,
        'alt+printscreen': True,
        'ctrl+alt+f1': True,
        'ctrl+alt+f2': True,
        'ctrl+alt+f3': True,
        'ctrl+alt+f4': True,
        'ctrl+alt+f5': True,
        'ctrl+alt+f6': True,
        'ctrl+alt+f7': True,
        'ctrl+alt+f8': True,
        'ctrl+alt+f9': True,
        'ctrl+alt+f10': True,
        'ctrl+alt+f11': True,
        'ctrl+alt+f12': True,
        'ctrl+alt+d': True,
        'ctrl+alt+l': True,
    }

    @classmethod
    def is_dangerous(cls, action: str) -> bool:
        action_lower = action.lower().strip()
        if action_lower in cls.DANGEROUS_COMBOS:
            return True
        if ('sysrq' in action_lower or 'print_screen' in action_lower or 'printscreen' in action_lower):
            if 'alt' in action_lower:
                return True
        if 'ctrl' in action_lower and 'alt' in action_lower:
            if 'del' in action_lower or 'delete' in action_lower:
                return True
        if 'ctrl' in action_lower and 'alt' in action_lower:
            if any(f'f{i}' in action_lower for i in range(1, 7)):
                return True
        return False


# ============================================
# КЛАСС: KEYBOARD EMULATOR (С СИНХРОНИЗАЦИЕЙ)
# ============================================
class KeyboardEmulator:
    def __init__(self, activity_tensor=None):
        self.keyboard = KeyboardController()
        self.pressed_keys = set()
        self.activity_tensor = activity_tensor
        self._lock = threading.Lock()

    def press_key(self, key_name: str, duration: float = 0.1, update_tensor: bool = True):
        with self._lock:
            try:
                key = self._get_key_object(key_name)
                if key:
                    self.keyboard.press(key)
                    self.pressed_keys.add(key_name)
                    if update_tensor and self.activity_tensor:
                        self.activity_tensor.update_key(key_name, True)
                    if duration > 0:
                        time.sleep(duration)
                        self.release_key(key_name, update_tensor=False)
                    return True
            except Exception as e:
                print(f"⚠️ Ошибка нажатия {key_name}: {e}")
            return False

    def release_key(self, key_name: str, update_tensor: bool = True):
        with self._lock:
            try:
                key = self._get_key_object(key_name)
                if key and key_name in self.pressed_keys:
                    self.keyboard.release(key)
                    self.pressed_keys.discard(key_name)
                    if update_tensor and self.activity_tensor:
                        self.activity_tensor.update_key(key_name, False)
                    return True
            except Exception as e:
                pass
            return False

    def release_all(self, update_tensor: bool = True):
        with self._lock:
            for key_name in list(self.pressed_keys):
                self.release_key(key_name, update_tensor=update_tensor)

    def press_combo(self, keys: List[str], duration: float = 0.1, update_tensor: bool = True):
        with self._lock:
            for key_name in keys:
                self.press_key(key_name, duration=0, update_tensor=update_tensor)
            time.sleep(duration)
            for key_name in reversed(keys):
                self.release_key(key_name, update_tensor=update_tensor)

    def hold_combo(self, keys: List[str], hold_duration: float, update_tensor: bool = True):
        with self._lock:
            for key_name in keys:
                self.press_key(key_name, duration=0, update_tensor=update_tensor)
            time.sleep(hold_duration)
            for key_name in reversed(keys):
                self.release_key(key_name, update_tensor=update_tensor)

    def is_pressed(self, key_name: str) -> bool:
        with self._lock:
            return key_name in self.pressed_keys

    def get_pressed_keys(self) -> Set[str]:
        with self._lock:
            return set(self.pressed_keys)

    def _get_key_object(self, key_name: str):
        special = {
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,
            'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
            'home': Key.home, 'end': Key.end,
            'page_up': Key.page_up, 'page_down': Key.page_down,
            'insert': Key.insert, 'delete': Key.delete,
            'space': Key.space, 'enter': Key.enter, 'tab': Key.tab,
            'escape': Key.esc, 'esc': Key.esc,
            'backspace': Key.backspace,
            'shift': Key.shift, 'ctrl': Key.ctrl, 'alt': Key.alt,
            'alt_l': Key.alt_l, 'alt_r': Key.alt_r,
            'ctrl_l': Key.ctrl_l, 'ctrl_r': Key.ctrl_r,
            'shift_l': Key.shift_l, 'shift_r': Key.shift_r,
            'win': Key.cmd, 'cmd': Key.cmd, 'windows': Key.cmd,
            'menu': Key.menu, 'apps': Key.menu,
            'print_screen': Key.print_screen, 'printscreen': Key.print_screen,
            'scroll_lock': Key.scroll_lock, 'pause': Key.pause,
            'num_lock': Key.num_lock,
            'caps_lock': Key.caps_lock,
        }

        if key_name.lower() in special:
            return special[key_name.lower()]
        if len(key_name) == 1:
            return key_name.lower()
        return None


# ============================================
# КЛАСС: MOUSE EMULATOR (С СИНХРОНИЗАЦИЕЙ)
# ============================================
class MouseEmulator:
    def __init__(self, activity_tensor=None):
        self.mouse = MouseController()
        self.pressed_buttons = set()
        self.activity_tensor = activity_tensor
        self._lock = threading.Lock()

        # Маппинг кнопок
        self.button_map = {
            'left': Button.left,
            'right': Button.right,
            'middle': Button.middle
        }

    def press_button(self, button: str, duration: float = 0.1, update_tensor: bool = True):
        with self._lock:
            try:
                btn = self._get_button(button)
                if btn:
                    self.mouse.press(btn)
                    self.pressed_buttons.add(button)
                    if update_tensor and self.activity_tensor:
                        self.activity_tensor.update_mouse_button(button, True)
                    if duration > 0:
                        time.sleep(duration)
                        self.release_button(button, update_tensor=False)
                    return True
            except Exception as e:
                print(f"⚠️ Ошибка нажатия кнопки мыши {button}: {e}")
            return False

    def release_button(self, button: str, update_tensor: bool = True):
        with self._lock:
            try:
                btn = self._get_button(button)
                if btn and button in self.pressed_buttons:
                    self.mouse.release(btn)
                    self.pressed_buttons.discard(button)
                    if update_tensor and self.activity_tensor:
                        self.activity_tensor.update_mouse_button(button, False)
                    return True
            except Exception as e:
                pass
            return False

    def release_all(self, update_tensor: bool = True):
        with self._lock:
            for button in list(self.pressed_buttons):
                self.release_button(button, update_tensor=update_tensor)

    def move_to(self, x: int, y: int, update_tensor: bool = True):
        with self._lock:
            self.mouse.position = (x, y)
            if update_tensor and self.activity_tensor:
                self.activity_tensor.update_mouse_position(x, y)

    def get_position(self) -> Tuple[int, int]:
        with self._lock:
            return self.mouse.position

    def scroll(self, dx: int = 0, dy: int = 0):
        with self._lock:
            self.mouse.scroll(dx, dy)

    def is_pressed(self, button: str) -> bool:
        with self._lock:
            return button in self.pressed_buttons

    def get_pressed_buttons(self) -> Set[str]:
        with self._lock:
            return set(self.pressed_buttons)

    def _get_button(self, button: str):
        return self.button_map.get(button.lower())


# ============================================
# КЛАСС: ACTIVITY TENSOR (СИНХРОНИЗИРОВАННЫЙ)
# ============================================
class ActivityTensor:
    """
    Синхронизированный тензор активности клавиш и мыши.
    Обновляется автоматически при любых действиях.
    """

    def __init__(self):
        self.key_states: Dict[str, bool] = defaultdict(bool)
        self.mouse_states: Dict[str, bool] = {
            'left': False,
            'right': False,
            'middle': False,
            'x1': False,
            'x2': False
        }
        self.mouse_x: int = 0
        self.mouse_y: int = 0
        self.history: deque = deque(maxlen=200)
        self.last_change_time: Dict[str, float] = defaultdict(float)
        self.press_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

        # Все клавиши
        self.all_keys = [
            'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j',
            'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't',
            'u', 'v', 'w', 'x', 'y', 'z',
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
            'f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8',
            'f9', 'f10', 'f11', 'f12',
            'space', 'enter', 'escape', 'tab', 'backspace',
            'shift', 'ctrl', 'alt', 'capslock',
            'numlock', 'scrolllock', 'pause',
            'insert', 'delete', 'home', 'end',
            'page_up', 'page_down',
            'print_screen', 'sysrq',
            'up', 'down', 'left', 'right',
            'tilde', 'minus', 'equals',
            'bracket_left', 'bracket_right',
            'backslash', 'semicolon', 'apostrophe',
            'comma', 'period', 'slash',
            'numpad_0', 'numpad_1', 'numpad_2', 'numpad_3', 'numpad_4',
            'numpad_5', 'numpad_6', 'numpad_7', 'numpad_8', 'numpad_9',
            'numpad_decimal', 'numpad_divide', 'numpad_multiply',
            'numpad_subtract', 'numpad_add', 'numpad_enter',
            'win', 'cmd', 'menu', 'apps'
        ]

        self.key_index = {key: i for i, key in enumerate(self.all_keys)}
        self.mouse_index = {'left': 0, 'right': 1, 'middle': 2, 'x1': 3, 'x2': 4}
        self.tensor_size = len(self.all_keys) + 5 + 2

    def update_key(self, key_name: str, pressed: bool):
        with self._lock:
            key_name = key_name.lower().strip()
            if key_name in self.key_index:
                if self.key_states[key_name] != pressed:
                    self.key_states[key_name] = pressed
                    self.last_change_time[key_name] = time.time()
                    if pressed:
                        self.press_counts[key_name] += 1
                    self._record_history(f"key:{key_name}:{pressed}")

    def update_mouse_button(self, button: str, pressed: bool):
        with self._lock:
            button = button.lower().strip()
            if button in self.mouse_states:
                if self.mouse_states[button] != pressed:
                    self.mouse_states[button] = pressed
                    self.last_change_time[f"mouse_{button}"] = time.time()
                    if pressed:
                        self.press_counts[f"mouse_{button}"] += 1
                    self._record_history(f"mouse:{button}:{pressed}")

    def update_mouse_position(self, x: int, y: int):
        with self._lock:
            if self.mouse_x != x or self.mouse_y != y:
                self.mouse_x = x
                self.mouse_y = y
                self._record_history(f"pos:{x},{y}")

    def _record_history(self, event: str):
        self.history.append((time.time(), event))

    def get_tensor(self) -> np.ndarray:
        with self._lock:
            key_vector = np.zeros(len(self.all_keys), dtype=np.float32)
            for key, state in self.key_states.items():
                if key in self.key_index:
                    key_vector[self.key_index[key]] = 1.0 if state else 0.0

            mouse_vector = np.zeros(5, dtype=np.float32)
            for btn, state in self.mouse_states.items():
                if btn in self.mouse_index:
                    mouse_vector[self.mouse_index[btn]] = 1.0 if state else 0.0

            pos_vector = np.array([
                self.mouse_x / 1920.0,
                self.mouse_y / 1080.0
            ], dtype=np.float32)

            return np.concatenate([key_vector, mouse_vector, pos_vector])

    def get_dense_tensor(self) -> np.ndarray:
        with self._lock:
            base = self.get_tensor()
            current_time = time.time()

            time_since = np.zeros(len(self.all_keys), dtype=np.float32)
            for key, last_time in self.last_change_time.items():
                if key in self.key_index:
                    time_since[self.key_index[key]] = min(1.0, (current_time - last_time) / 5.0)

            window_size = 2.0
            recent_activity = np.zeros(len(self.all_keys) + 5, dtype=np.float32)

            for t, event in self.history:
                if current_time - t < window_size:
                    parts = event.split(':')
                    if len(parts) >= 3:
                        if parts[0] == 'key' and parts[1] in self.key_index:
                            idx = self.key_index[parts[1]]
                            if parts[2] == 'True':
                                recent_activity[idx] = min(1.0, recent_activity[idx] + 0.1)
                        elif parts[0] == 'mouse' and parts[1] in self.mouse_index:
                            idx = len(self.all_keys) + self.mouse_index[parts[1]]
                            if parts[2] == 'True':
                                recent_activity[idx] = min(1.0, recent_activity[idx] + 0.1)

            return np.concatenate([base, time_since, recent_activity])

    def get_active_keys(self) -> List[str]:
        with self._lock:
            return [key for key, state in self.key_states.items() if state]

    def get_active_mouse_buttons(self) -> List[str]:
        with self._lock:
            return [btn for btn, state in self.mouse_states.items() if state]

    def get_mouse_position(self) -> Tuple[int, int]:
        with self._lock:
            return (self.mouse_x, self.mouse_y)

    def is_key_pressed(self, key_name: str) -> bool:
        with self._lock:
            return self.key_states.get(key_name.lower(), False)

    def is_mouse_pressed(self, button: str) -> bool:
        with self._lock:
            return self.mouse_states.get(button.lower(), False)

    def get_activity_summary(self) -> Dict:
        with self._lock:
            return {
                'active_keys': self.get_active_keys(),
                'active_mouse': self.get_active_mouse_buttons(),
                'mouse_position': (self.mouse_x, self.mouse_y),
                'total_key_presses': sum(self.press_counts.values()),
                'total_mouse_clicks': sum(self.press_counts.get(f"mouse_{btn}", 0)
                                          for btn in self.mouse_states.keys()),
                'key_count': len(self.key_states),
                'mouse_buttons': dict(self.mouse_states),
                'last_events': list(self.history)[-10:] if self.history else []
            }

    def reset(self):
        with self._lock:
            self.key_states.clear()
            for btn in self.mouse_states:
                self.mouse_states[btn] = False
            self.mouse_x = 0
            self.mouse_y = 0
            self.history.clear()
            self.last_change_time.clear()


# ============================================
# КЛАСС: SCREEN TOKENIZER
# ============================================
class ScreenTokenizer:
    def __init__(self, grid_height: int = 34, grid_width: int = 66):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.total_cells = grid_height * grid_width
        self.history = deque(maxlen=300)
        self.state_buffer = deque(maxlen=20)

    def encode_frame(self, frame: np.ndarray, keys: List[str] = None) -> List[int]:
        if frame is None:
            return [0] * self.total_cells
        small = cv2.resize(frame, (self.grid_width, self.grid_height))
        gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        tokens = gray.flatten().tolist()
        tokens = [int(t) for t in tokens]
        if keys:
            for key in keys[:10]:
                code = ScanCodeRegistry.get_code(key)
                tokens.append(code if code > 0 else 1)
        else:
            tokens.extend([0] * 10)
        self.history.append(tokens)
        self.state_buffer.append(tokens[:self.total_cells])
        return tokens

    def get_state_hash(self, tokens: List[int]) -> int:
        return hash(tuple(tokens))

    def get_context(self) -> List[int]:
        if not self.state_buffer:
            return [0] * min(100, self.total_cells)
        avg = np.mean(self.state_buffer, axis=0)
        return [int(v) for v in avg[:min(100, self.total_cells)]]

    def visualize_grid(self, tokens: List[int]) -> np.ndarray:
        if len(tokens) < self.total_cells:
            return np.zeros((self.grid_height, self.grid_width), dtype=np.uint8)
        grid = np.array(tokens[:self.total_cells]).reshape(self.grid_height, self.grid_width)
        grid = grid.astype(np.uint8)
        scale = 30
        enlarged = cv2.resize(grid, (self.grid_width * scale, self.grid_height * scale),
                              interpolation=cv2.INTER_NEAREST)
        for i in range(self.grid_height + 1):
            cv2.line(enlarged, (0, i * scale), (self.grid_width * scale, i * scale), (100, 100, 100), 1)
        for i in range(self.grid_width + 1):
            cv2.line(enlarged, (i * scale, 0), (i * scale, self.grid_height * scale), (100, 100, 100), 1)
        return enlarged

    def get_dimensions(self) -> Tuple[int, int]:
        return self.grid_height, self.grid_width


# ============================================
# КЛАСС: HYBRID DECISION TREE NODE
# ============================================
class HybridTreeNode:
    def __init__(self, feature_idx: int = None, threshold: float = None,
                 left: 'HybridTreeNode' = None, right: 'HybridTreeNode' = None,
                 value: float = None):
        self.feature_idx = feature_idx
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.leaf_count = 0

    def predict(self, x: List[float]) -> float:
        if self.value is not None:
            return self.value
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict(x)
        else:
            return self.right.predict(x)


# ============================================
# КЛАСС: HYBRID DECISION FOREST
# ============================================
class HybridDecisionForest:
    def __init__(self, n_trees: int = 14, max_depth: int = 8):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.is_trained = False
        self.feature_importance = defaultdict(float)
        self.training_count = 0
        self.branch_weights = {}
        self.node_visits = defaultdict(int)
        self.branch_performance = defaultdict(list)
        self.global_branch_weights = {'split': 1.0, 'leaf': 1.0}
        self.weight_decay = 0.99
        self.min_weight = 0.01
        self.max_weight = 1.0
        self.learning_rate = 0.05

    def fit(self, X: List[List[float]], y: List[float], update_existing: bool = False):
        if len(X) < 10:
            return
        self.training_count += 1
        if update_existing and self.is_trained:
            for i in range(min(3, self.n_trees // 2)):
                indices = np.random.choice(len(X), len(X), replace=True)
                X_sample = [X[j] for j in indices]
                y_sample = [y[j] for j in indices]
                tree = self._build_tree(X_sample, y_sample, 0, tree_idx=i)
                if len(self.trees) < self.n_trees:
                    self.trees.append(tree)
                else:
                    idx = random.randint(0, len(self.trees) - 1)
                    self.trees[idx] = tree
        else:
            self.trees = []
            self.branch_weights = {}
            self.node_visits = defaultdict(int)
            self.branch_performance = defaultdict(list)
            for i in range(self.n_trees):
                indices = np.random.choice(len(X), len(X), replace=True)
                X_sample = [X[j] for j in indices]
                y_sample = [y[j] for j in indices]
                tree = self._build_tree(X_sample, y_sample, 0, tree_idx=i)
                self.trees.append(tree)
            self.is_trained = True
        print(f"🌲 Лес обучен на {len(X)} примерах (обновление #{self.training_count})")

    def _build_tree(self, X: List[List[float]], y: List[float], depth: int,
                    tree_idx: int = 0, node_id: int = 0) -> HybridTreeNode:
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < 5:
            node = HybridTreeNode(value=np.mean(y))
            node.leaf_count = len(X)
            self._update_branch_weight(tree_idx, node_id, node.value, is_leaf=True)
            return node
        best_feature, best_threshold, best_gain = 0, 0, -1
        n_features = len(X[0])
        features_to_try = min(n_features, 30)
        feature_indices = random.sample(range(n_features), features_to_try)
        for feature in feature_indices:
            values = sorted(set(x[feature] for x in X))
            step = max(1, len(values) // 10)
            for threshold in values[::step]:
                left_y = [y[i] for i, x in enumerate(X) if x[feature] <= threshold]
                right_y = [y[i] for i, x in enumerate(X) if x[feature] > threshold]
                if len(left_y) < 2 or len(right_y) < 2:
                    continue
                gain = self._information_gain(y, left_y, right_y)
                if gain > best_gain:
                    best_gain, best_feature, best_threshold = gain, feature, threshold
        if best_gain <= 0:
            node = HybridTreeNode(value=np.mean(y))
            node.leaf_count = len(X)
            self._update_branch_weight(tree_idx, node_id, node.value, is_leaf=True)
            return node
        left_X = [x for x in X if x[best_feature] <= best_threshold]
        right_X = [x for x in X if x[best_feature] > best_threshold]
        left_y = [y[i] for i, x in enumerate(X) if x[best_feature] <= best_threshold]
        right_y = [y[i] for i, x in enumerate(X) if x[best_feature] > best_threshold]
        self.feature_importance[best_feature] += best_gain
        left_node = self._build_tree(left_X, left_y, depth + 1, tree_idx, node_id * 2 + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1, tree_idx, node_id * 2 + 2)
        node = HybridTreeNode(best_feature, best_threshold, left_node, right_node)
        self._update_branch_weight(tree_idx, node_id, best_gain, is_leaf=False)
        return node

    def _update_branch_weight(self, tree_idx: int, node_id: int, value: float, is_leaf: bool = False):
        key = (tree_idx, node_id)
        node_type = 'leaf' if is_leaf else 'split'
        base_weight = abs(value) if not is_leaf else 0.5
        global_weight = self.global_branch_weights.get(node_type, 1.0)
        if key in self.branch_weights:
            old_weight = self.branch_weights[key]
            new_weight = old_weight * self.weight_decay + (1 - self.weight_decay) * base_weight * global_weight
        else:
            new_weight = base_weight * global_weight
        new_weight = max(self.min_weight, min(self.max_weight, new_weight))
        self.branch_weights[key] = new_weight
        self.node_visits[key] += 1

    def predict(self, x: List[float]) -> float:
        if not self.is_trained or not self.trees:
            return 0.5
        predictions = []
        weights = []
        for tree_idx, tree in enumerate(self.trees):
            pred, path = self._predict_with_path(tree, x)
            weight = self._get_path_weight(tree_idx, path)
            predictions.append(pred)
            weights.append(weight)
        total_weight = sum(weights)
        if total_weight > 0:
            normalized_weights = [w / total_weight for w in weights]
        else:
            normalized_weights = [1.0 / len(predictions)] * len(predictions)
        weighted_pred = sum(p * w for p, w in zip(predictions, normalized_weights))
        return weighted_pred

    def _predict_with_path(self, tree: HybridTreeNode, x: List[float]) -> Tuple[float, List[int]]:
        path = []
        node = tree
        while node.value is None and node.left is not None and node.right is not None:
            path.append(node.feature_idx)
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value if node.value is not None else 0.5, path

    def _get_path_weight(self, tree_idx: int, path: List[int]) -> float:
        if not path:
            return 1.0
        total_weight = 0.0
        for depth, feature_idx in enumerate(path):
            node_id = sum(2 ** i for i in range(depth))
            key = (tree_idx, node_id)
            weight = self.branch_weights.get(key, 0.5)
            depth_factor = 1.0 / (depth + 1) ** 0.5
            total_weight += weight * depth_factor
        return total_weight / len(path) if path else 1.0

    def predict_proba(self, x: List[float]) -> List[float]:
        if not self.is_trained or not self.trees:
            return [0.11] * 9
        pred = self.predict(x)
        probs = [0.05] * 9
        idx = min(int(pred * 9), 8)
        total_weight = sum(self.branch_weights.values()) if self.branch_weights else 1.0
        confidence = min(0.9, 0.5 + (len(self.branch_weights) / (self.n_trees * 10)))
        probs[idx] = confidence
        remaining = 1.0 - confidence
        for i in range(9):
            if i != idx:
                distance = abs(i - idx)
                probs[i] = remaining * (1.0 / (distance + 1)) / sum(
                    1.0 / (abs(j - idx) + 1) for j in range(9) if j != idx)
        total = sum(probs)
        return [p / total for p in probs]

    def update_weights_online(self, reward: float, path: List[int], tree_idx: int = 0):
        if not path:
            return
        for depth, feature_idx in enumerate(path):
            node_id = sum(2 ** i for i in range(depth))
            key = (tree_idx, node_id)
            if key in self.branch_weights:
                adjustment = self.learning_rate * reward * (1.0 / (depth + 1))
                new_weight = self.branch_weights[key] * (1 + adjustment)
                new_weight = max(self.min_weight, min(self.max_weight, new_weight))
                self.branch_weights[key] = new_weight
                self.branch_performance[key].append(reward)
                if len(self.branch_performance[key]) > 100:
                    self.branch_performance[key] = self.branch_performance[key][-100:]
        if reward > 0.5:
            self.global_branch_weights['split'] *= (1 + self.learning_rate * 0.1)
            self.global_branch_weights['leaf'] *= (1 + self.learning_rate * 0.05)
        else:
            self.global_branch_weights['split'] *= (1 - self.learning_rate * 0.05)
            self.global_branch_weights['leaf'] *= (1 - self.learning_rate * 0.02)
        for key in list(self.global_branch_weights.keys()):
            self.global_branch_weights[key] = max(0.5, min(2.0, self.global_branch_weights[key]))

    def get_branch_statistics(self) -> Dict:
        stats = {
            'total_weights': len(self.branch_weights),
            'average_weight': sum(self.branch_weights.values()) / len(
                self.branch_weights) if self.branch_weights else 0,
            'min_weight': min(self.branch_weights.values()) if self.branch_weights else 0,
            'max_weight': max(self.branch_weights.values()) if self.branch_weights else 0,
            'global_weights': dict(self.global_branch_weights),
            'most_visited': sorted(self.node_visits.items(), key=lambda x: x[1], reverse=True)[:10]
        }
        return stats

    def _information_gain(self, parent: List[float], left: List[float], right: List[float]) -> float:
        def entropy(values):
            if not values:
                return 0
            probs = [values.count(v) / len(values) for v in set(values)]
            return -sum(p * math.log2(p) for p in probs if p > 0)

        p = len(left) / len(parent)
        return entropy(parent) - p * entropy(left) - (1 - p) * entropy(right)


# ============================================
# КЛАСС: SMART REWARD SYSTEM
# ============================================
class SmartRewardSystem:
    def __init__(self):
        self.reward_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.baseline_change = 0.05
        self.movement_threshold = 0.03
        self.stuck_threshold = 0.01
        self.last_changes = deque(maxlen=20)
        self.pattern_memory = defaultdict(list)
        self.action_success_rate = defaultdict(float)
        self.consecutive_failures = 0

    def calculate_smart_reward(self, before_state, after_state, action, context):
        change = self._calculate_change(before_state, after_state)
        self.last_changes.append(change)
        chaos_penalty = self._calculate_chaos_penalty(change)
        stagnation_penalty = self._calculate_stagnation_penalty(change)
        progress_bonus = self._calculate_progress_bonus(change)
        repetition_penalty = self._calculate_repetition_penalty(action)
        diversity_bonus = self._calculate_diversity_bonus(action)
        spam_penalty = self._calculate_spam_penalty()
        pattern_bonus = self._calculate_pattern_bonus(before_state, after_state)
        reward = (
                + progress_bonus * 2.0
                - chaos_penalty * 1.5
                - stagnation_penalty * 2.0
                - repetition_penalty * 0.5
                + diversity_bonus * 0.3
                - spam_penalty * 1.0
                + pattern_bonus * 0.8
        )
        reward = np.clip(reward, -3.0, 3.0)
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.action_success_rate[action] = self.action_success_rate.get(action, 0) * 0.9 + (reward > 0) * 0.1
        return reward

    def _calculate_change(self, before, after):
        if before is None or after is None:
            return 0.0
        if len(before) == 0 or len(after) == 0:
            return 0.0
        if len(before) != len(after):
            return 0.0
        diff = 0.0
        for i in range(len(before)):
            diff += abs(float(before[i]) - float(after[i]))
        return diff / len(before)

    def _calculate_chaos_penalty(self, change):
        if change > 0.2:
            penalty = (change - 0.2) * 5.0
            return min(penalty, 2.0)
        return 0.0

    def _calculate_stagnation_penalty(self, change):
        if change < self.stuck_threshold:
            self.consecutive_failures += 1
            penalty = min(self.consecutive_failures * 0.2, 2.0)
            return penalty
        else:
            self.consecutive_failures = 0
            return 0.0

    def _calculate_progress_bonus(self, change):
        if self.movement_threshold < change < 0.15:
            optimal = 0.08
            sigma = 0.05
            bonus = math.exp(-((change - optimal) ** 2) / (2 * sigma ** 2))
            return bonus * 1.5
        return 0.0

    def _calculate_repetition_penalty(self, action):
        if len(self.action_history) < 5:
            return 0.0
        recent_actions = list(self.action_history)
        if len(recent_actions) > 10:
            recent_actions = recent_actions[len(recent_actions) - 10:]
        repetition_count = recent_actions.count(action)
        if repetition_count > 3:
            penalty = (repetition_count - 3) * 0.3
            return min(penalty, 1.0)
        return 0.0

    def _calculate_diversity_bonus(self, action):
        if len(self.action_history) < 10:
            return 0.0
        action_list = list(self.action_history)
        if len(action_list) > 20:
            action_list = action_list[len(action_list) - 20:]
        unique_actions = len(set(action_list))
        diversity = unique_actions / min(20, len(self.action_history))
        if diversity > 0.5:
            return diversity * 0.3
        return 0.0

    def _calculate_spam_penalty(self):
        if len(self.action_history) < 5:
            return 0.0
        recent_actions = list(self.action_history)
        if len(recent_actions) > 10:
            recent_actions = recent_actions[len(recent_actions) - 10:]
        if len(recent_actions) > 8:
            penalty = (len(recent_actions) - 8) * 0.2
            return min(penalty, 1.0)
        return 0.0

    def _calculate_pattern_bonus(self, before, after):
        if before is None or after is None:
            return 0.0
        if len(before) == 0 or len(after) == 0:
            return 0.0
        if len(before) <= 150 or len(after) <= 150:
            return 0.0

        center_before = before[100:150]
        center_after = after[100:150]

        if len(center_before) == 0 or len(center_after) == 0:
            return 0.0

        center_change = 0.0
        for i in range(len(center_before)):
            center_change += abs(float(center_before[i]) - float(center_after[i]))
        center_change = center_change / len(center_before)

        if 0.02 < center_change < 0.1:
            return 0.5
        return 0.0

    def get_action_feedback(self, action):
        success_rate = self.action_success_rate.get(action, 0.0)
        if success_rate > 0.7:
            return "✅ Эффективно"
        elif success_rate > 0.4:
            return "🟡 Нормально"
        else:
            return "❌ Неэффективно"

    def get_statistics(self):
        avg_reward = np.mean(list(self.reward_history)) if self.reward_history else 0
        std_reward = np.std(list(self.reward_history)) if self.reward_history else 0
        action_list = list(self.action_history)
        if len(action_list) > 20:
            action_list = action_list[len(action_list) - 20:]
        return {
            'avg_reward': avg_reward,
            'std_reward': std_reward,
            'consecutive_failures': self.consecutive_failures,
            'action_diversity': len(set(action_list)) if action_list else 0,
            'best_action': max(self.action_success_rate,
                               key=self.action_success_rate.get) if self.action_success_rate else 'none',
            'best_action_rate': max(self.action_success_rate.values()) if self.action_success_rate else 0
        }


# ============================================
# КЛАСС: ACTION EXECUTOR (СИНХРОНИЗИРОВАННЫЙ)
# ============================================
class ActionExecutor:
    """
    Синхронизированный исполнитель действий.
    Объединяет клавиатуру и мышь с общей синхронизацией.
    """

    def __init__(self, activity_tensor: ActivityTensor):
        self.activity_tensor = activity_tensor
        self.keyboard = KeyboardEmulator(activity_tensor)
        self.mouse = MouseEmulator(activity_tensor)
        self.protection = SystemProtection()
        self.blocked_actions = []
        self._lock = threading.Lock()

    def execute_mouse_only(self, action: str, x: int = None, y: int = None) -> bool:
        """Выполняет действие только с мышью"""
        with self._lock:
            if x is None or y is None:
                x = random.randint(200, 1700)
                y = random.randint(200, 800)

            if action == 'click':
                return self._do_click(x, y, 'left')
            elif action == 'double_click':
                return self._do_click(x, y, 'left', clicks=2)
            elif action == 'right_click':
                return self._do_click(x, y, 'right')
            elif action == 'middle_click':
                return self._do_click(x, y, 'middle')
            elif action == 'drag_short':
                return self._do_drag(x, y, random.randint(50, 150))
            elif action == 'drag_medium':
                return self._do_drag(x, y, random.randint(150, 300))
            elif action == 'drag_long':
                return self._do_drag(x, y, random.randint(300, 500))
            elif action == 'scroll_up':
                self.mouse.scroll(dy=-1)
                return True
            elif action == 'scroll_down':
                self.mouse.scroll(dy=1)
                return True
            elif action == 'scroll_left':
                self.mouse.scroll(dx=-1)
                return True
            elif action == 'scroll_right':
                self.mouse.scroll(dx=1)
                return True
            return False

    def execute_keyboard_only(self, action: str) -> bool:
        """Выполняет действие только с клавиатурой"""
        with self._lock:
            if action == 'release_all':
                self.keyboard.release_all()
                self.mouse.release_all()
                return True
            if '+' in action:
                combo = action.split('+')
                return self._do_key_combo(combo)
            return self.keyboard.press_key(action, duration=random.uniform(0.05, 0.15))

    def execute_mixed(self, action: str) -> bool:
        """
        Выполняет смешанное действие: клавиши + мышь.
        Формат: "key1+key2+mouse_button" или "ctrl+shift+left"
        """
        with self._lock:
            parts = action.split('+')
            mouse_buttons = {'left': 'left', 'right': 'right', 'middle': 'middle'}

            mouse_part = None
            key_parts = []

            for part in parts:
                part_lower = part.lower().strip()
                if part_lower in mouse_buttons:
                    mouse_part = mouse_buttons[part_lower]
                else:
                    key_parts.append(part_lower)

            if mouse_part is None:
                return False

            # Проверяем опасные комбинации
            combo_str = '+'.join(key_parts + [mouse_part])
            if self.protection.is_dangerous(combo_str):
                self.blocked_actions.append(combo_str)
                print(f"🛡️ ЗАБЛОКИРОВАНО: {combo_str}")
                return False

            try:
                # Зажимаем клавиши
                for key in key_parts:
                    self.keyboard.press_key(key, duration=0, update_tensor=True)

                time.sleep(random.uniform(0.01, 0.03))

                # Нажимаем кнопку мыши
                self.mouse.press_button(mouse_part, duration=0.1, update_tensor=True)

                # Отпускаем клавиши
                for key in reversed(key_parts):
                    self.keyboard.release_key(key, update_tensor=True)

                return True

            except Exception as e:
                print(f"⚠️ Ошибка смешанного действия: {e}")
                self.keyboard.release_all(update_tensor=True)
                self.mouse.release_all(update_tensor=True)
                return False

    def execute_sequence(self, actions: List[str], delay: float = 0.1) -> bool:
        """Выполняет последовательность действий"""
        with self._lock:
            for action in actions:
                if not self.execute(action):
                    return False
                time.sleep(delay)
            return True

    def execute(self, action: str) -> bool:
        """Основной метод выполнения любого действия"""
        if self.protection.is_dangerous(action):
            self.blocked_actions.append(action)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {action}")
            return False

        if action in ['wait_short', 'wait_long']:
            duration = 0.5 if 'short' in action else 2.0
            time.sleep(duration + random.random() * 0.3)
            return True

        if action == 'release_all':
            self.keyboard.release_all(update_tensor=True)
            self.mouse.release_all(update_tensor=True)
            return True

        # Проверяем смешанные действия
        if any(btn in action for btn in ['left', 'right', 'middle']) and '+' in action:
            return self.execute_mixed(action)

        # Проверяем комбинации клавиш
        if '+' in action:
            return self.execute_keyboard_only(action)

        # Проверяем действия мыши
        if action in ['click', 'double_click', 'right_click', 'middle_click',
                      'drag_short', 'drag_medium', 'drag_long',
                      'scroll_up', 'scroll_down', 'scroll_left', 'scroll_right']:
            return self.execute_mouse_only(action)

        # Одиночная клавиша
        return self.execute_keyboard_only(action)

    def _do_click(self, x: int, y: int, button: str, clicks: int = 1):
        """Выполняет клик с человеческим поведением"""
        self.mouse.move_to(x + random.randint(-8, 8), y + random.randint(-8, 8), update_tensor=True)
        for i in range(clicks):
            self.mouse.press_button(button, duration=0, update_tensor=True)
            time.sleep(random.uniform(0.03, 0.08))
            self.mouse.release_button(button, update_tensor=True)
            if i < clicks - 1:
                time.sleep(random.uniform(0.05, 0.15))
        return True

    def _do_drag(self, from_x: int, from_y: int, distance: int):
        """Выполняет перетаскивание"""
        to_x = from_x + random.randint(-distance, distance)
        to_y = from_y + random.randint(-distance, distance)

        self.mouse.move_to(from_x, from_y, update_tensor=True)
        time.sleep(random.uniform(0.05, 0.1))

        self.mouse.press_button('left', duration=0, update_tensor=True)
        time.sleep(random.uniform(0.05, 0.1))

        steps = random.randint(10, 20)
        for i in range(steps):
            progress = i / steps
            deviation = math.sin(progress * math.pi) * random.randint(2, 10)
            x = from_x + (to_x - from_x) * progress + deviation
            y = from_y + (to_y - from_y) * progress + deviation * 0.2
            self.mouse.move_to(int(x), int(y), update_tensor=False)
            time.sleep(random.uniform(0.005, 0.015))

        self.mouse.move_to(to_x + random.randint(-3, 3), to_y + random.randint(-3, 3), update_tensor=True)
        time.sleep(random.uniform(0.02, 0.05))
        self.mouse.release_button('left', update_tensor=True)
        return True

    def _do_key_combo(self, combo: List[str]) -> bool:
        """Выполняет комбинацию клавиш"""
        normalized = []
        for key in combo:
            key = key.lower().strip()
            if key == 'win':
                key = 'cmd'
            normalized.append(key)

        combo_str = '+'.join(normalized)
        if self.protection.is_dangerous(combo_str):
            self.blocked_actions.append(combo_str)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {combo_str}")
            return False

        self.keyboard.press_combo(normalized, duration=random.uniform(0.1, 0.3), update_tensor=True)
        return True


# ============================================
# КЛАСС: STATE ENCODER (ОБЪЕДИНЯЕТ ВСЁ)
# ============================================
class StateEncoder:
    """
    Кодирует полное состояние: экран + активность клавиш/мыши
    """

    def __init__(self, activity_tensor: ActivityTensor, tokenizer: ScreenTokenizer):
        self.activity_tensor = activity_tensor
        self.tokenizer = tokenizer

    def encode(self, frame: np.ndarray, keys: List[str] = None) -> List[float]:
        """Кодирует полное состояние"""
        # Токены экрана
        tokens = self.tokenizer.encode_frame(frame, keys)
        context = self.tokenizer.get_context()

        # Тензор активности
        activity = self.activity_tensor.get_tensor()

        # Объединяем
        combined = tokens + context + list(activity)

        # Нормализуем
        result = [float(t) / 255.0 for t in combined[:350]]
        return result

    def encode_dense(self, frame: np.ndarray, keys: List[str] = None) -> List[float]:
        """Кодирует с плотным тензором активности"""
        tokens = self.tokenizer.encode_frame(frame, keys)
        context = self.tokenizer.get_context()
        dense_activity = self.activity_tensor.get_dense_tensor()

        combined = tokens + context + list(dense_activity)
        result = [float(t) / 255.0 for t in combined[:400]]
        return result