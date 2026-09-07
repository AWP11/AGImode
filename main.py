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
from typing import List, Tuple, Dict, Optional, Any
from pynput.mouse import Button, Controller as MouseController, Listener as MouseListener
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyboardListener
import threading
from datetime import datetime

GAME_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
MOUSE_IDLE_THRESHOLD = 1.5
LEARNING_INTERVAL = 10
RECORDING_BUFFER_SIZE = 500


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


class KeyboardEmulator:
    def __init__(self):
        self.keyboard = KeyboardController()
        self.pressed_keys = set()

    def press_key(self, key_name: str, duration: float = 0.1):
        try:
            key = self._get_key_object(key_name)
            if key:
                self.keyboard.press(key)
                self.pressed_keys.add(key_name)
                if duration > 0:
                    time.sleep(duration)
                    self.release_key(key_name)
                return True
        except Exception as e:
            print(f"⚠️ Ошибка нажатия {key_name}: {e}")
        return False

    def release_key(self, key_name: str):
        try:
            key = self._get_key_object(key_name)
            if key and key_name in self.pressed_keys:
                self.keyboard.release(key)
                self.pressed_keys.discard(key_name)
                return True
        except Exception as e:
            pass
        return False

    def release_all(self):
        for key_name in list(self.pressed_keys):
            self.release_key(key_name)

    def press_combo(self, keys: List[str], duration: float = 0.1):
        for key_name in keys:
            self.press_key(key_name, duration=0)
        time.sleep(duration)
        for key_name in reversed(keys):
            self.release_key(key_name)

    def hold_combo(self, keys: List[str], hold_duration: float):
        for key_name in keys:
            self.press_key(key_name, duration=0)
        time.sleep(hold_duration)
        for key_name in reversed(keys):
            self.release_key(key_name)

    def _get_key_object(self, key_name: str):
        # Безопасный доступ к специальным клавишам
        special = {
            # Функциональные клавиши
            'f1': Key.f1, 'f2': Key.f2, 'f3': Key.f3, 'f4': Key.f4,
            'f5': Key.f5, 'f6': Key.f6, 'f7': Key.f7, 'f8': Key.f8,
            'f9': Key.f9, 'f10': Key.f10, 'f11': Key.f11, 'f12': Key.f12,

            # Навигационные клавиши
            'up': Key.up, 'down': Key.down, 'left': Key.left, 'right': Key.right,
            'home': Key.home, 'end': Key.end,
            'page_up': Key.page_up, 'page_down': Key.page_down,
            'insert': Key.insert, 'delete': Key.delete,

            # Служебные клавиши
            'space': Key.space, 'enter': Key.enter, 'tab': Key.tab,
            'escape': Key.esc, 'esc': Key.esc,
            'backspace': Key.backspace,

            # Модификаторы
            'shift': Key.shift, 'ctrl': Key.ctrl, 'alt': Key.alt,
            'alt_l': Key.alt_l, 'alt_r': Key.alt_r,
            'ctrl_l': Key.ctrl_l, 'ctrl_r': Key.ctrl_r,
            'shift_l': Key.shift_l, 'shift_r': Key.shift_r,

            # Системные
            'win': Key.cmd, 'cmd': Key.cmd, 'windows': Key.cmd,
            'menu': Key.menu, 'apps': Key.menu,

            # Print Screen
            'print_screen': Key.print_screen, 'printscreen': Key.print_screen,

            # Scroll и Pause
            'scroll_lock': Key.scroll_lock, 'pause': Key.pause,
            'num_lock': Key.num_lock,

            # Медиа-клавиши (только если поддерживаются)
            'caps_lock': Key.caps_lock,
        }

        # Проверяем, есть ли клавиша в special
        if key_name.lower() in special:
            return special[key_name.lower()]

        # Для одиночных символов
        if len(key_name) == 1:
            return key_name.lower()

        # Игнорируем неподдерживаемые медиа-клавиши
        return None

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
        print(f"⚖️  Активных весов ветвей: {len(self.branch_weights)}")

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


class HybridForestBot:
    def __init__(self):
        self.sct = mss.mss()
        self.mouse = MouseController()
        self.keyboard = KeyboardEmulator()
        self.mouse_is_pressed = False
        self.press_start_time = 0
        self.press_start_pos = (0, 0)
        self.current_state = []
        self.last_mouse_move = time.time()
        self.last_action_time = time.time()
        self.action_counter = 0
        self.reward_history = deque(maxlen=100)
        self.bot_state = 'idle'
        self.recording = False
        self.auto_mode = False
        self.running = False
        self.forest = HybridDecisionForest(n_trees=16, max_depth=6)
        self.tokenizer = ScreenTokenizer(grid_height=66, grid_width=88)
        self.training_data = []
        self.training_labels = []
        self.experience_history = deque(maxlen=500)
        self.recent_actions = deque(maxlen=20)
        self.recent_rewards = deque(maxlen=20)
        self.model_file = "hybrid_forest_bot.pkl"
        self.protection = SystemProtection()
        self.blocked_actions = []
        self.reward_system = SmartRewardSystem()
        self.adaptive_learning = True
        self.learning_phase = 'exploration'
        self.phase_switches = 0
        self.exploration_threshold = 50
        self.exploitation_threshold = 200

        # ВСЕ КЛЮЧИ ДОЛЖНЫ БЫТЬ ОПРЕДЕЛЕНЫ ДО ЗАГРУЗКИ МОДЕЛИ
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
            'win_left', 'win_right', 'apps'
        ]
        self.mouse_actions = [
            'click', 'double_click', 'right_click', 'middle_click',
            'drag_short', 'drag_medium', 'drag_long',
            'scroll_up', 'scroll_down', 'scroll_left', 'scroll_right',
            'move_random', 'move_to_center', 'move_to_corner',
            'hover_short', 'hover_long',
        ]

        self.keyboard_actions = self.all_keys.copy()
        self.combo_actions = self._generate_random_combos()
        self.special_actions = [
            'wait_short', 'wait_long', 'release_all'
        ]
        self.actions = self.mouse_actions + self.keyboard_actions + \
                       self.combo_actions + self.special_actions

        # СТАТИСТИКА - ДОЛЖНА БЫТЬ ОПРЕДЕЛЕНА ДО ЗАГРУЗКИ
        self.stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'average_reward': 0,
            'last_action': 'none',
            'mouse_actions': 0,
            'keyboard_actions': 0,
            'combo_actions': 0
        }

        self.state_visits = defaultdict(int)
        self.novelty_decay = 0.999

        # ЗАГРУЗКА МОДЕЛИ ПОСЛЕ ИНИЦИАЛИЗАЦИИ ВСЕХ АТРИБУТОВ
        self.load_model()
        self.print_menu()

    def _generate_random_combos(self, n_combos: int = 100) -> List[str]:
        combos = []
        modifiers = ['ctrl', 'alt', 'shift', 'fn']
        for mod in modifiers:
            for key in random.sample(self.all_keys, min(15, len(self.all_keys))):
                if key not in modifiers and key != 'print_screen':
                    combo = f"{mod}+{key}"
                    if not SystemProtection.is_dangerous(combo):
                        combos.append(combo)
        for i in range(20):
            mod1, mod2 = random.sample(modifiers, 2)
            key = random.choice([k for k in self.all_keys if k not in modifiers and k != 'print_screen'])
            combo = f"{mod1}+{mod2}+{key}"
            if not SystemProtection.is_dangerous(combo):
                combos.append(combo)
        for i in range(30):
            keys = random.sample([k for k in self.all_keys if k not in modifiers and k != 'print_screen'], 2)
            combo = f"{keys[0]}+{keys[1]}"
            combos.append(combo)
        for i in range(10):
            keys = random.sample([k for k in self.all_keys if k != 'print_screen'], 3)
            combo = f"{keys[0]}+{keys[1]}+{keys[2]}"
            if not SystemProtection.is_dangerous(combo):
                combos.append(combo)
        return combos

    def print_menu(self):
        h, w = self.tokenizer.get_dimensions()
        print("\n" + "=" * 70)
        print("🌲 ГИБРИДНЫЙ ЛЕС + ПРЯМОУГОЛЬНАЯ ТОКЕНИЗАЦИЯ")
        print("=" * 70)
        print(f"🎯 Разрешение сетки: {w}×{h} = {h * w} пикселей")
        print(f"🌲 Деревьев: {self.forest.n_trees}")
        print(f"📊 Данных: {len(self.training_data)}")
        print(f" Статус: {'🟢 Обучен' if self.forest.is_trained else ' Не обучен'}")
        print(f"\n🖱️  Мышиных действий: {len(self.mouse_actions)}")
        print(f"⌨️  Одиночных клавиш: {len(self.keyboard_actions)}")
        print(f"🔗 Комбинаций клавиш: {len(self.combo_actions)}")
        print(f"⚡ Всего действий: {len(self.actions)}")
        print(f"🛡️  Защита: Активна")
        print("\n⌨️  УПРАВЛЕНИЕ:")
        print("  F1 - запись (вкл/выкл)")
        print("  F2 - авто-режим (вкл/выкл)")
        print("  F3 - сохранить модель")
        print("  F4 - очистить память")
        print("  F5 - показать статистику")
        print("  F6 - 📸 ПОКАЗАТЬ ЧТО ВИДИТ ИИ (скриншот + токены)")
        print("  F7 - показать важность признаков")
        print("  F8 - показать примеры комбинаций")
        print("  F9 - показать заблокированные действия")
        print("  ESC - остановка")
        print("=" * 70)

    def capture_screen(self):
        try:
            screenshot = self.sct.grab(GAME_REGION)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            return None

    def encode_state(self, frame: np.ndarray, keys: List[str] = None) -> List[float]:
        tokens = self.tokenizer.encode_frame(frame, keys)
        context = self.tokenizer.get_context()
        combined = tokens + context
        return [float(t) / 255.0 for t in combined[:300]]

    def calculate_adaptive_reward(self, before, after, action):
        base_reward = self.reward_system.calculate_smart_reward(before, after, action, self.current_state)
        if self.learning_phase == 'exploration':
            novelty_bonus = self._calculate_novelty_bonus(action)
            return base_reward + novelty_bonus * 0.5
        elif self.learning_phase == 'exploitation':
            repetition_penalty = self._calculate_repetition_penalty(action)
            return base_reward - repetition_penalty * 0.3
        elif self.learning_phase == 'fine_tuning':
            optimal_reward = self._calculate_optimal_reward(action)
            return base_reward * 0.7 + optimal_reward * 0.3
        return base_reward

    def _calculate_novelty_bonus(self, action):
        if len(self.reward_system.action_history) < 10:
            return 0.0
        action_list = list(self.reward_system.action_history)
        if len(action_list) > 10:
            action_list = action_list[len(action_list) - 10:]
        recent_actions = set(action_list)
        if action not in recent_actions:
            return 0.5
        return 0.0

    def _calculate_optimal_reward(self, action):
        success_rate = self.reward_system.action_success_rate.get(action, 0.0)
        if isinstance(success_rate, (int, float)):
            return success_rate * 0.5
        return 0.0

    def _calculate_repetition_penalty(self, action):
        if len(self.reward_system.action_history) < 5:
            return 0.0
        action_list = list(self.reward_system.action_history)
        if len(action_list) > 10:
            action_list = action_list[len(action_list) - 10:]
        repetition_count = action_list.count(action)
        if repetition_count > 2:
            return (repetition_count - 2) * 0.2
        return 0.0

    def update_learning_phase(self):
        total_actions = self.stats['total_actions']
        avg_reward = np.mean(list(self.reward_system.reward_history)) if self.reward_system.reward_history else 0
        if total_actions < self.exploration_threshold:
            self.learning_phase = 'exploration'
        elif total_actions < self.exploitation_threshold:
            if avg_reward > 0.5:
                self.learning_phase = 'exploitation'
            else:
                self.learning_phase = 'exploration'
        else:
            if avg_reward > 1.0:
                self.learning_phase = 'fine_tuning'
            else:
                self.learning_phase = 'exploitation'
        if self.phase_switches < 10:
            print(f"🔄 Фаза обучения: {self.learning_phase} (награда: {avg_reward:.2f})")
            self.phase_switches += 1

    def select_action_with_phase(self, state):
        if self.learning_phase == 'exploration':
            if random.random() < 0.7:
                return self._select_random_action(), 0.3
            else:
                return self._select_smart_action(state), 0.7
        elif self.learning_phase == 'exploitation':
            if random.random() < 0.8:
                return self._select_smart_action(state), 0.8
            else:
                return self._select_random_action(), 0.2
        elif self.learning_phase == 'fine_tuning':
            if random.random() < 0.95:
                return self._select_smart_action(state), 0.95
            else:
                return self._select_random_action(), 0.05

    def _select_random_action(self):
        action = random.choice(self.actions)
        while SystemProtection.is_dangerous(action):
            action = random.choice(self.actions)
        return action

    def _select_smart_action(self, state):
        if self.forest.is_trained:
            try:
                probas = self.forest.predict_proba(state)
                weighted_probas = []
                for i, prob in enumerate(probas):
                    action = self.actions[i]
                    success_rate = self.reward_system.action_success_rate.get(action, 0.0)
                    weighted_prob = prob * 0.7 + success_rate * 0.3
                    weighted_probas.append(weighted_prob)
                action_idx = np.argmax(weighted_probas)
                action = self.actions[action_idx]
                if SystemProtection.is_dangerous(action):
                    return self._select_random_action()
                return action
            except:
                return self._select_random_action()
        return self._select_random_action()

    def calculate_reward(self, before: List[float], after: List[float]) -> float:
        if before is None or after is None:
            return 0.0
        if len(before) == 0 or len(after) == 0:
            return 0.0
        if len(before) != len(after):
            return 0.0

        diff = 0.0
        for i in range(len(before)):
            diff += abs(float(before[i]) - float(after[i]))

        max_diff = len(before)
        change = diff / max_diff if max_diff > 0 else 0
        time_since_last = time.time() - self.last_action_time
        time_factor = min(1.0, 1.0 / (time_since_last + 0.1))

        if change > 0.03:
            sigmoid_input = (change - 0.05) * 20
            raw_reward = 1.0 / (1.0 + math.exp(-sigmoid_input))
            raw_reward = raw_reward * 2 - 1
            reward = raw_reward * 3.0
            if self.stats['successful_actions'] > 0:
                reward *= 1.2
            reward *= (0.5 + 0.5 * time_factor)
            self.stats['successful_actions'] += 1
        else:
            stagnation = max(0, 1.0 - change / 0.03)
            raw_penalty = 1.0 / (1.0 + math.exp(stagnation * 10 - 5))
            reward = -1.5 * raw_penalty * (1 + (1 - time_factor))

        if self.stats['total_actions'] > 5 and self.stats['average_reward'] < 0:
            reward += 0.3

        if len(before) >= 50:
            state_hash = hash(tuple(int(float(x) * 20) for x in before[:50]))
            visit_count = self.state_visits.get(state_hash, 0)
            novelty_factor = 1.0 / (1.0 + math.exp(visit_count / 5 - 3))
            novelty_bonus = 1.5 * novelty_factor
            reward += novelty_bonus
            self.state_visits[state_hash] = visit_count + 1
        else:
            reward += 0.5

        self.novelty_decay *= 0.9999
        if len(self.state_visits) > 10000:
            for key in list(self.state_visits.keys())[:1000]:
                del self.state_visits[key]

        reward = max(-3.0, min(3.0, reward))

        if self.forest.is_trained and len(self.current_state) > 0:
            tree_idx = random.randint(0, len(self.forest.trees) - 1) if self.forest.trees else 0
            try:
                if tree_idx < len(self.forest.trees):
                    _, path = self.forest._predict_with_path(self.forest.trees[tree_idx], self.current_state)
                    self.forest.update_weights_online(reward, path, tree_idx)
            except Exception as e:
                pass

        return reward

    def train(self, force: bool = False):
        if len(self.training_data) < 30:
            print(f"⚠️ Недостаточно данных для обучения ({len(self.training_data)}/30)")
            return
        if not force and len(self.training_data) % LEARNING_INTERVAL != 0:
            return
        try:
            self.forest.fit(self.training_data, self.training_labels,
                            update_existing=self.forest.is_trained)
            accuracy = self.evaluate_model()
            print(f"📊 Точность модели: {accuracy:.2%}")
        except Exception as e:
            print(f"⚠️ Ошибка обучения: {e}")

    def evaluate_model(self) -> float:
        if len(self.training_data) < 10 or not self.forest.is_trained:
            return 0.0
        test_size = min(20, len(self.training_data))
        indices = random.sample(range(len(self.training_data)), test_size)
        correct = 0
        for idx in indices:
            pred = self.forest.predict(self.training_data[idx])
            actual = self.training_labels[idx]
            if abs(pred - actual) < 0.5:
                correct += 1
        return correct / test_size if test_size > 0 else 0.0

    def select_action(self, state: List[float]) -> Tuple[str, float]:
        if self.forest.is_trained and random.random() < 0.85:
            try:
                probas = self.forest.predict_proba(state)
                action_idx = np.argmax(probas)
                confidence = probas[action_idx]
                if confidence < 0.3:
                    action_idx = random.randint(0, len(self.actions) - 1)
                    confidence = 0.2
                action = self.actions[action_idx]
                if SystemProtection.is_dangerous(action):
                    action_idx = random.randint(0, len(self.actions) - 1)
                    action = self.actions[action_idx]
                    confidence = 0.1
                return action, confidence
            except:
                pass
        action_idx = random.randint(0, len(self.actions) - 1)
        action = self.actions[action_idx]
        while SystemProtection.is_dangerous(action):
            action_idx = random.randint(0, len(self.actions) - 1)
            action = self.actions[action_idx]
        return action, 0.1

    def execute_action(self, action: str) -> bool:
        if SystemProtection.is_dangerous(action):
            self.blocked_actions.append(action)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {action}")
            return False
        if action == 'wait_short':
            time.sleep(0.5 + random.random() * 0.3)
            return True
        elif action == 'wait_long':
            time.sleep(2.0 + random.random() * 2.0)
            return True
        if action == 'release_all':
            self.keyboard.release_all()
            return True
        if action in self.mouse_actions:
            return self._execute_mouse_action(action)
        if '+' in action:
            combo = action.split('+')
            return self._execute_combo_action(combo)
        if action in self.keyboard_actions:
            return self._execute_keyboard_action(action)
        return False

    def _execute_mouse_action(self, action: str) -> bool:
        if self.stats['total_actions'] > 0 and self.stats['average_reward'] > 0:
            x = random.randint(200, 1700)
            y = random.randint(200, 800)
        else:
            x = random.randint(800, 1120)
            y = random.randint(400, 680)
        self.stats['mouse_actions'] += 1
        if action in ['click', 'double_click', 'right_click']:
            self.human_click(x, y, action)
            return True
        elif action in ['drag_short', 'drag_long']:
            distance = random.randint(50, 150) if 'short' in action else random.randint(200, 400)
            to_x = x + random.randint(-distance, distance)
            to_y = y + random.randint(-distance, distance)
            self.human_drag(x, y, to_x, to_y)
            return True
        elif action in ['scroll_up', 'scroll_down']:
            delta = -1 if 'up' in action else 1
            self.human_scroll(delta)
            return True
        return False

    def _execute_keyboard_action(self, action: str) -> bool:
        self.stats['keyboard_actions'] += 1
        if action == 'shift_tab':
            self.keyboard.press_combo(['shift', 'tab'], duration=0.1)
            return True
        self.keyboard.press_key(action, duration=random.uniform(0.05, 0.15))
        return True

    def _execute_combo_action(self, combo: List[str]) -> bool:
        self.stats['combo_actions'] += 1
        normalized_combo = []
        for key in combo:
            key = key.lower().strip()
            if key == 'win':
                key = 'cmd'
            normalized_combo.append(key)

        # Проверяем опасные комбинации
        combo_str = '+'.join(normalized_combo)
        if SystemProtection.is_dangerous(combo_str):
            self.blocked_actions.append(combo_str)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {combo_str}")
            return False

        self.keyboard.press_combo(normalized_combo, duration=random.uniform(0.1, 0.3))
        return True

    def human_click(self, x, y, click_type='click'):
        offset_x = random.randint(-8, 8)
        offset_y = random.randint(-8, 8)
        self.human_move(x + offset_x, y + offset_y)
        if click_type == 'right_click':
            self.mouse.press(Button.right)
            time.sleep(random.uniform(0.03, 0.08))
            self.mouse.release(Button.right)
            return
        clicks = {'click': 1, 'double_click': 2}.get(click_type, 1)
        for i in range(clicks):
            self.mouse.press(Button.left)
            time.sleep(random.uniform(0.03, 0.08))
            self.mouse.release(Button.left)
            if i < clicks - 1:
                time.sleep(random.uniform(0.05, 0.15))

    def human_drag(self, from_x, from_y, to_x, to_y):
        self.human_move(from_x, from_y)
        time.sleep(random.uniform(0.05, 0.1))
        self.mouse.press(Button.left)
        time.sleep(random.uniform(0.05, 0.1))
        steps = random.randint(10, 20)
        for i in range(steps):
            progress = i / steps
            deviation = math.sin(progress * math.pi) * random.randint(2, 10)
            x = from_x + (to_x - from_x) * progress + deviation
            y = from_y + (to_y - from_y) * progress + deviation * 0.2
            self.mouse.position = (int(x), int(y))
            time.sleep(random.uniform(0.005, 0.015))
        self.mouse.position = (to_x + random.randint(-3, 3),
                               to_y + random.randint(-3, 3))
        time.sleep(random.uniform(0.02, 0.05))
        self.mouse.release(Button.left)

    def human_move(self, target_x, target_y):
        start_x, start_y = self.mouse.position
        steps = random.randint(8, 25)
        for i in range(steps):
            progress = i / steps
            deviation = math.sin(progress * math.pi) * random.randint(3, 20)
            x = start_x + (target_x - start_x) * progress + deviation
            y = start_y + (target_y - start_y) * progress + deviation * 0.3
            self.mouse.position = (int(x), int(y))
            time.sleep(random.uniform(0.003, 0.025))
        self.mouse.position = (target_x + random.randint(-3, 3),
                               target_y + random.randint(-3, 3))

    def human_scroll(self, delta_y):
        for _ in range(random.randint(1, 3)):
            self.mouse.scroll(0, delta_y / abs(delta_y) if delta_y != 0 else 0)
            time.sleep(random.uniform(0.01, 0.03))

    def on_mouse_move(self, x, y):
        self.last_mouse_move = time.time()

    def on_mouse_click(self, x, y, button, pressed):
        if button != Button.left:
            return
        if pressed:
            self.mouse_is_pressed = True
            self.press_start_time = time.time()
            self.press_start_pos = (x, y)
        else:
            self.mouse_is_pressed = False
            if self.recording and self.current_state:
                action = 'click'
                if time.time() - self.press_start_time > 0.5:
                    action = 'drag_long'
                self.record_action(action)

    def show_ai_vision(self):
        h, w = self.tokenizer.get_dimensions()
        print("\n" + "=" * 80)
        print("📸 ЧТО ВИДИТ ИИ (ПРЯМОУГОЛЬНАЯ ТОКЕНИЗАЦИЯ)")
        print("=" * 80)
        frame = self.capture_screen()
        if frame is None:
            print("❌ Не удалось захватить экран")
            return
        tokens = self.tokenizer.encode_frame(frame, [])
        grid = np.array(tokens[:self.tokenizer.total_cells])
        grid = grid.reshape(self.tokenizer.grid_height, self.tokenizer.grid_width)
        print(f"\n📐 Размер сетки: {w}×{h} = {self.tokenizer.total_cells} пикселей")
        print(f"📊 Диапазон значений: {grid.min():.0f} - {grid.max():.0f}")
        print(f"📈 Средняя яркость: {grid.mean():.1f}")
        print(f"📉 Стандартное отклонение: {grid.std():.1f}")
        print("\n🔢 МАТРИЦА ТОКЕНОВ (значения яркости):")
        print("   " + " ".join(f"{i:3d}" for i in range(min(16, self.tokenizer.grid_width))))
        print("   " + "-" * (4 * min(16, self.tokenizer.grid_width)))
        for y in range(self.tokenizer.grid_height):
            row_str = f"{y:2d}| "
            for x in range(min(16, self.tokenizer.grid_width)):
                val = grid[y, x]
                if val < 60:
                    char = "█"
                elif val < 120:
                    char = "▓"
                elif val < 180:
                    char = "▒"
                elif val < 240:
                    char = "░"
                else:
                    char = " "
                row_str += f"{char} "
            print(row_str)
        print("\n📊 ПЕРВЫЕ 50 ПРИЗНАКОВ (нормализованные):")
        if self.current_state:
            state_preview = self.current_state[:50]
            print("  " + " ".join(f"{v:6.3f}" for v in state_preview[:20]))
            if len(state_preview) > 20:
                print("  ... (еще {} значений)".format(len(state_preview) - 20))
        if self.forest.is_trained and self.current_state:
            try:
                pred = self.forest.predict(self.current_state)
                probas = self.forest.predict_proba(self.current_state)
                print(f"\n🤖 ПРОГНОЗ МОДЕЛИ:")
                print(f"  Предсказание: {pred:.3f}")
                print(f"  Уверенность: {max(probas):.1%}")
                best_action_idx = np.argmax(probas)
                if best_action_idx < len(self.actions):
                    print(f"  Рекомендуемое действие: {self.actions[best_action_idx]}")
                print(f"\n📊 ТОП-5 ДЕЙСТВИЙ ПО ВЕРОЯТНОСТИ:")
                top_indices = np.argsort(probas)[-5:][::-1]
                for idx in top_indices:
                    if idx < len(self.actions):
                        print(f"  {self.actions[idx]}: {probas[idx]:.1%}")
            except Exception as e:
                print(f"⚠️ Ошибка прогноза: {e}")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"ai_vision_{timestamp}_original.png", frame)
        vis_grid = self.tokenizer.visualize_grid(tokens)
        cv2.imwrite(f"ai_vision_{timestamp}_tokens.png", vis_grid)
        frame_h, frame_w = frame.shape[:2]
        vis_h = self.tokenizer.grid_height * 30
        vis_w = self.tokenizer.grid_width * 30
        combined = np.zeros((max(frame_h, vis_h), frame_w + vis_w + 20, 3), dtype=np.uint8)
        combined[:frame_h, :frame_w] = frame
        token_display = cv2.cvtColor(vis_grid, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(token_display, (0, 0), (vis_w, vis_h), (0, 255, 0), 2)
        combined[:vis_h, frame_w + 10:frame_w + 10 + vis_w] = token_display
        cv2.putText(combined, "Original Screen", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(combined, "AI Vision (Tokenized)", (frame_w + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0),
                    2)
        cv2.putText(combined, f"Grid: {w}×{h} = {self.tokenizer.total_cells}px", (frame_w + 20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.imwrite(f"ai_vision_{timestamp}_combined.png", combined)
        print(f"\n💾 Сохранено:")
        print(f"  - {timestamp}_original.png (оригинальный скриншот)")
        print(f"  - {timestamp}_tokens.png (токенизированная сетка)")
        print(f"  - {timestamp}_combined.png (комбинированное изображение)")
        print("=" * 80)

    def on_key_press(self, key):
        try:
            if key == Key.f1:
                self.recording = not self.recording
                print(f"\n📝 Запись: {'🟢 ВКЛ' if self.recording else '🔴 ВЫКЛ'}")
                self.bot_state = 'recording' if self.recording else 'idle'
            elif key == Key.f2:
                self.auto_mode = not self.auto_mode
                print(f"\n🤖 Авто-режим: {'🟢 ВКЛ' if self.auto_mode else '🔴 ВЫКЛ'}")
                self.bot_state = 'auto' if self.auto_mode else 'idle'
            elif key == Key.f3:
                self.save_model()
            elif key == Key.f4:
                self.clear_memory()
            elif key == Key.f5:
                self.show_stats()
            elif key == Key.f6:
                self.show_ai_vision()
            elif key == Key.f7:
                self.show_feature_importance()
            elif key == Key.f8:
                self.show_combos()
            elif key == Key.f9:
                self.show_blocked_stats()
            elif hasattr(key, 'char') and key.char:
                if self.recording:
                    self.record_action(f'key_{key.char}')
        except Exception as e:
            print(f"⚠️ Ошибка обработки клавиши: {e}")
        return True

    def record_action(self, action: str):
        if not self.current_state:
            return
        try:
            action_idx = self.actions.index(action) if action in self.actions else 0
            self.training_data.append(self.current_state.copy())
            self.training_labels.append(action_idx)
            self.recent_actions.append(action)
            if len(self.training_data) > RECORDING_BUFFER_SIZE:
                self.training_data = self.training_data[-RECORDING_BUFFER_SIZE:]
                self.training_labels = self.training_labels[-RECORDING_BUFFER_SIZE:]
            if len(self.training_data) % 10 == 0:
                self.train()
        except Exception as e:
            print(f"⚠️ Ошибка записи: {e}")

    def show_stats(self):
        h, w = self.tokenizer.get_dimensions()
        print("\n📊 СТАТИСТИКА:")
        print(f"  Размер сетки: {w}×{h} = {h * w} пикселей")
        print(f"  Данных для обучения: {len(self.training_data)}")
        print(f"  Лес обучен: {self.forest.is_trained}")
        print(f"  Деревьев: {len(self.forest.trees)}")
        print(f"  Всего действий: {self.stats['total_actions']}")
        print(f"  Успешных действий: {self.stats['successful_actions']}")
        print(f"  Мышиных действий: {self.stats['mouse_actions']}")
        print(f"  Клавиатурных действий: {self.stats['keyboard_actions']}")
        print(f"  Комбинаций клавиш: {self.stats['combo_actions']}")
        print(f"  Заблокировано: {len(self.blocked_actions)}")
        if self.stats['total_actions'] > 0:
            print(f"  Успешность: {self.stats['successful_actions'] / self.stats['total_actions']:.1%}")
        print(f"  Средняя награда: {self.stats['average_reward']:.2f}")
        print(f"  Последнее действие: {self.stats['last_action']}")
        print(f"  Состояние бота: {self.bot_state}")
        print(f"  Уникальных состояний: {len(self.state_visits)}")
        if self.forest.is_trained:
            accuracy = self.evaluate_model()
            print(f"  Точность модели: {accuracy:.1%}")
            branch_stats = self.forest.get_branch_statistics()
            print(f"\n⚖️  СТАТИСТИКА ВЕТВЕЙ:")
            print(f"  Активных весов: {branch_stats['total_weights']}")
            print(f"  Средний вес: {branch_stats['average_weight']:.3f}")
            print(f"  Мин/Макс вес: {branch_stats['min_weight']:.3f}/{branch_stats['max_weight']:.3f}")
            print(f"  Глобальные веса: {branch_stats['global_weights']}")
        reward_stats = self.reward_system.get_statistics()
        print(f"\n🎯 СТАТИСТИКА НАГРАД:")
        print(f"  Средняя награда: {reward_stats['avg_reward']:.2f}")
        print(f"  Стандартное отклонение: {reward_stats['std_reward']:.2f}")
        print(f"  Последовательных провалов: {reward_stats['consecutive_failures']}")
        print(f"  Разнообразие действий: {reward_stats['action_diversity']}/20")
        print(f"  Лучшее действие: {reward_stats['best_action']} ({reward_stats['best_action_rate']:.1%})")
        print(f"  Текущая фаза: {self.learning_phase}")

    def show_combos(self):
        print("\n🔗 ПРИМЕРЫ КОМБИНАЦИЙ КЛАВИШ:")
        if not self.combo_actions:
            print("  Нет комбинаций")
            return
        for i, combo in enumerate(self.combo_actions[:20], 1):
            print(f"  {i}. {combo}")
        if len(self.combo_actions) > 20:
            print(f"  ... и еще {len(self.combo_actions) - 20} комбинаций")

    def show_blocked_stats(self):
        if not self.blocked_actions:
            print("\n✅ Опасные действия не блокировались")
            return
        print("\n🛡️ ЗАБЛОКИРОВАННЫЕ ДЕЙСТВИЯ:")
        blocked_count = defaultdict(int)
        for action in self.blocked_actions:
            blocked_count[action] += 1
        for action, count in blocked_count.items():
            print(f"  {action}: {count} раз(а)")

    def show_feature_importance(self):
        if not self.forest.feature_importance:
            print("\n❌ Нет данных о важности признаков")
            return
        print("\n📊 ВАЖНОСТЬ ПРИЗНАКОВ:")
        sorted_features = sorted(self.forest.feature_importance.items(),
                                 key=lambda x: x[1], reverse=True)[:20]
        for idx, (feature, importance) in enumerate(sorted_features, 1):
            print(f"  {idx}. Признак {feature}: {importance:.4f}")

    def save_model(self):
        data = {
            'training_data': self.training_data,
            'training_labels': self.training_labels,
            'forest': self.forest,
            'stats': self.stats,
            'actions': self.actions,
            'combo_actions': self.combo_actions,
            'blocked_actions': self.blocked_actions,
            'timestamp': time.time()
        }
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Модель сохранена (данных: {len(self.training_data)})")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

    def load_model(self):
        if os.path.exists(self.model_file):
            try:
                with open(self.model_file, 'rb') as f:
                    data = pickle.load(f)

                    # Загружаем данные обучения
                    self.training_data = data.get('training_data', [])
                    self.training_labels = data.get('training_labels', [])

                    # Загружаем лес, если он есть
                    if 'forest' in data:
                        self.forest = data['forest']

                    # Обновляем stats, не заменяя весь объект
                    if 'stats' in data:
                        for key, value in data['stats'].items():
                            self.stats[key] = value

                    # Загружаем комбинации
                    if 'combo_actions' in data:
                        self.combo_actions = data['combo_actions']

                    # Загружаем заблокированные действия
                    if 'blocked_actions' in data:
                        self.blocked_actions = data['blocked_actions']

                print(f"📂 Модель загружена (данных: {len(self.training_data)})")
            except Exception as e:
                print(f"⚠️ Ошибка загрузки модели: {e}")

    def clear_memory(self):
        self.training_data = []
        self.training_labels = []
        self.forest = HybridDecisionForest()
        self.recent_actions.clear()
        self.recent_rewards.clear()
        self.stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'average_reward': 0,
            'last_action': 'none',
            'mouse_actions': 0,
            'keyboard_actions': 0,
            'combo_actions': 0
        }
        self.state_visits.clear()
        self.novelty_decay = 0.99999
        self.blocked_actions = []
        self.reward_system = SmartRewardSystem()
        self.learning_phase = 'exploration'
        self.phase_switches = 0
        print("🧹 Память очищена")

    def start_listeners(self):
        mouse_listener = MouseListener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click
        )
        mouse_listener.daemon = True
        mouse_listener.start()
        keyboard_listener = KeyboardListener(on_press=self.on_key_press)
        keyboard_listener.daemon = True
        keyboard_listener.start()
        print("✅ Слушатели запущены")

    def run(self):
        self.running = True
        self.last_mouse_move = time.time()
        self.last_action_time = time.time()
        self.start_listeners()
        self.print_menu()
        while self.running:
            try:
                frame = self.capture_screen()
                if frame is None:
                    time.sleep(0.1)
                    continue
                self.current_state = self.encode_state(frame, [])
                if self.auto_mode:
                    if time.time() - self.last_mouse_move > MOUSE_IDLE_THRESHOLD:
                        self.update_learning_phase()
                        action, confidence = self.select_action_with_phase(self.current_state)
                        executed = self.execute_action(action)
                        self.stats['total_actions'] += 1
                        self.stats['last_action'] = action
                        if executed:
                            time.sleep(0.2)
                            new_frame = self.capture_screen()
                            new_state = self.encode_state(new_frame, [])
                            reward = self.calculate_adaptive_reward(
                                self.current_state, new_state, action
                            )
                            self.recent_rewards.append(reward)
                            self.stats['average_reward'] = np.mean(
                                list(self.recent_rewards)) if self.recent_rewards else 0
                            if self.recording:
                                self.training_data.append(self.current_state.copy())
                                self.training_labels.append(self.actions.index(action))
                                self.recent_actions.append(action)
                                if len(self.training_data) % LEARNING_INTERVAL == 0:
                                    self.train()
                            feedback = self.reward_system.get_action_feedback(action)
                            stats = self.reward_system.get_statistics()
                            print(
                                f"🎮 {action:15} | "
                                f"Награда: {reward:6.2f} | "
                                f"Уверенность: {confidence:.2f} | "
                                f"Фаза: {self.learning_phase[:3]} | "
                                f"{feedback}"
                            )
                            if self.stats['total_actions'] % 10 == 0:
                                print(f"📊 Средняя награда: {stats['avg_reward']:.2f} | "
                                      f"Разнообразие: {stats['action_diversity']}/20 | "
                                      f"Лучшее действие: {stats['best_action']} ({stats['best_action_rate']:.1%})")
                            self.current_state = new_state
                            self.last_action_time = time.time()
                time.sleep(0.05)
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"⚠️ Ошибка: {e}")
                time.sleep(0.5)
        self.save_model()
        print("\n🛑 Бот остановлен")
        print(f"📊 Итоговая статистика: {self.stats}")


def signal_handler(sig, frame):
    _ = sig, frame
    print("\n🛑 Остановка")
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    bot = HybridForestBot()
    bot.run()