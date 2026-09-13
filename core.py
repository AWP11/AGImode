# core.py
# ============================================
# ЯДРО СИСТЕМЫ - ВСЯ БАЗОВАЯ ЛОГИКА (v3)
# С поддержкой параметров действий и самоподкрепления
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
import threading
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import List, Tuple, Dict, Optional, Any, Set
from pynput.mouse import Button, Controller as MouseController, Listener as MouseListener
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyboardListener
from datetime import datetime

SCREEN_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
MOUSE_IDLE_THRESHOLD = 1.5
LEARNING_INTERVAL = 10
RECORDING_BUFFER_SIZE = 2000


# ============================================
# SIMPLE RESONANCE
# ============================================
class SimpleResonance:
    def __init__(self, n_branches: int, alpha: float = 0.3,
                 lr: float = 0.01, max_iter: int = 3, tol: float = 1e-3,
                 decay: float = 0.999):
        self.n = max(1, n_branches)
        self.alpha = alpha
        self.lr = lr
        self.max_iter = max_iter
        self.tol = tol
        self.decay = decay
        self.W = np.zeros((self.n, self.n), dtype=np.float32)
        self.reward_baseline = 0.0
        self._lock = threading.RLock()

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        state.pop('_rng', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(42)

    def settle(self, a: np.ndarray) -> np.ndarray:
        a = a.astype(np.float32).copy()
        if a.size != self.n:
            if a.size < self.n:
                a = np.concatenate([a, np.zeros(self.n - a.size, dtype=np.float32)])
            else:
                a = a[:self.n]
        for _ in range(self.max_iter):
            lateral = self.W @ a
            a_new = np.tanh(a + self.alpha * lateral)
            if np.linalg.norm(a_new - a) < self.tol:
                break
            a = a_new
        return a

    def update(self, activations: np.ndarray, reward: float):
        with self._lock:
            if activations.size != self.n:
                if activations.size < self.n:
                    activations = np.concatenate(
                        [activations, np.zeros(self.n - activations.size, dtype=np.float32)]
                    )
                else:
                    activations = activations[:self.n]

            self.reward_baseline = 0.99 * self.reward_baseline + 0.01 * reward
            dopamine = reward - self.reward_baseline

            outer = np.outer(activations, activations)
            self.W += self.lr * dopamine * outer
            np.clip(self.W, -1.0, 1.0, out=self.W)
            np.fill_diagonal(self.W, 0.0)
            self.W *= self.decay

    def get_statistics(self) -> Dict:
        with self._lock:
            if self.W.size == 0 or self.n < 2:
                return {'mean': 0.0, 'std': 0.0, 'positive': 0,
                        'negative': 0, 'baseline': self.reward_baseline}
            mask = ~np.eye(self.n, dtype=bool)
            off_diag = self.W[mask]
            return {
                'mean': float(off_diag.mean()),
                'std': float(off_diag.std()),
                'positive': int((off_diag > 0.01).sum()),
                'negative': int((off_diag < -0.01).sum()),
                'baseline': self.reward_baseline,
            }

    def reset(self):
        with self._lock:
            self.W = np.zeros((self.n, self.n), dtype=np.float32)
            self.reward_baseline = 0.0


# ============================================
# WEIGHTED SLIDING MEMORY (с параметрами)
# ============================================
class WeightedSlidingMemory:
    def __init__(
        self,
        max_size: int = 2000,
        decay: float = 0.997,
        soft_threshold: float = 0.15,
        hard_threshold: float = 0.02,
        min_lifetime: int = 200,
        w_reward: float = 1.0,
        w_branch: float = 0.8,
        w_novelty: float = 0.6,
        novelty_scale: float = 50.0,
        base_strength: float = 0.5,
        min_strength: float = 0.2,
        max_strength: float = 5.0,
        rebalance_interval: int = 200,
    ):
        self.records: List[Dict[str, Any]] = []
        self.max_size = max_size
        self.decay = decay
        self.soft_threshold = soft_threshold
        self.hard_threshold = hard_threshold
        self.min_lifetime = min_lifetime
        self.w_reward = w_reward
        self.w_branch = w_branch
        self.w_novelty = w_novelty
        self.novelty_scale = novelty_scale
        self.base_strength = base_strength
        self.min_strength = min_strength
        self.max_strength = max_strength
        self.rebalance_interval = rebalance_interval

        self.step_counter = 0
        self.state_visits: Dict[int, int] = defaultdict(int)
        self._rng = np.random.default_rng(42)
        self._lock = threading.RLock()

        self._last_prune_removed = 0

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        state.pop('_rng', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()
        self._rng = np.random.default_rng(42)

    def _hash_state(self, state) -> int:
        if state is None:
            return 0
        try:
            arr = np.asarray(state, dtype=np.float32)
            quant = np.round(arr * 8.0).astype(np.int16)
            return hash(quant.tobytes())
        except Exception:
            return hash(str(state)[:200])

    def add(self, state, label, reward: float = 0.0,
            branch_weight: float = 0.0, params: Optional[Dict] = None):
        with self._lock:
            h = self._hash_state(state)
            self.state_visits[h] += 1
            visits = self.state_visits[h]

            novelty = math.exp(-visits / self.novelty_scale)

            S0 = (
                self.w_reward * abs(reward)
                + self.w_branch * max(0.0, branch_weight)
                + self.w_novelty * novelty
                + self.base_strength
            )
            S0 = float(np.clip(S0, self.min_strength, self.max_strength))

            self.records.append({
                'state': state,
                'label': label,
                'reward': float(reward),
                'branch_weight': float(branch_weight),
                'strength': S0,
                'birth_step': self.step_counter,
                'last_used': self.step_counter,
                'hash': h,
                'params': params or {},
                'sub_reward': 0.0,
            })

            if len(self.records) > self.max_size:
                self.records.sort(key=lambda r: r['strength'], reverse=True)
                self.records = self.records[:self.max_size]

    def step(self):
        with self._lock:
            self.step_counter += 1

            for r in self.records:
                r['strength'] *= self.decay

            before = len(self.records)
            self.records = [
                r for r in self.records
                if not (
                    r['strength'] < self.hard_threshold
                    and (self.step_counter - r['birth_step']) > self.min_lifetime
                )
            ]
            self._last_prune_removed = before - len(self.records)

            if self.step_counter % self.rebalance_interval == 0:
                self._rebalance()

    def _rebalance(self):
        if not self.records:
            return
        strengths = np.array([r['strength'] for r in self.records], dtype=np.float64)
        mean = strengths.mean()
        if mean < 1e-6:
            return
        scale = float(np.clip(1.0 / mean, 0.5, 2.0))
        for r in self.records:
            r['strength'] = float(np.clip(
                r['strength'] * scale, self.min_strength, self.max_strength
            ))

    def reinforce(self, state_hash: int, reward: float,
                  branch_weight: float = 0.0, sub_reward: float = 0.0):
        with self._lock:
            boost = (
                self.w_reward * abs(reward)
                + self.w_branch * max(0.0, branch_weight)
            )
            for r in self.records:
                if r['hash'] == state_hash:
                    r['strength'] = float(np.clip(
                        r['strength'] + boost, self.min_strength, self.max_strength
                    ))
                    r['last_used'] = self.step_counter
                    r['sub_reward'] = float(np.clip(
                        r.get('sub_reward', 0.0) + sub_reward, -1.0, 1.0
                    ))

    def _weighted_indices(self, candidates: List[Dict], n: int) -> np.ndarray:
        m = len(candidates)
        if n >= m:
            return np.arange(m)

        weights = np.array([c['strength'] for c in candidates], dtype=np.float64)
        weights = np.maximum(weights, 1e-9)
        cdf = np.cumsum(weights)
        total = cdf[-1]
        if total <= 0 or not np.isfinite(total):
            return self._rng.choice(m, size=n, replace=False)

        r = self._rng.random(n) * total
        idx = np.searchsorted(cdf, r, side='right')
        idx = np.clip(idx, 0, m - 1)

        if len(np.unique(idx)) < n:
            remaining = n - len(np.unique(idx))
            extra = self._rng.choice(m, size=remaining, replace=False)
            idx = np.concatenate([idx, extra])
        return idx[:n]

    def sample(self, n: int):
        with self._lock:
            if not self.records:
                return [], []

            candidates = [r for r in self.records if r['strength'] >= self.soft_threshold]
            if len(candidates) < max(10, n // 2):
                candidates = self.records

            n = min(n, len(candidates))
            if n <= 0:
                return [], []

            idx = self._weighted_indices(candidates, n)
            X = [candidates[i]['state'] for i in idx]
            y = [candidates[i]['label'] for i in idx]
            return X, y

    def get_all(self):
        with self._lock:
            alive = [r for r in self.records if r['strength'] >= self.soft_threshold]
            if not alive:
                alive = self.records
            return [r['state'] for r in alive], [r['label'] for r in alive]

    def get_param_samples(self, label: int, action_name: str) -> List[Dict]:
        with self._lock:
            return [
                r for r in self.records
                if r['label'] == label and r.get('params')
            ]

    def update_param_sub_reward(self, state_hash: int, label: int, sub_reward: float):
        with self._lock:
            for r in self.records:
                if r['hash'] == state_hash and r['label'] == label:
                    r['sub_reward'] = float(np.clip(
                        r.get('sub_reward', 0.0) + sub_reward, -1.0, 1.0
                    ))

    def __len__(self):
        with self._lock:
            return len(self.records)

    def clear(self):
        with self._lock:
            self.records.clear()
            self.state_visits.clear()
            self.step_counter = 0

    def get_statistics(self) -> Dict:
        with self._lock:
            if not self.records:
                return {
                    'size': 0, 'max_size': self.max_size,
                    'avg_strength': 0.0, 'min_strength': 0.0,
                    'max_strength': 0.0, 'weak': 0, 'strong': 0,
                    'fill_ratio': 0.0, 'unique_states': 0,
                    'last_pruned': self._last_prune_removed,
                    'with_params': 0,
                }
            s = np.array([r['strength'] for r in self.records], dtype=np.float64)
            with_params = sum(1 for r in self.records if r.get('params'))
            return {
                'size': len(self.records),
                'max_size': self.max_size,
                'avg_strength': float(s.mean()),
                'min_strength': float(s.min()),
                'max_strength': float(s.max()),
                'weak': int((s < self.soft_threshold).sum()),
                'strong': int((s >= self.soft_threshold * 2).sum()),
                'fill_ratio': len(self.records) / self.max_size,
                'unique_states': len(self.state_visits),
                'last_pruned': self._last_prune_removed,
                'with_params': with_params,
            }

    def is_full(self) -> bool:
        with self._lock:
            return len(self.records) >= self.max_size


# ============================================
# SLIDING MEMORY (legacy, с параметрами)
# ============================================
class SlidingMemory:
    def __init__(self, max_size: int = 500):
        self.max_size = max_size
        self.X = deque(maxlen=max_size)
        self.y = deque(maxlen=max_size)
        self.params = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self._lock = threading.RLock()

    def add(self, state, label, reward: float = 0.0,
            branch_weight: float = 0.0, params: Optional[Dict] = None):
        with self._lock:
            self.X.append(state)
            self.y.append(label)
            self.params.append(params or {})
            self.timestamps.append(time.time())

    def add_batch(self, states, labels):
        with self._lock:
            for s, l in zip(states, labels):
                self.X.append(s)
                self.y.append(l)
                self.params.append({})
                self.timestamps.append(time.time())

    def get_all(self):
        with self._lock:
            return list(self.X), list(self.y)

    def sample(self, n: int):
        with self._lock:
            if len(self.X) == 0:
                return [], []
            if len(self.X) <= n:
                return list(self.X), list(self.y)
            indices = random.sample(range(len(self.X)), n)
            return [self.X[i] for i in indices], [self.y[i] for i in indices]

    def __len__(self):
        with self._lock:
            return len(self.X)

    def clear(self):
        with self._lock:
            self.X.clear()
            self.y.clear()
            self.params.clear()
            self.timestamps.clear()

    def get_statistics(self) -> Dict:
        with self._lock:
            if not self.timestamps:
                return {
                    'size': 0, 'max_size': self.max_size,
                    'oldest_age': 0.0, 'newest_age': 0.0, 'fill_ratio': 0.0,
                    'avg_strength': 0.0, 'weak': 0, 'strong': 0,
                }
            now = time.time()
            return {
                'size': len(self.X),
                'max_size': self.max_size,
                'oldest_age': now - self.timestamps[0],
                'newest_age': now - self.timestamps[-1],
                'fill_ratio': len(self.X) / self.max_size,
                'avg_strength': 0.0, 'weak': 0, 'strong': 0,
            }

    def is_full(self) -> bool:
        with self._lock:
            return len(self.X) >= self.max_size


# ============================================
# PARAMETER MEMORY — сердце самоподкрепления параметров
# ============================================
class ParameterMemory:
    """
    Хранит параметры действий: (x, y, duration, distance, direction).
    Для каждого (state_hash, action_idx) хранит взвешенное среднее параметров.
    Sub-reward обновляет вес записи — удачные параметры усиливаются.
    """

    def __init__(self, n_actions: int, decay: float = 0.995):
        self.n_actions = max(1, n_actions)
        self.decay = decay
        self._lock = threading.RLock()

        # (state_hash, action_idx) -> {param_name: [sum_weighted, sum_weights]}
        self.param_stats: Dict[Tuple[int, int], Dict[str, List[float]]] = {}
        # (state_hash, action_idx) -> общий sub_reward
        self.sub_rewards: Dict[Tuple[int, int], float] = defaultdict(float)
        # сколько раз видели пару
        self.visit_counts: Dict[Tuple[int, int], int] = defaultdict(int)

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def add(self, state_hash: int, action_idx: int, params: Dict[str, float],
            sub_reward: float = 0.0, weight: float = 1.0):
        with self._lock:
            key = (int(state_hash), int(action_idx))
            if key not in self.param_stats:
                self.param_stats[key] = {}

            for name, value in params.items():
                if name not in self.param_stats[key]:
                    self.param_stats[key][name] = [0.0, 0.0]
                self.param_stats[key][name][0] += float(value) * weight
                self.param_stats[key][name][1] += weight

            self.sub_rewards[key] = float(np.clip(
                self.sub_rewards.get(key, 0.0) * 0.9 + sub_reward * 0.1, -1.0, 1.0
            ))
            self.visit_counts[key] += 1

    def reinforce(self, state_hash: int, action_idx: int,
                  sub_reward: float, weight: float = 1.0):
        """Усилить/ослабить запись по (state_hash, action_idx)."""
        with self._lock:
            key = (int(state_hash), int(action_idx))
            if key not in self.param_stats:
                return
            self.sub_rewards[key] = float(np.clip(
                self.sub_rewards[key] + sub_reward * weight, -1.0, 1.0
            ))

    def predict(self, state_hash: int, action_idx: int,
                fallback: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Предсказывает параметры для (state_hash, action_idx).
        Если точного совпадения нет — усредняет по всем записям с этим action_idx,
        взвешивая по sub_reward.
        """
        with self._lock:
            key = (int(state_hash), int(action_idx))

            # Точное совпадение
            if key in self.param_stats and self.param_stats[key]:
                return self._extract_mean(self.param_stats[key])

            # Усреднение по action_idx с учётом sub_reward
            candidates = [
                (k, stats) for k, stats in self.param_stats.items()
                if k[1] == action_idx
            ]
            if not candidates:
                return fallback or {}

            # Взвешенное среднее: вес = 1 + sub_reward
            combined: Dict[str, List[float]] = {}
            for k, stats in candidates:
                w = max(0.1, 1.0 + self.sub_rewards.get(k, 0.0))
                for name, (s, sw) in stats.items():
                    if name not in combined:
                        combined[name] = [0.0, 0.0]
                    combined[name][0] += s * w
                    combined[name][1] += sw * w

            return self._extract_mean(combined)

    def _extract_mean(self, stats: Dict[str, List[float]]) -> Dict[str, float]:
        result = {}
        for name, (s, sw) in stats.items():
            if sw > 0:
                result[name] = s / sw
        return result

    def get_sub_reward(self, state_hash: int, action_idx: int) -> float:
        with self._lock:
            return self.sub_rewards.get((int(state_hash), int(action_idx)), 0.0)

    def step(self):
        with self._lock:
            for key in list(self.sub_rewards.keys()):
                self.sub_rewards[key] *= self.decay
                if abs(self.sub_rewards[key]) < 1e-4:
                    del self.sub_rewards[key]

    def __len__(self):
        with self._lock:
            return len(self.param_stats)

    def clear(self):
        with self._lock:
            self.param_stats.clear()
            self.sub_rewards.clear()
            self.visit_counts.clear()

    def get_statistics(self) -> Dict:
        with self._lock:
            if not self.param_stats:
                return {
                    'size': 0, 'avg_sub_reward': 0.0,
                    'positive': 0, 'negative': 0, 'unique_actions': 0,
                }
            rewards = list(self.sub_rewards.values())
            actions = set(k[1] for k in self.param_stats.keys())
            return {
                'size': len(self.param_stats),
                'avg_sub_reward': float(np.mean(rewards)) if rewards else 0.0,
                'positive': sum(1 for r in rewards if r > 0.05),
                'negative': sum(1 for r in rewards if r < -0.05),
                'unique_actions': len(actions),
            }


# ============================================
# SCAN CODE REGISTRY
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
        'space': 0x39, 'enter': 0x1C, 'escape': 0x01,
        'tab': 0x0F, 'backspace': 0x0E,
        'shift': 0x2A, 'ctrl': 0x1D, 'alt': 0x38,
        'capslock': 0x3A, 'numlock': 0x45, 'scrolllock': 0x46,
        'pause': 0xC5, 'insert': 0xD2, 'delete': 0xD3,
        'home': 0xC7, 'end': 0xCF,
        'page_up': 0xC9, 'page_down': 0xD1,
        'print_screen': 0xE0 + 0x37, 'sysrq': 0xE0 + 0x37,
        'up': 0xC8, 'down': 0xD0, 'left': 0xCB, 'right': 0xCD,
        'tilde': 0x29, 'minus': 0x0C, 'equals': 0x0D,
        'bracket_left': 0x1A, 'bracket_right': 0x1B,
        'backslash': 0x2B, 'semicolon': 0x27, 'apostrophe': 0x28,
        'comma': 0x33, 'period': 0x34, 'slash': 0x35,
        'numpad_0': 0x52, 'numpad_1': 0x4F, 'numpad_2': 0x50,
        'numpad_3': 0x51, 'numpad_4': 0x4B, 'numpad_5': 0x4C,
        'numpad_6': 0x4D, 'numpad_7': 0x47, 'numpad_8': 0x48,
        'numpad_9': 0x49, 'numpad_decimal': 0x53,
        'numpad_divide': 0xB5, 'numpad_multiply': 0x37,
        'numpad_subtract': 0x4A, 'numpad_add': 0x4E,
        'numpad_enter': 0x9C,
        'win_left': 0x5B, 'win_right': 0x5C, 'apps': 0x5D,
    }

    CODE_TO_KEY = {v: k for k, v in KEY_CODES.items()}

    @classmethod
    def get_code(cls, key_name: str) -> int:
        return cls.KEY_CODES.get(key_name.lower(), 0)

    @classmethod
    def get_key_name(cls, code: int) -> str:
        return cls.CODE_TO_KEY.get(code, 'unknown')


# ============================================
# SYSTEM PROTECTION
# ============================================
class SystemProtection:
    DANGEROUS_COMBOS = {
        'ctrl+alt+del': True, 'ctrl+alt+delete': True,
        'alt+sysrq+b': True, 'alt+sysrq+o': True, 'alt+sysrq+s': True,
        'alt+sysrq+u': True, 'alt+sysrq+e': True, 'alt+sysrq+i': True,
        'alt+sysrq+r': True, 'alt+sysrq+k': True,
        'alt+print_screen': True, 'alt+sysrq': True, 'alt+printscreen': True,
        'ctrl+alt+f1': True, 'ctrl+alt+f2': True, 'ctrl+alt+f3': True,
        'ctrl+alt+f4': True, 'ctrl+alt+f5': True, 'ctrl+alt+f6': True,
        'ctrl+alt+f7': True, 'ctrl+alt+f8': True, 'ctrl+alt+f9': True,
        'ctrl+alt+f10': True, 'ctrl+alt+f11': True, 'ctrl+alt+f12': True,
        'ctrl+alt+d': True, 'ctrl+alt+l': True,
        'alt+f4': True, 'win+l': True, 'cmd+l': True,
        'ctrl+alt+esc': True, 'ctrl+alt+backspace': True,
    }

    @classmethod
    def is_dangerous(cls, action: str) -> bool:
        action_lower = action.lower().strip()
        if action_lower in cls.DANGEROUS_COMBOS:
            return True
        if ('sysrq' in action_lower or 'print_screen' in action_lower
                or 'printscreen' in action_lower):
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
# KEYBOARD EMULATOR
# ============================================
class KeyboardEmulator:
    def __init__(self, activity_tensor=None):
        self.keyboard = KeyboardController()
        self.pressed_keys = set()
        self.activity_tensor = activity_tensor
        self._lock = threading.RLock()

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
            except Exception:
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
            'num_lock': Key.num_lock, 'caps_lock': Key.caps_lock,
        }
        if key_name.lower() in special:
            return special[key_name.lower()]
        if len(key_name) == 1:
            return key_name.lower()
        return None


# ============================================
# MOUSE EMULATOR
# ============================================
class MouseEmulator:
    def __init__(self, activity_tensor=None):
        self.mouse = MouseController()
        self.pressed_buttons = set()
        self.activity_tensor = activity_tensor
        self._lock = threading.RLock()
        self.button_map = {
            'left': Button.left, 'right': Button.right, 'middle': Button.middle,
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
            except Exception:
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
# ACTIVITY TENSOR
# ============================================
class ActivityTensor:
    def __init__(self, n_actions: int = 0):
        self.key_states: Dict[str, bool] = defaultdict(bool)
        self.mouse_states: Dict[str, bool] = {
            'left': False, 'right': False, 'middle': False,
            'x1': False, 'x2': False,
        }
        self.mouse_x: int = 0
        self.mouse_y: int = 0
        self.history: deque = deque(maxlen=200)
        self.last_change_time: Dict[str, float] = defaultdict(float)
        self.press_counts: Dict[str, int] = defaultdict(int)
        self._lock = threading.RLock()

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
            'win', 'cmd', 'menu', 'apps',
        ]

        self._n_keys = len(self.all_keys)
        self.key_index = {key: i for i, key in enumerate(self.all_keys)}
        self.mouse_index = {'left': 0, 'right': 1, 'middle': 2, 'x1': 3, 'x2': 4}
        self._n_mouse = 5
        self._n_pos = 2
        self.tensor_size = self._n_keys + self._n_mouse + self._n_pos
        self._dense_extra = self._n_keys + self._n_mouse
        self._dense_size = self.tensor_size + self._n_keys + self._dense_extra
        self._screen_region_w = 1920.0
        self._screen_region_h = 1080.0

        self.n_actions = max(1, n_actions)
        self.last_action_onehot = np.zeros(self.n_actions, dtype=np.float32)
        self.last_action_idx = -1
        self._base_tensor_size = self.tensor_size + self.n_actions

    def update_key(self, key_name: str, pressed: bool):
        with self._lock:
            key_name = key_name.lower().strip()
            idx = self.key_index.get(key_name)
            if idx is None:
                return
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

    def update_last_action(self, action_idx: int):
        with self._lock:
            if 0 <= action_idx < self.n_actions:
                self.last_action_onehot = np.zeros(self.n_actions, dtype=np.float32)
                self.last_action_onehot[action_idx] = 1.0
                self.last_action_idx = action_idx
                self._record_history(f"action:{action_idx}")

    def _record_history(self, event: str):
        self.history.append((time.time(), event))

    def get_tensor(self) -> np.ndarray:
        with self._lock:
            key_vector = np.zeros(self._n_keys, dtype=np.float32)
            for key, state in self.key_states.items():
                idx = self.key_index.get(key)
                if idx is not None:
                    key_vector[idx] = 1.0 if state else 0.0

            mouse_vector = np.zeros(self._n_mouse, dtype=np.float32)
            for btn, state in self.mouse_states.items():
                idx = self.mouse_index.get(btn)
                if idx is not None:
                    mouse_vector[idx] = 1.0 if state else 0.0

            pos_vector = np.array([
                self.mouse_x / self._screen_region_w,
                self.mouse_y / self._screen_region_h,
            ], dtype=np.float32)

            return np.concatenate([
                key_vector, mouse_vector, pos_vector, self.last_action_onehot
            ])

    def get_dense_tensor(self) -> np.ndarray:
        with self._lock:
            base = self.get_tensor()
            current_time = time.time()
            time_since = np.zeros(self._n_keys, dtype=np.float32)
            for key, last_time in self.last_change_time.items():
                idx = self.key_index.get(key)
                if idx is not None:
                    time_since[idx] = min(1.0, (current_time - last_time) / 5.0)

            window_size = 2.0
            recent_activity = np.zeros(self._dense_extra, dtype=np.float32)
            for t, event in self.history:
                if current_time - t < window_size:
                    parts = event.split(':')
                    if len(parts) >= 3:
                        if parts[0] == 'key':
                            idx = self.key_index.get(parts[1])
                            if idx is not None and parts[2] == 'True':
                                recent_activity[idx] = min(1.0, recent_activity[idx] + 0.1)
                        elif parts[0] == 'mouse':
                            midx = self.mouse_index.get(parts[1])
                            if midx is not None and parts[2] == 'True':
                                pos = self._n_keys + midx
                                recent_activity[pos] = min(1.0, recent_activity[pos] + 0.1)
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
                'total_mouse_clicks': sum(
                    self.press_counts.get(f"mouse_{btn}", 0)
                    for btn in self.mouse_states.keys()
                ),
                'key_count': len(self.key_states),
                'mouse_buttons': dict(self.mouse_states),
                'last_events': list(self.history)[-10:] if self.history else [],
                'last_action_idx': self.last_action_idx,
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
            self.last_action_onehot = np.zeros(self.n_actions, dtype=np.float32)
            self.last_action_idx = -1


# ============================================
# SCREEN TOKENIZER
# ============================================
class ScreenTokenizer:
    def __init__(self, grid_height: int = 34, grid_width: int = 60,
                 use_rgb: bool = False,
                 foveated: bool = False,
                 focus_source=None):
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.use_rgb = use_rgb
        self.channels = 3 if use_rgb else 1

        self.foveated = foveated
        self.focus_source = focus_source

        if foveated:
            raw_zones = [
                (grid_height, grid_width, 0.25),
                (max(2, grid_height // 2), max(2, grid_width // 2), 0.50),
                (max(2, grid_height // 4), max(2, grid_width // 4), 1.00),
            ]
        else:
            raw_zones = [(grid_height, grid_width, 1.0)]

        self.zones: List[Tuple[int, int, float]] = []
        self._zone_lens: List[int] = []
        self._zone_shapes: List[Tuple[int, ...]] = []
        self._zone_areas_ratio: List[float] = []

        total_cells = 0
        for (zh, zw, area) in raw_zones:
            zone_len = zh * zw * self.channels
            self.zones.append((zh, zw, area))
            self._zone_lens.append(zone_len)
            if use_rgb:
                self._zone_shapes.append((zh, zw, 3))
            else:
                self._zone_shapes.append((zh, zw))
            self._zone_areas_ratio.append(area)
            total_cells += zone_len

        self.total_cells = total_cells
        self._keys_extra = 10
        self._full_len = self.total_cells + self._keys_extra

        self._viz_shapes = []
        for (zh, zw, _area), shape in zip(self.zones, self._zone_shapes):
            self._viz_shapes.append((zh, zw, shape))

        if not foveated and use_rgb:
            self._uniform_shape = (grid_height, grid_width, 3)
        elif not foveated:
            self._uniform_shape = (grid_height, grid_width)
        else:
            self._uniform_shape = None

        self.history = deque(maxlen=400)
        self.state_buffer = deque(maxlen=25)

        self._encode_buffer = np.empty(self._full_len, dtype=np.int32)
        self._context_size = min(100, self.total_cells)

    def _get_focus(self):
        if self.focus_source is not None:
            try:
                fx, fy = self.focus_source()
                return float(np.clip(fx, 0.0, 1.0)), float(np.clip(fy, 0.0, 1.0))
            except Exception:
                pass
        return 0.5, 0.5

    def _crop_around_focus(self, frame, area_ratio, out_h, out_w):
        h, w = frame.shape[:2]
        cx, cy = self._get_focus()
        side = math.sqrt(area_ratio)
        ch = max(1, int(h * side))
        cw = max(1, int(w * side))
        x1 = max(0, min(w - cw, int(cx * w - cw / 2)))
        y1 = max(0, min(h - ch, int(cy * h - ch / 2)))
        crop = frame[y1:y1 + ch, x1:x1 + cw]
        return cv2.resize(crop, (out_w, out_h), interpolation=cv2.INTER_AREA)

    def encode_frame(self, frame: np.ndarray, keys: List[str] = None) -> List[int]:
        if frame is None:
            return [0] * self._full_len

        tokens = []
        if self.use_rgb:
            for i, (zh, zw, area_ratio) in enumerate(self.zones):
                zone_img = self._crop_around_focus(frame, area_ratio, zh, zw)
                tokens.extend(zone_img.reshape(-1).tolist())
        else:
            for i, (zh, zw, area_ratio) in enumerate(self.zones):
                zone_img = self._crop_around_focus(frame, area_ratio, zh, zw)
                gray = cv2.cvtColor(zone_img, cv2.COLOR_BGR2GRAY)
                tokens.extend(gray.flatten().tolist())

        if keys:
            extra = [0] * self._keys_extra
            for i, key in enumerate(keys[:self._keys_extra]):
                code = ScanCodeRegistry.get_code(key)
                extra[i] = code if code > 0 else 1
            tokens.extend(extra)
        else:
            tokens.extend([0] * self._keys_extra)

        self.history.append(tokens)
        self.state_buffer.append(tokens[:self.total_cells])
        return tokens

    def get_state_hash(self, tokens: List[int]) -> int:
        return hash(tuple(tokens))

    def get_context(self) -> List[int]:
        if not self.state_buffer:
            return [0] * self._context_size
        avg = np.mean(self.state_buffer, axis=0)
        return [int(v) for v in avg[:self._context_size]]

    def visualize_grid(self, tokens: List[int]) -> np.ndarray:
        if not self.foveated:
            return self._visualize_uniform(tokens)
        return self._visualize_foveated(tokens)

    def _visualize_uniform(self, tokens):
        if len(tokens) < self.total_cells:
            return np.zeros(self._uniform_shape, dtype=np.uint8)
        data = tokens[:self.total_cells]
        if self.use_rgb:
            grid = np.array(data, dtype=np.uint8).reshape(self._uniform_shape)
        else:
            grid = np.array(data, dtype=np.uint8).reshape(self._uniform_shape)
        return self._draw_grid(grid, self.grid_width, self.grid_height)

    def _visualize_foveated(self, tokens):
        images = []
        offset = 0
        for i, (zh, zw, _) in enumerate(self.zones):
            zone_len = self._zone_lens[i]
            shape = self._zone_shapes[i]
            data = tokens[offset:offset + zone_len]
            offset += zone_len
            if len(data) < zone_len:
                data = list(data) + [0] * (zone_len - len(data))
            elif len(data) > zone_len:
                data = list(data[:zone_len])
            grid = np.array(data, dtype=np.uint8).reshape(shape)
            images.append(self._draw_grid(grid, zw, zh))

        max_h = max(img.shape[0] for img in images)
        padded = []
        for img in images:
            if len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            if img.shape[0] < max_h:
                pad = np.zeros((max_h - img.shape[0], img.shape[1], 3), dtype=np.uint8)
                img = np.vstack([img, pad])
            padded.append(img)

        sep = np.full((max_h, 20, 3), 50, dtype=np.uint8)
        result = padded[0]
        for p in padded[1:]:
            result = np.hstack([result, sep, p])
        return result

    def _draw_grid(self, grid, gw, gh, scale=20):
        if len(grid.shape) == 2:
            enlarged = cv2.resize(grid, (gw * scale, gh * scale),
                                  interpolation=cv2.INTER_NEAREST)
            enlarged = cv2.cvtColor(enlarged, cv2.COLOR_GRAY2BGR)
        else:
            enlarged = cv2.resize(grid, (gw * scale, gh * scale),
                                  interpolation=cv2.INTER_NEAREST)
        color_line = (100, 100, 100)
        for i in range(gh + 1):
            cv2.line(enlarged, (0, i * scale), (gw * scale, i * scale), color_line, 1)
        for i in range(gw + 1):
            cv2.line(enlarged, (i * scale, 0), (i * scale, gh * scale), color_line, 1)
        return enlarged

    def get_dimensions(self) -> Tuple[int, int]:
        return self.grid_height, self.grid_width


# ============================================
# HYBRID DECISION TREE NODE
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
        self.node_id: int = 0

    def predict(self, x: List[float]) -> float:
        if self.value is not None:
            return self.value
        if x[self.feature_idx] <= self.threshold:
            return self.left.predict(x)
        return self.right.predict(x)


# ============================================
# HEBBIAN WEIGHTS
# ============================================
class HebbianWeights:
    def __init__(self,
                 learning_rate: float = 0.05,
                 decay: float = 0.001,
                 eligibility_decay: float = 0.95,
                 bcm_threshold: float = 0.3,
                 baseline_momentum: float = 0.99,
                 use_bcm: bool = True,
                 use_dopamine: bool = True,
                 use_metaplasticity: bool = True):
        self.weights = defaultdict(float)
        self.eligibility = defaultdict(float)

        self.learning_rate = learning_rate
        self.decay = decay
        self.eligibility_decay = eligibility_decay
        self.bcm_threshold = bcm_threshold
        self.baseline_momentum = baseline_momentum

        self.use_bcm = use_bcm
        self.use_dopamine = use_dopamine
        self.use_metaplasticity = use_metaplasticity

        self.reward_baseline = 0.0

        self.activation_counts = defaultdict(int)
        self.reward_history = defaultdict(list)
        self.lr_history = defaultdict(list)

        self._lock = threading.RLock()

    def activate_path(self, tree_idx: int, path: List[int]):
        with self._lock:
            for i in range(len(path) - 1):
                parent = path[i]
                child = path[i + 1]
                key = (tree_idx, parent, child)
                self.eligibility[key] = 1.0
                self.activation_counts[key] += 1

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_lock', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self._lock = threading.RLock()

    def apply_reward(self, reward: float):
        with self._lock:
            self.reward_baseline = (
                self.baseline_momentum * self.reward_baseline
                + (1 - self.baseline_momentum) * reward
            )

            dopamine = (reward - self.reward_baseline) if self.use_dopamine else reward

            for key in list(self.eligibility.keys()):
                elig = self.eligibility[key]
                if elig < 0.01:
                    self.eligibility[key] *= self.eligibility_decay
                    if self.eligibility[key] < 1e-4:
                        del self.eligibility[key]
                    continue

                lr = self._get_adaptive_lr(key)

                if self.use_bcm:
                    threshold_factor = (elig - self.bcm_threshold)
                else:
                    threshold_factor = 1.0

                delta = lr * dopamine * elig * threshold_factor
                self.weights[key] += delta
                self.weights[key] *= (1 - self.decay)
                self.weights[key] = max(-1.0, min(1.0, self.weights[key]))

                self.reward_history[key].append(reward)
                if len(self.reward_history[key]) > 100:
                    self.reward_history[key] = self.reward_history[key][-100:]

                self.eligibility[key] *= self.eligibility_decay

    def _get_adaptive_lr(self, key) -> float:
        if not self.use_metaplasticity:
            return self.learning_rate
        history = self.reward_history[key]
        if len(history) < 10:
            return self.learning_rate
        recent_std = float(np.std(history[-10:]))
        if recent_std < 0.1:
            return self.learning_rate * 0.5
        elif recent_std > 0.5:
            return self.learning_rate * 1.5
        return self.learning_rate

    def get_weight(self, tree_idx: int, parent: int, child: int) -> float:
        with self._lock:
            return self.weights.get((tree_idx, parent, child), 0.0)

    def get_path_weight(self, tree_idx: int, path: List[int]) -> float:
        with self._lock:
            if len(path) < 2:
                return 0.0
            total = 0.0
            for i in range(len(path) - 1):
                w = self.weights.get((tree_idx, path[i], path[i + 1]), 0.0)
                total += w
            return total / (len(path) - 1)

    def get_statistics(self) -> Dict:
        with self._lock:
            if not self.weights:
                return {
                    'total_connections': 0, 'avg_weight': 0.0,
                    'min_weight': 0.0, 'max_weight': 0.0,
                    'positive': 0, 'negative': 0, 'zero': 0,
                    'active_eligibility': 0, 'baseline': 0.0,
                }
            values = list(self.weights.values())
            return {
                'total_connections': len(self.weights),
                'avg_weight': sum(values) / len(values),
                'min_weight': min(values),
                'max_weight': max(values),
                'positive': sum(1 for v in values if v > 0.01),
                'negative': sum(1 for v in values if v < -0.01),
                'zero': sum(1 for v in values if abs(v) <= 0.01),
                'active_eligibility': len(self.eligibility),
                'baseline': self.reward_baseline,
            }

    def reset(self):
        with self._lock:
            self.weights.clear()
            self.eligibility.clear()
            self.activation_counts.clear()
            self.reward_history.clear()
            self.lr_history.clear()
            self.reward_baseline = 0.0


# ============================================
# HYBRID DECISION FOREST
# ============================================
class HybridDecisionForest:
    def __init__(self, n_trees: int = 14, max_depth: int = 8,
                 hebbian_lr: float = 0.05,
                 hebbian_decay: float = 0.001,
                 eligibility_decay: float = 0.95,
                 resonance_alpha: float = 0.3,
                 resonance_lr: float = 0.01):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.trees = []
        self.is_trained = False
        self.feature_importance = defaultdict(float)
        self.training_count = 0

        self.hebbian = HebbianWeights(
            learning_rate=hebbian_lr,
            decay=hebbian_decay,
            eligibility_decay=eligibility_decay,
            bcm_threshold=0.3,
            use_bcm=True,
            use_dopamine=True,
            use_metaplasticity=True,
        )

        self.resonance = SimpleResonance(
            n_branches=n_trees,
            alpha=resonance_alpha,
            lr=resonance_lr,
            max_iter=3,
            tol=1e-3,
        )

        self.global_branch_weights = {'split': 1.0, 'leaf': 1.0}
        self.node_visits = defaultdict(int)

        self.reward_history = deque(maxlen=200)
        self.drift_detected = False

        self._last_activations: Optional[np.ndarray] = None
        self._last_paths: Optional[List[List[int]]] = None

        self._pred_cache: Dict[int, float] = {}
        self._pred_cache_max = 256

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop('_pred_cache', None)
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        if 'resonance' not in self.__dict__:
            self.resonance = SimpleResonance(n_branches=self.n_trees, alpha=0.3, lr=0.01, max_iter=3)
        if '_last_activations' not in self.__dict__:
            self._last_activations = None
        if '_last_paths' not in self.__dict__:
            self._last_paths = None
        if 'reward_history' not in self.__dict__:
            self.reward_history = deque(maxlen=200)
        if 'drift_detected' not in self.__dict__:
            self.drift_detected = False
        self._pred_cache = {}
        self._pred_cache_max = 256

    def fit(self, X: List[List[float]], y: List[float], update_existing: bool = False):
        if len(X) < 10:
            return
        self.training_count += 1
        self._pred_cache.clear()

        if update_existing and self.is_trained:
            for i in range(min(3, self.n_trees // 2)):
                indices = np.random.choice(len(X), len(X), replace=True)
                X_sample = [X[j] for j in indices]
                y_sample = [y[j] for j in indices]
                tree = self._build_tree(X_sample, y_sample, 0, tree_idx=i)
                if len(self.trees) < self.n_trees:
                    self.trees.append(tree)
                else:
                    self._replace_worst_tree(tree)
        else:
            self.trees = []
            self.hebbian.reset()
            self.resonance.reset()
            self.node_visits = defaultdict(int)
            for i in range(self.n_trees):
                indices = np.random.choice(len(X), len(X), replace=True)
                X_sample = [X[j] for j in indices]
                y_sample = [y[j] for j in indices]
                tree = self._build_tree(X_sample, y_sample, 0, tree_idx=i)
                self.trees.append(tree)
            self.is_trained = True

        print(f"🌲 Лес обучен на {len(X)} примерах (обновление #{self.training_count})")

    def _replace_worst_tree(self, new_tree: HybridTreeNode):
        tree_scores = defaultdict(list)
        for (tree_idx, _, _), w in self.hebbian.weights.items():
            tree_scores[tree_idx].append(w)
        avg_scores = {i: (np.mean(ws) if ws else 0.0) for i, ws in tree_scores.items()}
        all_indices = list(range(len(self.trees)))
        worst_idx = min(all_indices, key=lambda i: avg_scores.get(i, 0.0))
        self.trees[worst_idx] = new_tree

    def _build_tree(self, X: List[List[float]], y: List[float], depth: int,
                    tree_idx: int = 0, node_id: int = 0) -> HybridTreeNode:
        if depth >= self.max_depth or len(set(y)) == 1 or len(X) < 5:
            node = HybridTreeNode(value=float(np.mean(y)))
            node.leaf_count = len(X)
            node.node_id = node_id
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
            node = HybridTreeNode(value=float(np.mean(y)))
            node.leaf_count = len(X)
            node.node_id = node_id
            return node

        left_X = [x for x in X if x[best_feature] <= best_threshold]
        right_X = [x for x in X if x[best_feature] > best_threshold]
        left_y = [y[i] for i, x in enumerate(X) if x[best_feature] <= best_threshold]
        right_y = [y[i] for i, x in enumerate(X) if x[best_feature] > best_threshold]

        self.feature_importance[best_feature] += best_gain

        left_node = self._build_tree(left_X, left_y, depth + 1, tree_idx, node_id * 2 + 1)
        right_node = self._build_tree(right_X, right_y, depth + 1, tree_idx, node_id * 2 + 2)

        node = HybridTreeNode(best_feature, best_threshold, left_node, right_node)
        node.node_id = node_id
        return node

    def _information_gain(self, parent: List[float], left: List[float], right: List[float]) -> float:
        def variance(values):
            if not values:
                return 0
            mean = sum(values) / len(values)
            return sum((v - mean) ** 2 for v in values) / len(values)
        p = len(left) / len(parent)
        return variance(parent) - p * variance(left) - (1 - p) * variance(right)

    def predict(self, x: List[float]) -> float:
        if not self.is_trained or not self.trees:
            return 0.5

        try:
            key = hash(bytes(np.round(np.asarray(x, dtype=np.float32) * 16.0).astype(np.int8).tobytes()))
        except Exception:
            key = None

        if key is not None and key in self._pred_cache:
            return self._pred_cache[key]

        activations = []
        leaf_values = []
        paths = []
        for tree_idx, tree in enumerate(self.trees):
            pred, path = self._predict_with_path(tree, x)
            w = self.hebbian.get_path_weight(tree_idx, path)
            a = 1.0 / (1.0 + math.exp(-w * 3.0))
            activations.append(a)
            leaf_values.append(pred)
            paths.append(path)

        a0 = np.array(activations, dtype=np.float32)
        leaf_values = np.array(leaf_values, dtype=np.float32)

        if np.abs(a0).sum() < 0.1:
            if a0.sum() > 0:
                a_norm = a0 / a0.sum()
            else:
                a_norm = np.ones_like(a0) / len(a0)
            pred = float(np.sum(a_norm * leaf_values))
            self._last_activations = a0
            self._last_paths = paths
        else:
            a_res = self.resonance.settle(a0)
            a_norm = np.abs(a_res) / (np.abs(a_res).sum() + 1e-8)
            pred = float(np.sum(a_norm * leaf_values))
            self._last_activations = a_res
            self._last_paths = paths

        if key is not None:
            if len(self._pred_cache) >= self._pred_cache_max:
                for k in list(self._pred_cache.keys())[:self._pred_cache_max // 2]:
                    self._pred_cache.pop(k, None)
            self._pred_cache[key] = pred

        return pred

    def _predict_with_path(self, tree: HybridTreeNode, x: List[float]) -> Tuple[float, List[int]]:
        path = []
        node = tree
        while node.value is None and node.left is not None and node.right is not None:
            path.append(node.node_id)
            if x[node.feature_idx] <= node.threshold:
                node = node.left
            else:
                node = node.right
        path.append(node.node_id)
        return (node.value if node.value is not None else 0.5), path

    def predict_proba(self, x: List[float], n_classes: int = None) -> List[float]:
        if n_classes is None:
            n_classes = 9
        if n_classes <= 0:
            return []
        if not self.is_trained or not self.trees:
            return [1.0 / n_classes] * n_classes

        tree_preds = []
        tree_weights = []
        for tree_idx, tree in enumerate(self.trees):
            pred, path = self._predict_with_path(tree, x)
            w = self.hebbian.get_path_weight(tree_idx, path)
            a = 1.0 / (1.0 + math.exp(-w * 3.0))
            tree_preds.append(pred)
            tree_weights.append(max(a, 1e-3))

        tree_preds = np.array(tree_preds, dtype=np.float32)
        tree_weights = np.array(tree_weights, dtype=np.float32)

        centers = np.linspace(0.0, 1.0, n_classes, dtype=np.float32)

        probs = np.zeros(n_classes, dtype=np.float64)
        for pred, w in zip(tree_preds, tree_weights):
            logits = -((centers - pred) ** 2) / 0.05
            logits -= logits.max()
            soft = np.exp(logits)
            soft /= soft.sum() + 1e-12
            probs += w * soft

        probs /= probs.sum() + 1e-12
        probs = 0.95 * probs + 0.05 / n_classes
        probs /= probs.sum()
        return probs.tolist()

    def update_weights_online(self, reward: float, path: List[int] = None, tree_idx: int = 0):
        if path is not None:
            self.hebbian.activate_path(tree_idx, path)
        self.hebbian.apply_reward(reward)

        if self._last_activations is not None:
            self.resonance.update(self._last_activations, reward)

        self.reward_history.append(reward)

        if reward > self.hebbian.reward_baseline:
            self.global_branch_weights['split'] *= (1 + 0.005)
            self.global_branch_weights['leaf'] *= (1 + 0.0025)
        else:
            self.global_branch_weights['split'] *= (1 - 0.0025)
            self.global_branch_weights['leaf'] *= (1 - 0.001)
        for key in list(self.global_branch_weights.keys()):
            self.global_branch_weights[key] = max(0.5, min(2.0, self.global_branch_weights[key]))

        self._pred_cache.clear()

    def get_resonance_state(self) -> Optional[np.ndarray]:
        return self._last_activations

    def get_branch_weight_for_state(self, x: List[float]) -> float:
        if not self.is_trained or not self.trees:
            return 0.0
        try:
            self.predict(x)
            if self._last_activations is not None:
                return float(np.abs(self._last_activations).mean())
        except Exception:
            pass
        return 0.0

    def detect_drift(self) -> bool:
        if len(self.reward_history) < 50:
            return False
        recent = list(self.reward_history)[-20:]
        older = list(self.reward_history)[-50:-20]
        if not recent or not older:
            return False
        recent_mean = float(np.mean(recent))
        older_mean = float(np.mean(older))
        if older_mean > 0 and recent_mean < older_mean * 0.7:
            self.drift_detected = True
            return True
        return False

    def get_branch_statistics(self) -> Dict:
        stats = self.hebbian.get_statistics()
        stats['global_weights'] = dict(self.global_branch_weights)
        stats['most_visited'] = sorted(
            self.node_visits.items(), key=lambda x: x[1], reverse=True
        )[:10]
        stats['resonance'] = self.resonance.get_statistics()
        return stats


# ============================================
# SMART REWARD SYSTEM
# ============================================
class SmartRewardSystem:
    def __init__(self):
        self.reward_history = deque(maxlen=100)
        self.action_history = deque(maxlen=50)
        self.state_history = deque(maxlen=30)
        self.baseline_change = 0.05
        self.movement_threshold = 0.03
        self.stuck_threshold = 0.01
        self.last_changes = deque(maxlen=20)
        self.pattern_memory = defaultdict(list)
        self.action_success_rate = defaultdict(float)
        self.consecutive_failures = 0
        self.consecutive_repeats = 0
        self.consecutive_cycles = 0

    def calculate_smart_reward(self, before_state, after_state, action, context):
        change = self._calculate_change(before_state, after_state)
        self.last_changes.append(change)
        if after_state is not None:
            self.state_history.append(after_state)

        chaos_penalty = self._calculate_chaos_penalty(change)
        stagnation_penalty = self._calculate_stagnation_penalty(change)
        repetition_penalty = self._calculate_repetition_penalty(action)
        cycle_penalty, _ = self._calculate_cycle_penalty(before_state, change)
        progress_bonus = self._calculate_progress_bonus(change)
        diversity_bonus = self._calculate_diversity_bonus(action)
        spam_penalty = self._calculate_spam_penalty()
        pattern_bonus = self._calculate_pattern_bonus(before_state, after_state)
        activity_bonus = 0.2 if change > self.stuck_threshold else 0.0

        reward = (
            + progress_bonus * 2.0
            - chaos_penalty * 1.5
            - stagnation_penalty * 2.5
            - repetition_penalty * 0.4
            - cycle_penalty * 0.6
            + diversity_bonus * 0.3
            - spam_penalty * 1.0
            + pattern_bonus * 0.8
            + activity_bonus * 0.5
        )
        reward = float(np.clip(reward, -3.0, 3.0))
        self.reward_history.append(reward)
        self.action_history.append(action)
        self.action_success_rate[action] = (
            self.action_success_rate.get(action, 0) * 0.9 + (reward > 0) * 0.1
        )
        return reward

    def _calculate_change(self, before, after):
        if before is None or after is None:
            return 0.0
        if len(before) == 0 or len(after) == 0 or len(before) != len(after):
            return 0.0
        diff = sum(abs(float(b) - float(a)) for b, a in zip(before, after))
        return diff / len(before)

    def _calculate_stagnation_penalty(self, change):
        if change < self.stuck_threshold:
            self.consecutive_failures += 1
            self.consecutive_cycles = 0
            return min(self.consecutive_failures * 0.3, 2.5)
        self.consecutive_failures = 0
        return 0.0

    def _calculate_repetition_penalty(self, action):
        if len(self.action_history) < 5:
            return 0.0
        recent = list(self.action_history)[-10:]
        count = recent.count(action)
        if count > 2:
            self.consecutive_repeats += 1
            return min((count - 2) * 0.15, 0.8)
        self.consecutive_repeats = 0
        return 0.0

    def _calculate_cycle_penalty(self, current_state, immediate_change):
        if current_state is None or len(self.state_history) < 3:
            self.consecutive_cycles = 0
            return 0.0, False
        min_diff = float('inf')
        for past in list(self.state_history)[-6:-1]:
            if len(past) == len(current_state):
                diff = sum(abs(float(a) - float(b))
                           for a, b in zip(current_state, past)) / len(current_state)
                if diff < min_diff:
                    min_diff = diff
        if min_diff < 0.02 and immediate_change > self.movement_threshold:
            self.consecutive_cycles += 1
            return min(self.consecutive_cycles * 0.05, 0.4), True
        self.consecutive_cycles = 0
        return 0.0, False

    def _calculate_chaos_penalty(self, change):
        if change > 0.2:
            return min((change - 0.2) * 5.0, 2.0)
        return 0.0

    def _calculate_progress_bonus(self, change):
        if self.movement_threshold < change < 0.15:
            optimal, sigma = 0.08, 0.05
            return math.exp(-((change - optimal) ** 2) / (2 * sigma ** 2)) * 1.5
        return 0.0

    def _calculate_diversity_bonus(self, action):
        if len(self.action_history) < 10:
            return 0.0
        unique = len(set(list(self.action_history)[-20:]))
        diversity = unique / min(20, len(self.action_history))
        return diversity * 0.3 if diversity > 0.5 else 0.0

    def _calculate_spam_penalty(self):
        if len(self.action_history) < 5:
            return 0.0
        recent = list(self.action_history)[-10:]
        if len(recent) > 8:
            return min((len(recent) - 8) * 0.2, 1.0)
        return 0.0

    def _calculate_pattern_bonus(self, before, after):
        if before is None or after is None:
            return 0.0
        if len(before) <= 150 or len(after) <= 150:
            return 0.0
        cb = before[100:150]
        ca = after[100:150]
        if not cb or not ca:
            return 0.0
        change = sum(abs(float(a) - float(b)) for a, b in zip(cb, ca)) / len(cb)
        return 0.5 if 0.02 < change < 0.1 else 0.0

    def get_action_feedback(self, action):
        rate = self.action_success_rate.get(action, 0.0)
        if rate > 0.7:
            return "✅ Эффективно"
        elif rate > 0.4:
            return "🟡 Нормально"
        return "❌ Неэффективно"

    def get_statistics(self):
        avg = float(np.mean(list(self.reward_history))) if self.reward_history else 0
        std = float(np.std(list(self.reward_history))) if self.reward_history else 0
        actions = (list(self.action_history)[-20:]
                   if len(self.action_history) > 20 else list(self.action_history))
        return {
            'avg_reward': avg,
            'std_reward': std,
            'consecutive_failures': self.consecutive_failures,
            'consecutive_cycles': self.consecutive_cycles,
            'action_diversity': len(set(actions)) if actions else 0,
            'best_action': (max(self.action_success_rate, key=self.action_success_rate.get)
                            if self.action_success_rate else 'none'),
            'best_action_rate': (max(self.action_success_rate.values())
                                 if self.action_success_rate else 0),
        }


# ============================================
# ACTION EXECUTOR (с ParameterMemory)
# ============================================
class ActionExecutor:
    MOUSE_ACTIONS = {'click', 'double_click', 'right_click', 'middle_click',
                     'drag_short', 'drag_medium', 'drag_long',
                     'scroll_up', 'scroll_down', 'scroll_left', 'scroll_right'}

    def __init__(self, activity_tensor: ActivityTensor, n_actions: int = 0,
                 parameter_memory: Optional[ParameterMemory] = None):
        self.activity_tensor = activity_tensor
        self.keyboard = KeyboardEmulator(activity_tensor=None)
        self.mouse = MouseEmulator(activity_tensor=None)
        self.protection = SystemProtection()
        self.blocked_actions = []
        self._lock = threading.RLock()

        self.n_actions = max(1, n_actions)
        self.parameter_memory = parameter_memory or ParameterMemory(self.n_actions)

        self._last_state = None
        self._last_state_hash: Optional[int] = None
        self._last_action: Optional[str] = None
        self._last_action_idx: int = -1

        self.default_ranges = {
            'x': (200, 1700),
            'y': (200, 800),
            'duration': (0.05, 0.2),
            'distance': (50, 500),
        }

    def set_context(self, state, state_hash: int, action: str, action_idx: int):
        with self._lock:
            self._last_state = state
            self._last_state_hash = int(state_hash)
            self._last_action = action
            self._last_action_idx = int(action_idx)

    def set_state(self, state):
        with self._lock:
            self._last_state = state

    def execute_mouse_only(self, action: str, x: int = None, y: int = None) -> bool:
        with self._lock:
            predicted = self._predict_params(action)
            if x is None:
                x = predicted.get('x', random.randint(*self.default_ranges['x']))
            if y is None:
                y = predicted.get('y', random.randint(*self.default_ranges['y']))

            x = int(np.clip(x, 0, 1919))
            y = int(np.clip(y, 0, 1079))

            if action == 'click':
                return self._do_click(x, y, 'left')
            elif action == 'double_click':
                return self._do_click(x, y, 'left', clicks=2)
            elif action == 'right_click':
                return self._do_click(x, y, 'right')
            elif action == 'middle_click':
                return self._do_click(x, y, 'middle')
            elif action in ('drag_short', 'drag_medium', 'drag_long'):
                dist = predicted.get('distance')
                if dist is None:
                    if action == 'drag_short':
                        dist = random.randint(50, 150)
                    elif action == 'drag_medium':
                        dist = random.randint(150, 300)
                    else:
                        dist = random.randint(300, 500)
                return self._do_drag(x, y, int(np.clip(dist, 20, 800)))
            elif action == 'scroll_up':
                self.mouse.scroll(dy=-1); return True
            elif action == 'scroll_down':
                self.mouse.scroll(dy=1); return True
            elif action == 'scroll_left':
                self.mouse.scroll(dx=-1); return True
            elif action == 'scroll_right':
                self.mouse.scroll(dx=1); return True
            return False

    def _predict_params(self, action: str) -> Dict[str, float]:
        if self._last_state_hash is None or self._last_action_idx < 0:
            return {}
        try:
            return self.parameter_memory.predict(
                self._last_state_hash, self._last_action_idx
            )
        except Exception:
            return {}

    def get_last_params(self) -> Dict[str, float]:
        with self._lock:
            return self._predict_params(self._last_action or '')

    def execute_keyboard_only(self, action: str) -> bool:
        with self._lock:
            if action == 'release_all':
                self.keyboard.release_all()
                self.mouse.release_all()
                return True
            if '+' in action:
                return self._do_key_combo(action.split('+'))
            predicted = self._predict_params(action)
            duration = predicted.get('duration', random.uniform(0.05, 0.15))
            duration = float(np.clip(duration, 0.02, 1.0))
            return self.keyboard.press_key(action, duration=duration)

    def execute_mixed(self, action: str) -> bool:
        with self._lock:
            parts = action.split('+')
            mouse_buttons = {'left', 'right', 'middle'}
            mouse_part = None
            key_parts = []
            for p in parts:
                pl = p.lower().strip()
                if pl in mouse_buttons:
                    mouse_part = pl
                else:
                    key_parts.append(pl)
            if mouse_part is None:
                return False
            combo_str = '+'.join(key_parts + [mouse_part])
            if self.protection.is_dangerous(combo_str):
                self.blocked_actions.append(combo_str)
                print(f"🛡️ ЗАБЛОКИРОВАНО: {combo_str}")
                return False
            try:
                for key in key_parts:
                    self.keyboard.press_key(key, duration=0, update_tensor=True)
                time.sleep(random.uniform(0.01, 0.03))
                self.mouse.press_button(mouse_part, duration=0.1, update_tensor=True)
                for key in reversed(key_parts):
                    self.keyboard.release_key(key, update_tensor=True)
                return True
            except Exception as e:
                print(f"⚠️ Ошибка mixed: {e}")
                self.keyboard.release_all(update_tensor=True)
                self.mouse.release_all(update_tensor=True)
                return False

    def execute_sequence(self, actions: List[str], delay: float = 0.1) -> bool:
        with self._lock:
            for a in actions:
                if not self.execute(a):
                    return False
                time.sleep(delay)
            return True

    def execute(self, action: str) -> bool:
        if self.protection.is_dangerous(action):
            self.blocked_actions.append(action)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {action}")
            return False
        if action in ['wait_short', 'wait_long']:
            time.sleep((0.5 if 'short' in action else 2.0) + random.random() * 0.3)
            return True
        if action == 'release_all':
            self.keyboard.release_all(update_tensor=False)
            self.mouse.release_all(update_tensor=False)
            return True
        if action.startswith('mouse_') and '+' in action:
            return self.execute_mixed(action)
        if '+' in action:
            return self.execute_keyboard_only(action)
        if action in self.MOUSE_ACTIONS:
            return self.execute_mouse_only(action)
        return self.execute_keyboard_only(action)

    def _do_click(self, x: int, y: int, button: str, clicks: int = 1):
        self.mouse.move_to(x + random.randint(-8, 8),
                           y + random.randint(-8, 8), update_tensor=True)
        for i in range(clicks):
            self.mouse.press_button(button, duration=0, update_tensor=True)
            time.sleep(random.uniform(0.03, 0.08))
            self.mouse.release_button(button, update_tensor=True)
            if i < clicks - 1:
                time.sleep(random.uniform(0.05, 0.15))
        return True

    def _do_drag(self, from_x: int, from_y: int, distance: int):
        to_x = from_x + random.randint(-distance, distance)
        to_y = from_y + random.randint(-distance, distance)
        self.mouse.move_to(from_x, from_y, update_tensor=True)
        time.sleep(random.uniform(0.05, 0.1))
        self.mouse.press_button('left', duration=0, update_tensor=True)
        time.sleep(random.uniform(0.05, 0.1))
        steps = random.randint(10, 20)
        for i in range(steps):
            progress = i / steps
            dev = math.sin(progress * math.pi) * random.randint(2, 10)
            x = from_x + (to_x - from_x) * progress + dev
            y = from_y + (to_y - from_y) * progress + dev * 0.2
            self.mouse.move_to(int(x), int(y), update_tensor=False)
            time.sleep(random.uniform(0.005, 0.015))
        self.mouse.move_to(to_x + random.randint(-3, 3),
                           to_y + random.randint(-3, 3), update_tensor=True)
        time.sleep(random.uniform(0.02, 0.05))
        self.mouse.release_button('left', update_tensor=True)
        return True

    def _do_key_combo(self, combo: List[str]) -> bool:
        normalized = [('cmd' if k.lower().strip() == 'win' else k.lower().strip())
                      for k in combo]
        combo_str = '+'.join(normalized)
        if self.protection.is_dangerous(combo_str):
            self.blocked_actions.append(combo_str)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {combo_str}")
            return False
        predicted = self._predict_params(combo_str)
        duration = predicted.get('duration', random.uniform(0.1, 0.3))
        duration = float(np.clip(duration, 0.03, 1.5))
        self.keyboard.press_combo(normalized, duration=duration, update_tensor=True)
        return True


# ============================================
# STATE ENCODER
# ============================================
class StateEncoder:
    def __init__(self, activity_tensor: ActivityTensor, tokenizer: ScreenTokenizer,
                 max_vector_size: int = 3200):
        self.activity_tensor = activity_tensor
        self.tokenizer = tokenizer
        self.max_vector_size = max_vector_size

        self._activity_size = activity_tensor.tensor_size + activity_tensor.n_actions
        self._context_size = tokenizer._context_size
        self._screen_size = tokenizer.total_cells

        self._total_combined = self._activity_size + self._context_size + self._screen_size
        self._final_size = min(self._total_combined, max_vector_size)
        self._padding_size = max(0, max_vector_size - self._total_combined)

        self._buffer = np.zeros(max_vector_size, dtype=np.float32)

    def encode(self, frame, keys=None) -> np.ndarray:
        if frame is None:
            return np.zeros(self.max_vector_size, dtype=np.float32)

        tokens = self.tokenizer.encode_frame(frame, keys)
        context = self.tokenizer.get_context()
        activity = self.activity_tensor.get_tensor()

        screen_np = np.array(tokens[:self._screen_size], dtype=np.float32) / 255.0
        context_np = np.array(context[:self._context_size], dtype=np.float32) / 255.0
        activity_np = np.asarray(activity, dtype=np.float32)

        n = 0
        an = activity_np.shape[0]
        cn = context_np.shape[0]
        sn = screen_np.shape[0]

        end_a = min(n + an, self.max_vector_size)
        self._buffer[n:end_a] = activity_np[:end_a - n]
        n = end_a

        end_c = min(n + cn, self.max_vector_size)
        self._buffer[n:end_c] = context_np[:end_c - n]
        n = end_c

        end_s = min(n + sn, self.max_vector_size)
        self._buffer[n:end_s] = screen_np[:end_s - n]
        n = end_s

        if n < self.max_vector_size:
            self._buffer[n:] = 0.0

        return self._buffer.copy()

    def encode_dense(self, frame: np.ndarray, keys: List[str] = None) -> List[float]:
        tokens = self.tokenizer.encode_frame(frame, keys)
        context = self.tokenizer.get_context()
        dense = self.activity_tensor.get_dense_tensor()

        screen_np = np.array(tokens[:self._screen_size], dtype=np.float32) / 255.0
        context_np = np.array(context[:self._context_size], dtype=np.float32) / 255.0
        dense_np = np.asarray(dense, dtype=np.float32)

        combined = np.concatenate([dense_np, context_np, screen_np])
        if combined.shape[0] > 1480:
            return combined[:1480].tolist()
        else:
            padding = np.zeros(1480 - combined.shape[0], dtype=np.float32)
            return np.concatenate([combined, padding]).tolist()
