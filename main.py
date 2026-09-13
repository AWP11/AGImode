# main.py
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
from typing import List, Tuple, Dict, Optional, Any
from datetime import datetime

from pynput.mouse import Button, Controller as MouseController, Listener as MouseListener
from pynput.keyboard import Key, Controller as KeyboardController, Listener as KeyboardListener

from core import (
    ScreenTokenizer,
    SmartRewardSystem,
    SystemProtection,
    ScanCodeRegistry,
    KeyboardEmulator,
    MouseEmulator,
    ActivityTensor,
    ActionExecutor,
    StateEncoder,
    HybridTreeNode,
    HybridDecisionForest,
    SlidingMemory,
    WeightedSlidingMemory,
    SimpleResonance,
    ParameterMemory,
)

SCREEN_REGION = {"top": 0, "left": 0, "width": 1920, "height": 1080}
MOUSE_IDLE_THRESHOLD = 1.5
LEARNING_INTERVAL = 15
RECORDING_BUFFER_SIZE = 3000


class HybridForestBot:
    def __init__(self):
        self.sct = mss.mss()

        self._init_actions()

        self.activity_tensor = ActivityTensor(n_actions=len(self.actions))

        self.keyboard = KeyboardEmulator(activity_tensor=None)
        self.mouse_emulator = MouseEmulator(activity_tensor=None)

        # --- ParameterMemory ДО ActionExecutor ---
        self.parameter_memory = ParameterMemory(
            n_actions=len(self.actions),
            decay=0.995,
        )

        self.executor = ActionExecutor(
            activity_tensor=self.activity_tensor,
            n_actions=len(self.actions),
            parameter_memory=self.parameter_memory,
        )

        self.tokenizer = ScreenTokenizer(
            grid_height=55,
            grid_width=98,
            use_rgb=True,
            foveated=True,
            focus_source=self._get_mouse_focus,
        )
        self.state_encoder = StateEncoder(self.activity_tensor, self.tokenizer)

        self.mouse = MouseController()

        self.mouse_is_pressed = False
        self.press_start_time = 0
        self.press_start_pos = (0, 0)
        self.drag_path = deque(maxlen=200)
        self.key_press_times: Dict[str, float] = {}

        self.current_state: List[float] = []
        self.state_lock = threading.RLock()

        self.last_mouse_move = time.time()
        self.last_action_time = time.time()
        self.action_counter = 0

        self.reward_history = deque(maxlen=300)
        self.bot_state = 'idle'
        self.recording = False
        self.auto_mode = False
        self.running = False

        self.forest = HybridDecisionForest(n_trees=22, max_depth=8)

        self.memory = WeightedSlidingMemory(
            max_size=RECORDING_BUFFER_SIZE,
            decay=0.99,
            soft_threshold=0.15,
            hard_threshold=0.02,
            min_lifetime=200,
            w_reward=1.0,
            w_branch=0.8,
            w_novelty=0.6,
            novelty_scale=50.0,
            base_strength=0.5,
            min_strength=0.2,
            max_strength=5.0,
            rebalance_interval=200,
        )
        self.data_lock = threading.RLock()

        self.experience_history = deque(maxlen=RECORDING_BUFFER_SIZE)
        self.recent_actions = deque(maxlen=LEARNING_INTERVAL)
        self.recent_rewards = deque(maxlen=LEARNING_INTERVAL)
        self.model_file = "hybrid_forest_bot.pkl"

        self.reward_system = SmartRewardSystem()

        self.adaptive_learning = True
        self.learning_phase = 'exploration'
        self.phase_switches = 0
        self.exploration_threshold = 50
        self.exploitation_threshold = 200

        self.blocked_actions: List[str] = []

        self.stats = {
            'total_actions': 0,
            'successful_actions': 0,
            'average_reward': 0,
            'last_action': 'none',
            'mouse_actions': 0,
            'keyboard_actions': 0,
            'combo_actions': 0,
        }

        self.state_visits = defaultdict(int)
        self.novelty_decay = 0.995

        self.load_model()
        self.print_menu()

    def _init_actions(self):
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
            'win_left', 'win_right', 'apps',
        ]

        self.mouse_actions = [
            'click', 'double_click', 'right_click', 'middle_click',
            'drag_short', 'drag_medium', 'drag_long',
            'scroll_up', 'scroll_down', 'scroll_left', 'scroll_right',
        ]

        self.keyboard_actions = self.all_keys.copy()
        self.combo_actions = self._generate_random_combos()
        self.special_actions = ['wait_short', 'wait_long', 'release_all']

        self.actions = (
            self.mouse_actions
            + self.keyboard_actions
            + self.combo_actions
            + self.special_actions
        )

    def _generate_random_combos(self, n_combos: int = 100) -> List[str]:
        combos = []
        modifiers = ['ctrl', 'alt', 'shift']

        for mod in modifiers:
            for key in random.sample(self.all_keys, min(15, len(self.all_keys))):
                if key not in modifiers and key != 'print_screen':
                    combo = f"{mod}+{key}"
                    if not SystemProtection.is_dangerous(combo):
                        combos.append(combo)

        for _ in range(20):
            mod1, mod2 = random.sample(modifiers, 2)
            key = random.choice([k for k in self.all_keys
                                 if k not in modifiers and k != 'print_screen'])
            combo = f"{mod1}+{mod2}+{key}"
            if not SystemProtection.is_dangerous(combo):
                combos.append(combo)

        for _ in range(30):
            keys = random.sample([k for k in self.all_keys
                                  if k not in modifiers and k != 'print_screen'], 2)
            combos.append(f"{keys[0]}+{keys[1]}")

        for _ in range(10):
            keys = random.sample([k for k in self.all_keys if k != 'print_screen'], 3)
            combo = f"{keys[0]}+{keys[1]}+{keys[2]}"
            if not SystemProtection.is_dangerous(combo):
                combos.append(combo)

        return combos

    def _get_mouse_focus(self):
        try:
            x, y = self.mouse.position
            return x / 1920.0, y / 1080.0
        except Exception:
            return 0.5, 0.5

    def print_menu(self):
        h, w = self.tokenizer.get_dimensions()
        print("\n" + "=" * 70)
        print("🌲 ГИБРИДНЫЙ ЛЕС + ВЕТВЕ́ННЫЙ РЕЗОНАНС (v3)")
        print("=" * 70)
        print(f"🎯 Разрешение сетки: {w}×{h} = {h * w} пикселей")
        print(f"🌲 Деревьев: {self.forest.n_trees}")
        print(f"🎵 Резонанс ветвей: активен (α=0.3, 3 итерации)")
        print(f"🧠 Память: {len(self.memory)}/{self.memory.max_size} (взвешенная v2)")
        print(f"🎛️  Память параметров: {len(self.parameter_memory)} записей")
        print(f"📊 Статус: {'🟢 Обучен' if self.forest.is_trained else '⚪ Не обучен'}")
        print(f"\n🖱️  Мышиных действий: {len(self.mouse_actions)}")
        print(f"⌨️  Одиночных клавиш: {len(self.keyboard_actions)}")
        print(f"🔗 Комбинаций клавиш: {len(self.combo_actions)}")
        print(f"⚡ Всего действий: {len(self.actions)}")
        print(f"🛡️  Защита: Активна")
        print("\n⌨️  УПРАВЛЕНИЕ:")
        print("  F1 - запись (вкл/выкл)")
        print("  F2 - авто-режим (вкл/выкл)")
        print("  F3 - сохранить модель")
        print("  F5 - показать статистику")
        print("  F6 - 📸 ПОКАЗАТЬ ЧТО ВИДИТ ИИ")
        print("  F7 - показать важность признаков")
        print("  F8 - показать примеры комбинаций")
        print("  F9 - показать заблокированные действия")
        print("  ESC - остановка")
        print("=" * 70)

    def capture_screen(self):
        try:
            screenshot = self.sct.grab(SCREEN_REGION)
            img = np.array(screenshot)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception:
            return None

    def encode_state(self, frame: np.ndarray, keys: List[str] = None) -> List[float]:
        if frame is None:
            return []
        return self.state_encoder.encode(frame, keys or []).tolist()

    def calculate_adaptive_reward(self, before, after, action):
        base_reward = self.reward_system.calculate_smart_reward(
            before, after, action, self.current_state
        )
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
        action_list = list(self.reward_system.action_history)[-10:]
        recent_actions = set(action_list)
        return 0.5 if action not in recent_actions else 0.0

    def _calculate_optimal_reward(self, action):
        success_rate = self.reward_system.action_success_rate.get(action, 0.0)
        if isinstance(success_rate, (int, float)):
            return success_rate * 0.5
        return 0.0

    def _calculate_repetition_penalty(self, action):
        if len(self.reward_system.action_history) < 5:
            return 0.0
        action_list = list(self.reward_system.action_history)[-10:]
        repetition_count = action_list.count(action)
        if repetition_count > 2:
            return (repetition_count - 2) * 0.2
        return 0.0

    def update_learning_phase(self):
        total_actions = self.stats['total_actions']
        avg_reward = (np.mean(list(self.reward_system.reward_history))
                      if self.reward_system.reward_history else 0)
        if total_actions < self.exploration_threshold:
            self.learning_phase = 'exploration'
        elif total_actions < self.exploitation_threshold:
            self.learning_phase = 'exploitation' if avg_reward > 0.5 else 'exploration'
        else:
            self.learning_phase = 'fine_tuning' if avg_reward > 1.0 else 'exploitation'
        if self.phase_switches < 10:
            print(f"🔄 Фаза обучения: {self.learning_phase} (награда: {avg_reward:.2f})")
            self.phase_switches += 1

    def select_action_with_phase(self, state):
        if self.learning_phase == 'exploration':
            if random.random() < 0.7:
                return self._select_random_action(), 0.3
            return self._select_smart_action(state), 0.7
        elif self.learning_phase == 'exploitation':
            if random.random() < 0.8:
                return self._select_smart_action(state), 0.8
            return self._select_random_action(), 0.2
        else:
            if random.random() < 0.95:
                return self._select_smart_action(state), 0.95
            return self._select_random_action(), 0.05

    def _select_random_action(self):
        action = random.choice(self.actions)
        while SystemProtection.is_dangerous(action):
            action = random.choice(self.actions)
        return action

    def _select_smart_action(self, state):
        if not self.forest.is_trained or not state:
            return self._select_random_action()

        try:
            n_classes = len(self.actions)
            probas = self.forest.predict_proba(state, n_classes=n_classes)

            weighted = []
            recent_list = list(self.recent_actions)
            for i, prob in enumerate(probas):
                action = self.actions[i]
                success_rate = self.reward_system.action_success_rate.get(action, 0.0)

                recent_count = recent_list.count(action)
                recent_penalty = min(recent_count * 0.1, 0.5)

                weighted.append(prob * 0.6 + success_rate * 0.3 - recent_penalty)

            action_idx = int(np.argmax(weighted))
            action = self.actions[action_idx]

            if SystemProtection.is_dangerous(action):
                return self._select_random_action()
            return action
        except Exception as e:
            print(f"⚠️ _select_smart_action error: {e}")
            return self._select_random_action()

    def select_action(self, state: List[float]) -> Tuple[str, float]:
        if self.forest.is_trained and state and random.random() < 0.85:
            try:
                n_classes = len(self.actions)
                probas = self.forest.predict_proba(state, n_classes=n_classes)
                action_idx = int(np.argmax(probas))
                action = self.actions[action_idx]
                if SystemProtection.is_dangerous(action):
                    return self._select_random_action(), 0.1
                return action, float(probas[action_idx])
            except Exception:
                pass
        return self._select_random_action(), 0.1

    def execute_action(self, action: str) -> bool:
        if SystemProtection.is_dangerous(action):
            self.blocked_actions.append(action)
            print(f"🛡️ ЗАБЛОКИРОВАНО: {action}")
            return False

        if action in self.mouse_actions:
            self.stats['mouse_actions'] += 1
        elif '+' in action:
            self.stats['combo_actions'] += 1
        elif action in self.keyboard_actions:
            self.stats['keyboard_actions'] += 1

        return self.executor.execute(action)

    def train(self, force: bool = False):
        if len(self.memory) < 30:
            print(f"⚠️ Недостаточно данных ({len(self.memory)}/30)")
            return
        if not force and len(self.memory) % LEARNING_INTERVAL != 0:
            return

        X, y = self.memory.sample(min(500, len(self.memory)))
        if len(X) < 10:
            return

        try:
            self.forest.fit(X, y, update_existing=self.forest.is_trained)
            accuracy = self.evaluate_model()
            mem_stats = self.memory.get_statistics()
            param_stats = self.parameter_memory.get_statistics()
            print(f"📊 Hit-rate: {accuracy:.2%} | "
                  f"Память: {mem_stats['size']}/{mem_stats['max_size']} "
                  f"(средняя сила: {mem_stats['avg_strength']:.2f}, "
                  f"слабых: {mem_stats['weak']}, сильных: {mem_stats['strong']}, "
                  f"с параметрами: {mem_stats.get('with_params', 0)}) | "
                  f"Параметры: {param_stats['size']} "
                  f"(sub_reward ср.: {param_stats['avg_sub_reward']:.2f})")
        except Exception as e:
            print(f"⚠️ Ошибка обучения: {e}")

    def evaluate_model(self) -> float:
        if len(self.memory) < 10 or not self.forest.is_trained:
            return 0.0
        test_size = min(30, len(self.memory))
        X_test, y_test = self.memory.sample(test_size)
        if not X_test:
            return 0.0
        correct = 0
        for state, label in zip(X_test, y_test):
            try:
                probas = self.forest.predict_proba(state, n_classes=len(self.actions))
                if int(np.argmax(probas)) == int(label):
                    correct += 1
            except Exception:
                pass
        return correct / len(X_test)

    def record_action(self, action: str, from_bot: bool = False,
                      reward: float = 0.0, params: Optional[Dict] = None):
        with self.state_lock:
            if not self.current_state:
                return
            state_copy = self.current_state.copy()

        try:
            action_idx = self.actions.index(action) if action in self.actions else 0
        except ValueError:
            action_idx = 0

        branch_weight = 0.0
        try:
            branch_weight = self.forest.get_branch_weight_for_state(state_copy)
        except Exception:
            branch_weight = 0.0

        state_hash = self.memory._hash_state(state_copy)

        try:
            self.memory.add(
                state_copy,
                action_idx,
                reward=reward,
                branch_weight=branch_weight,
                params=params or {},
            )
            if params:
                self.parameter_memory.add(
                    state_hash,
                    action_idx,
                    params,
                    sub_reward=0.0,
                    weight=1.0,
                )
            self.recent_actions.append(action)
            self.activity_tensor.update_last_action(action_idx)
        except Exception as e:
            print(f"⚠️ memory.add error: {e}")
            return

        if len(self.memory) % LEARNING_INTERVAL == 0:
            try:
                self.train()
            except Exception as e:
                print(f"⚠️ train error: {e}")

    def on_mouse_move(self, x, y):
        self.last_mouse_move = time.time()
        if self.mouse_is_pressed:
            self.drag_path.append((x, y))
        self.activity_tensor.update_mouse_position(x, y)

    def on_mouse_click(self, x, y, button, pressed):
        if button == Button.left:
            if pressed:
                self.mouse_is_pressed = True
                self.press_start_time = time.time()
                self.press_start_pos = (x, y)
                self.drag_path.clear()
                self.drag_path.append((x, y))
                self.activity_tensor.update_mouse_button('left', True)
            else:
                self.mouse_is_pressed = False
                self.activity_tensor.update_mouse_button('left', False)
                hold = time.time() - self.press_start_time

                if self.recording:
                    start_x, start_y = self.press_start_pos
                    distance = math.hypot(x - start_x, y - start_y)

                    if hold > 0.5 or distance > 250:
                        action = 'drag_long'
                    elif hold > 0.25 or distance > 100:
                        action = 'drag_medium'
                    elif hold > 0.12 or distance > 30:
                        action = 'drag_short'
                    else:
                        action = 'click'

                    if action in self.actions:
                        params = {
                            'x': float(start_x),
                            'y': float(start_y),
                            'duration': float(hold),
                            'distance': float(distance),
                        }
                        self.record_action(action, from_bot=False, params=params)
        elif button == Button.right:
            self.activity_tensor.update_mouse_button('right', pressed)
        elif button == Button.middle:
            self.activity_tensor.update_mouse_button('middle', pressed)

    def on_key_press(self, key):
        try:
            key_name = self._key_to_name(key)
            if key_name:
                self.key_press_times[key_name] = time.time()
                self.activity_tensor.update_key(key_name, True)

            if key == Key.f1:
                self.recording = not self.recording
                print(f"\n📝 Запись: {'🟢 ВКЛ' if self.recording else '🔴 ВЫКЛ'}")
                self.bot_state = 'recording' if self.recording else 'idle'
            elif key == Key.f2:
                self.auto_mode = not self.auto_mode
                print(f"\n🤖 Авто-режим: {'🟢 ВКЛ' if self.auto_mode else '🔴 ВЫКЛ'}")
                self.bot_state = 'auto' if self.auto_mode else 'idle'
                if self.auto_mode:
                    self.last_mouse_move = 0
            elif key == Key.f3:
                self.save_model()
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
        except Exception as e:
            print(f"⚠️ Ошибка обработки клавиши: {e}")
        return True

    def on_key_release(self, key):
        try:
            key_name = self._key_to_name(key)
            if not key_name:
                return True

            duration = None
            if key_name in self.key_press_times:
                duration = time.time() - self.key_press_times.pop(key_name)

            self.activity_tensor.update_key(key_name, False)

            if (self.recording and duration is not None
                    and key_name in self.actions and duration > 0.02):
                params = {'duration': float(duration)}
                self.record_action(key_name, from_bot=False, params=params)
        except Exception as e:
            print(f"⚠️ Ошибка обработки release: {e}")
        return True

    def _key_to_name(self, key) -> Optional[str]:
        if hasattr(key, 'char') and key.char:
            return key.char.lower()
        mapping = {
            Key.space: 'space',
            Key.enter: 'enter',
            Key.esc: 'escape',
            Key.tab: 'tab',
            Key.backspace: 'backspace',
            Key.shift: 'shift',
            Key.ctrl: 'ctrl',
            Key.alt: 'alt',
            Key.up: 'up',
            Key.down: 'down',
            Key.left: 'left',
            Key.right: 'right',
            Key.f1: 'f1', Key.f2: 'f2', Key.f3: 'f3', Key.f4: 'f4',
            Key.f5: 'f5', Key.f6: 'f6', Key.f7: 'f7', Key.f8: 'f8',
            Key.f9: 'f9', Key.f10: 'f10', Key.f11: 'f11', Key.f12: 'f12',
            Key.home: 'home', Key.end: 'end',
            Key.page_up: 'page_up', Key.page_down: 'page_down',
            Key.insert: 'insert', Key.delete: 'delete',
            Key.caps_lock: 'capslock',
            Key.num_lock: 'numlock',
            Key.scroll_lock: 'scrolllock',
            Key.pause: 'pause',
            Key.print_screen: 'print_screen',
            Key.cmd: 'win_left',
        }
        return mapping.get(key, None)

    def start_listeners(self):
        mouse_listener = MouseListener(
            on_move=self.on_mouse_move,
            on_click=self.on_mouse_click,
        )
        mouse_listener.daemon = True
        mouse_listener.start()

        keyboard_listener = KeyboardListener(
            on_press=self.on_key_press,
            on_release=self.on_key_release,
        )
        keyboard_listener.daemon = True
        keyboard_listener.start()
        print("✅ Слушатели запущены")

    def show_ai_vision(self):
        h, w = self.tokenizer.get_dimensions()
        channels = 3 if self.tokenizer.use_rgb else 1
        print("\n" + "=" * 80)
        print(f"📸 ЧТО ВИДИТ ИИ ({'RGB' if self.tokenizer.use_rgb else 'GRAY'}"
              f"{' FOVEATED' if getattr(self.tokenizer, 'foveated', False) else ''})")
        print("=" * 80)

        frame = self.capture_screen()
        if frame is None:
            print("❌ Не удалось захватить экран")
            return

        tokens = self.tokenizer.encode_frame(frame, [])

        if getattr(self.tokenizer, 'foveated', False):
            print(f"\n👁️  ФОВЕАЛЬНАЯ ТОКЕНИЗАЦИЯ ({len(self.tokenizer.zones)} зон)")
            offset = 0
            for i, (zh, zw, area) in enumerate(self.tokenizer.zones):
                zone_len = self.tokenizer._zone_lens[i]
                zone_data = tokens[offset:offset + zone_len]
                offset += zone_len
                if self.tokenizer.use_rgb:
                    grid = np.array(zone_data).reshape(zh, zw, 3)
                    brightness = np.mean(grid, axis=2).astype(np.uint8)
                    r_mean = grid[:, :, 0].mean()
                    g_mean = grid[:, :, 1].mean()
                    b_mean = grid[:, :, 2].mean()
                    print(f"\n  Зона {i + 1}: {zw}×{zh} | площадь экрана: {area:.0%}")
                    print(f"    Среднее RGB: ({r_mean:.1f}, {g_mean:.1f}, {b_mean:.1f})")
                else:
                    grid = np.array(zone_data).reshape(zh, zw)
                    brightness = grid
                    print(f"\n  Зона {i + 1}: {zw}×{zh} | площадь: {area:.0%} | "
                          f"яркость: {grid.mean():.1f}")

                print(f"    МАТРИЦА (первые 16 столбцов):")
                print("      " + " ".join(f"{x:3d}" for x in range(min(16, zw))))
                print("      " + "-" * (4 * min(16, zw)))
                for y in range(min(zh, 20)):
                    row_str = f"    {y:2d}| "
                    for x in range(min(16, zw)):
                        val = brightness[y, x]
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
        else:
            if self.tokenizer.use_rgb:
                grid = np.array(tokens[:self.tokenizer.total_cells]).reshape(h, w, 3)
                print(f"\n📐 Размер сетки: {w}×{h}×{channels} = {self.tokenizer.total_cells}")
                r_mean, g_mean, b_mean = (grid[:, :, 0].mean(),
                                          grid[:, :, 1].mean(),
                                          grid[:, :, 2].mean())
                print(f"📊 Среднее RGB: ({r_mean:.1f}, {g_mean:.1f}, {b_mean:.1f})")
                brightness = np.mean(grid, axis=2).astype(np.uint8)
            else:
                grid = np.array(tokens[:self.tokenizer.total_cells]).reshape(h, w)
                brightness = grid
                print(f"\n📐 Размер сетки: {w}×{h} = {self.tokenizer.total_cells}")
                print(f"📊 Диапазон яркости: {grid.min():.0f} - {grid.max():.0f}")
                print(f"📈 Средняя яркость: {grid.mean():.1f}")

            print(f"📉 Стандартное отклонение: {brightness.std():.1f}")

            print("\n🔢 МАТРИЦА ЯРКОСТИ (первые 16 столбцов):")
            print("   " + " ".join(f"{i:3d}" for i in range(min(16, w))))
            print("   " + "-" * (4 * min(16, w)))
            for y in range(h):
                row_str = f"{y:2d}| "
                for x in range(min(16, w)):
                    val = brightness[y, x]
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

        act_summary = self.activity_tensor.get_activity_summary()
        print(f"\n🎮 АКТИВНОСТЬ:")
        print(f"  Активные клавиши: {act_summary['active_keys'][:10]}")
        print(f"  Активные кнопки мыши: {act_summary['active_mouse']}")
        print(f"  Позиция мыши: {act_summary['mouse_position']}")
        print(f"  Всего нажатий: {act_summary['total_key_presses']}")

        res_state = self.forest.get_resonance_state()
        if res_state is not None:
            print(f"\n🎵 РЕЗОНАНС ВЕТВЕЙ:")
            for i, val in enumerate(res_state):
                bar = "█" * int(abs(val) * 20)
                sign = "+" if val > 0 else "-"
                print(f"  Ветвь {i:2d}: {sign}{bar} {val:+.3f}")

        if self.forest.is_trained and self.current_state:
            try:
                n_classes = len(self.actions)
                probas = self.forest.predict_proba(self.current_state, n_classes=n_classes)
                top_indices = np.argsort(probas)[-5:][::-1]
                print(f"\n🤖 ТОП-5 ДЕЙСТВИЙ:")
                for idx in top_indices:
                    action_name = self.actions[idx] if idx < len(self.actions) else "unknown"
                    print(f"  {action_name:25} → {probas[idx]:.3f}")
            except Exception as e:
                print(f"⚠️ Ошибка прогноза: {e}")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"ai_vision_{timestamp}_original.png", frame)
        vis_grid = self.tokenizer.visualize_grid(tokens)
        cv2.imwrite(f"ai_vision_{timestamp}_tokens.png", vis_grid)

        print(f"\n💾 Сохранено: ai_vision_{timestamp}_*.png")
        print("=" * 80)

    def show_stats(self):
        h, w = self.tokenizer.get_dimensions()
        mem_stats = self.memory.get_statistics()
        param_stats = self.parameter_memory.get_statistics()
        print("\n📊 СТАТИСТИКА:")
        print(f"  Размер сетки: {w}×{h} = {h * w} пикселей")
        print(f"  Память: {mem_stats['size']}/{mem_stats['max_size']} "
              f"({mem_stats['fill_ratio']:.1%})")
        print(f"    Средняя сила: {mem_stats.get('avg_strength', 0):.3f}")
        print(f"    Мин/Макс сила: {mem_stats.get('min_strength', 0):.3f}/"
              f"{mem_stats.get('max_strength', 0):.3f}")
        print(f"    Слабых: {mem_stats.get('weak', 0)} | "
              f"Сильных: {mem_stats.get('strong', 0)} | "
              f"Уникальных состояний: {mem_stats.get('unique_states', 0)} | "
              f"С параметрами: {mem_stats.get('with_params', 0)}")
        print(f"    Удалено на последнем шаге: {mem_stats.get('last_pruned', 0)}")

        print(f"\n🎛️  ПАМЯТЬ ПАРАМЕТРОВ:")
        print(f"  Записей: {param_stats['size']}")
        print(f"  Средний sub_reward: {param_stats['avg_sub_reward']:+.3f}")
        print(f"  Положительных: {param_stats['positive']}")
        print(f"  Отрицательных: {param_stats['negative']}")
        print(f"  Уникальных действий: {param_stats['unique_actions']}")

        print(f"\n  Лес обучен: {self.forest.is_trained}")
        print(f"  Деревьев: {len(self.forest.trees)}")
        print(f"  Всего действий: {self.stats['total_actions']}")
        print(f"  Успешных: {self.stats['successful_actions']}")
        print(f"  Мышиных: {self.stats['mouse_actions']}")
        print(f"  Клавиатурных: {self.stats['keyboard_actions']}")
        print(f"  Комбинаций: {self.stats['combo_actions']}")
        print(f"  Заблокировано: {len(self.blocked_actions)}")
        if self.stats['total_actions'] > 0:
            print(f"  Успешность: "
                  f"{self.stats['successful_actions'] / self.stats['total_actions']:.1%}")
        print(f"  Средняя награда: {self.stats['average_reward']:.2f}")
        print(f"  Последнее действие: {self.stats['last_action']}")
        print(f"  Состояние бота: {self.bot_state}")
        print(f"  Уникальных состояний: {len(self.state_visits)}")
        print(f"  Фаза обучения: {self.learning_phase}")

        if self.forest.is_trained:
            accuracy = self.evaluate_model()
            print(f"  Hit-rate top-1: {accuracy:.1%}")
            branch_stats = self.forest.get_branch_statistics()
            print(f"\n⚖️  СТАТИСТИКА ВЕТВЕЙ:")
            print(f"  Активных весов: {branch_stats['total_connections']}")
            print(f"  Средний вес: {branch_stats['avg_weight']:.3f}")
            print(f"  Мин/Макс вес: {branch_stats['min_weight']:.3f}/"
                  f"{branch_stats['max_weight']:.3f}")
            print(f"  Глобальные веса: {branch_stats['global_weights']}")

            if 'resonance' in branch_stats:
                res = branch_stats['resonance']
                print(f"\n🎵 РЕЗОНАНС ВЕТВЕЙ:")
                print(f"  Средняя связь: {res['mean']:+.3f}")
                print(f"  Ст. отклонение: {res['std']:.3f}")
                print(f"  Усиливающих связей: {res['positive']}")
                print(f"  Подавляющих связей: {res['negative']}")
                print(f"  Baseline награды: {res['baseline']:+.3f}")

        reward_stats = self.reward_system.get_statistics()
        print(f"\n🎯 СТАТИСТИКА НАГРАД:")
        print(f"  Средняя награда: {reward_stats['avg_reward']:.2f}")
        print(f"  Ст. отклонение: {reward_stats['std_reward']:.2f}")
        print(f"  Провалов подряд: {reward_stats['consecutive_failures']}")
        print(f"  Разнообразие: {reward_stats['action_diversity']}/20")
        print(f"  Лучшее действие: {reward_stats['best_action']} "
              f"({reward_stats['best_action_rate']:.1%})")

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
        X, y = self.memory.get_all()
        data = {
            'training_data': X,
            'training_labels': y,
            'forest': self.forest,
            'parameter_memory': self.parameter_memory,
            'stats': dict(self.stats),
            'actions': list(self.actions),
            'combo_actions': list(self.combo_actions),
            'blocked_actions': list(self.blocked_actions),
            'timestamp': time.time(),
        }
        try:
            with open(self.model_file, 'wb') as f:
                pickle.dump(data, f)
            print(f"💾 Модель сохранена (память: {len(self.memory)}, "
                  f"параметры: {len(self.parameter_memory)})")
        except Exception as e:
            print(f"⚠️ Ошибка сохранения: {e}")

    def load_model(self):
        if not os.path.exists(self.model_file):
            return
        try:
            with open(self.model_file, 'rb') as f:
                data = pickle.load(f)

            self.memory.clear()
            X = data.get('training_data', [])
            y = data.get('training_labels', [])
            for s, l in zip(X, y):
                self.memory.add(s, l, reward=0.0, branch_weight=0.0)

            if 'forest' in data:
                self.forest = data['forest']
                self.forest._pred_cache = {}
                self.forest._pred_cache_max = 256

            if 'parameter_memory' in data:
                loaded_pm = data['parameter_memory']
                if isinstance(loaded_pm, ParameterMemory):
                    self.parameter_memory = loaded_pm
                    self.executor.parameter_memory = loaded_pm

            if 'stats' in data:
                for key, value in data['stats'].items():
                    self.stats[key] = value
            if 'combo_actions' in data:
                self.combo_actions = data['combo_actions']
            if 'blocked_actions' in data:
                self.blocked_actions = data['blocked_actions']

            print(f"📂 Модель загружена (память: {len(self.memory)}, "
                  f"параметры: {len(self.parameter_memory)})")
        except Exception as e:
            print(f"⚠️ Ошибка загрузки модели: {e}")

    def clear_memory(self):
        self.memory.clear()
        self.parameter_memory.clear()
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
            'combo_actions': 0,
        }
        self.state_visits.clear()
        self.novelty_decay = 0.99999
        self.blocked_actions = []
        self.reward_system = SmartRewardSystem()
        self.learning_phase = 'exploration'
        self.phase_switches = 0
        print("🧹 Память очищена")

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

                new_state = self.encode_state(frame, [])
                with self.state_lock:
                    self.current_state = new_state

                self.memory.step()
                self.parameter_memory.step()

                if self.auto_mode:
                    if time.time() - self.last_mouse_move > MOUSE_IDLE_THRESHOLD:
                        self.update_learning_phase()
                        action, confidence = self.select_action_with_phase(self.current_state)

                        action_idx = (self.actions.index(action)
                                      if action in self.actions else 0)
                        state_hash = self.memory._hash_state(self.current_state)

                        self.executor.set_context(
                            state=self.current_state,
                            state_hash=state_hash,
                            action=action,
                            action_idx=action_idx,
                        )

                        executed = self.execute_action(action)

                        self.stats['total_actions'] += 1
                        self.stats['last_action'] = action

                        if executed:
                            time.sleep(0.2)
                            new_frame = self.capture_screen()
                            after_state = self.encode_state(new_frame, [])

                            reward = self.calculate_adaptive_reward(
                                self.current_state, after_state, action
                            )
                            self.recent_rewards.append(reward)
                            self.stats['average_reward'] = (
                                np.mean(list(self.recent_rewards))
                                if self.recent_rewards else 0
                            )

                            self.parameter_memory.reinforce(
                                state_hash, action_idx,
                                sub_reward=reward, weight=0.5,
                            )

                            last_params = self.executor.get_last_params()

                            self.record_action(
                                action, from_bot=True, reward=reward,
                                params=last_params if last_params else None,
                            )

                            try:
                                if self.forest._last_paths is not None:
                                    self.forest.update_weights_online(reward)
                                else:
                                    self.forest.update_weights_online(
                                        reward, path=[], tree_idx=0
                                    )
                            except Exception as e:
                                print(f"⚠️ update_weights_online error: {e}")

                            feedback = self.reward_system.get_action_feedback(action)
                            stats = self.reward_system.get_statistics()
                            params_str = ""
                            if last_params:
                                parts = []
                                if 'x' in last_params:
                                    parts.append(f"x={int(last_params['x'])}")
                                if 'y' in last_params:
                                    parts.append(f"y={int(last_params['y'])}")
                                if 'duration' in last_params:
                                    parts.append(f"d={last_params['duration']:.2f}s")
                                if 'distance' in last_params and last_params['distance'] > 1:
                                    parts.append(f"dist={int(last_params['distance'])}")
                                if parts:
                                    params_str = " | " + " ".join(parts)

                            print(
                                f"🎮 {action:25} | "
                                f"Награда: {reward:6.2f} | "
                                f"Уверенность: {confidence:.2f} | "
                                f"Фаза: {self.learning_phase[:3]} | "
                                f"Память: {len(self.memory)}/{self.memory.max_size}"
                                f"{params_str} | "
                                f"{feedback}"
                            )
                            if self.stats['total_actions'] % 10 == 0:
                                mem_stats = self.memory.get_statistics()
                                param_stats = self.parameter_memory.get_statistics()
                                print(
                                    f"📊 Средняя награда: {stats['avg_reward']:.2f} | "
                                    f"Разнообразие: {stats['action_diversity']}/20 | "
                                    f"Лучшее: {stats['best_action']} "
                                    f"({stats['best_action_rate']:.1%}) | "
                                    f"Сила памяти: {mem_stats['avg_strength']:.2f} | "
                                    f"Параметры: {param_stats['size']} "
                                    f"(sr: {param_stats['avg_sub_reward']:+.2f})"
                                )

                            with self.state_lock:
                                self.current_state = after_state
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
