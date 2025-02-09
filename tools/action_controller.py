import time
from pynput.mouse import Button, Controller as MouseController
from pynput.keyboard import Controller as KeyboardController
# from state import State

class ActionController:
    def __init__(self):
        self.keyboard = KeyboardController()
        self.mouse = MouseController()
        self.attack_step = 0
        self.is_attacking = False
        self.attack_time = None
        self.attack_offset = 0
        self.is_dodge = False
        self.step_list = []
        self.capture_time = 0.25
        self.attack_step_list = [
            { 'wait': 0.25, 'pre_attack_wait': 0, 'pre_dodge_wait': 0 },
            { 'wait': 0.4, 'pre_attack_wait': 0, 'pre_dodge_wait': 0.13 },
            { 'wait': 0.3, 'pre_attack_wait': 0.45, 'pre_dodge_wait': 0 },
            { 'wait': 0.35, 'pre_attack_wait': 0.25, 'pre_dodge_wait': 0.1 },
            { 'wait': 0.65, 'pre_attack_wait': 1.1, 'pre_dodge_wait': 0.3 },
            { 'wait': 0.8, 'pre_attack_wait': 0, 'pre_dodge_wait': 0 },
        ]
        self.old_attack_step_list = [
            {'wait': 0.25, 'next_dodge': 0.38, 'next_shipo': 0.25, 'next_attack': 0.25, 'next_shipo_last': 0.75, 'next_attack_last': 0.75},
            {'wait': 0.4, 'next_dodge': 0.4, 'next_shipo': 0.55, 'next_attack': 0.85, 'next_shipo_last': 1.3, 'next_attack_last': 1},
            {'wait': 0.35, 'next_dodge': 0.45, 'next_shipo': 0.35, 'next_attack': 0.5, 'next_shipo_last': 1.05, 'next_attack_last': 1.05},
            {'wait': 0.35, 'next_dodge': 0.35, 'next_shipo': 0.65, 'next_attack': 1.4, 'next_shipo_last': 2, 'next_attack_last': 1.6},
            {'wait': 1.05, 'next_dodge': 0.35, 'next_shipo': -1, 'next_attack': 1.85, 'next_shipo_last': -1, 'next_attack_last': -1},
        ]

    def left_dodge(self):
        # 翻滚后攻击0.25s 翻滚后翻滚0.3s
        if self.is_attacking:
            distance_time = time.time() - self.attack_time
            attack = self.attack_step_list[self.attack_step]
            if attack['next_shipo_last'] > distance_time:
                time.sleep(abs(attack['next_shipo_last'] - distance_time))
                # time.sleep(attack['next_shipo_last'] - distance_time)
                self.attack_offset = 0.3
        self.keyboard.press('a')
        self.keyboard.press(' ')
        time.sleep(0.05)
        self.keyboard.release(' ')
        self.keyboard.release('a')
        time.sleep(0.1)

    def dodge(self, step, direction):
        dodge_sleep = 0.35
        if step != 0 and not self.is_dodge:
            step = step % len(self.attack_step_list)
            current = self.attack_step_list[step]
            wait = current['pre_dodge_wait']
            time.sleep(wait)
        self.keyboard.press(direction)
        self.keyboard.press(' ')
        time.sleep(0.05)
        self.keyboard.release(' ')
        self.keyboard.release(direction)
        self.is_dodge = True
        time.sleep(dodge_sleep - self.capture_time)

    def counter_attack(self):
        self.mouse.click(Button.right)
        time.sleep(0.5)
        # 识破 1.25s可翻滚  1.7s接轻棍
        # 固定后摇1.25   衔接轻棍前摇0.45

    def feng_chuan_hua(self):
        self.keyboard.press('c')
        time.sleep(0.05)
        self.keyboard.release('c')
        # 凤穿花 0.6s之前翻滚取消  1s之后可翻滚
        self.left_dodge()
        self.counter_attack()
        time.sleep(1)
        self.keyboard.press('z')
        time.sleep(0.05)
        self.keyboard.release('z')

    def attack_old(self):
        attack_step_len = len(self.attack_step_list)
        previous_step, previous, current = None, None, None
        attack_offset_temp = self.attack_offset
        if self.attack_offset:
            time.sleep(self.attack_offset)
            self.attack_offset = 0
        # print('------------start---------------')
        if self.is_attacking:
            previous_step = (self.attack_step + attack_step_len) % attack_step_len
            print(f'previous_step: {previous_step}')
            previous = self.attack_step_list[previous_step]
            is_in_attack = (time.time() - self.attack_time + attack_offset_temp) < previous['next_attack_last']
            # print('攻击连段' if is_in_attack else '攻击中断')
            if not is_in_attack:
                self.attack_step = 0
            else:
                print(f"wait{(previous['next_attack'] - previous['wait']):f}")
                time.sleep(previous['next_attack'] - previous['wait'])

        self.attack_step = (self.attack_step + 1) % attack_step_len
        if self.attack_step == 1: self.is_attacking = True
        current = self.attack_step_list[self.attack_step]
        self.mouse.click(Button.left)
        self.attack_time = time.time() 
        time.sleep(current['wait'])
        # time.sleep(0.2)
        # print(f'wait{current["wait"]:f}')
        # print('------------end---------------')
        # print('\n')

    def forward(self):
        self.keyboard.press('w')
        time.sleep(0.5)
    
    def forward_dodge(self):
        if self.is_attacking:
            distance_time = time.time() - self.attack_time
            attack = self.attack_step_list[self.attack_step]
            if attack['next_shipo_last'] > distance_time:
                time.sleep(attack['next_shipo_last'] - distance_time)
                self.attack_offset = 0.3
        self.keyboard.press('w')
        self.keyboard.press(' ')
        time.sleep(0.05)
        self.keyboard.release(' ')
        self.keyboard.release('w')
        time.sleep(0.1)

    def release_all(self):
        self.keyboard.release('w')

    def health(self):
        self.dodge(1, 's')
        time.sleep(0.3)
        self.keyboard.press('r')
        time.sleep(0.1)
        self.keyboard.release('r')
        time.sleep(1.6 - self.capture_time)

    def attack(self, step):
        step = step % len(self.attack_step_list)
        action = self.attack_step_list[step]
        pre_swing = action['pre_attack_wait']
        time.sleep(0.15 if self.is_dodge else max(pre_swing, 0.01))
        self.mouse.click(Button.left)
        post_swing = action['wait']
        time.sleep(post_swing - self.capture_time)
        self.is_dodge = False
    
# if __name__ == '__main__':
#     time.sleep(2)
#     ac = ActionController()
#     state = State()

#     step = state.read_attack_step()
#     ac.attack(step)
#     step = state.read_attack_step()
#     ac.attack(step)
#     step = state.read_attack_step()
#     ac.attack(step)
#     step = state.read_attack_step()
#     ac.attack(step)
#     step = state.read_attack_step()
#     ac.attack(step)