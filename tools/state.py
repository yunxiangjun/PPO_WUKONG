from pymem import Pymem
from pymem.process import module_from_name
import numpy as np
import time
import traceback
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

keyboard = KeyboardController()
mouse = MouseController()

class State:
    def __init__(self, boss_name='guangzhi'):
        # 连接到进程
        self.pm = Pymem("b1-Win64-Shipping.exe")
        # 获取模块基址
        self.module = module_from_name(self.pm.process_handle, "b1-Win64-Shipping.exe")
        # 游戏是否在加载界面
        self.game_loading_scene = True
        self.fighting_scene = False
        self.w_press = False
        self.base_address = self.module.lpBaseOfDll
        self.boss_name = boss_name
        self.boss_hp_list = {
            'guangzhi': { 'address': 0x1DE4D518, 'offsets': [0x8 ,0x18 ,0x48 ,0xA0 ,0x78 ,0x38 ,0xA0 ,0x10 ,0x108] }
        }
        self.boss_position_list = {
            'guangzhi': { 'address': 0x1D8AC018, 'offsets': [0x10, 0x18, 0x2B0, 0x58, 0x284] }
        }
        self.step_list = []
    def getOffsetAddress(self, address, offset):
        # 按照偏移链依次读取
        try:
            for offset in offset:
                address = self.pm.read_longlong(address) + offset
        except Exception as e:
            print(f'计算偏移地址异常: {traceback.extract_tb(e.__traceback__)}')
            address = 0xFFFFFFFFFFFFFFFF
        return address
    
    def safe_read_float(self, address):
        try:
            return self.pm.read_float(address)
        except:
            print(f'获取地址{address}失败')
            return -1
    
    def safe_read_bytes(self, address):
        try:
            return int.from_bytes(self.pm.read_bytes(address, 4), byteorder='little')
        except:
            print(f'获取地址{address}失败')
            return -1

    #自身生命
    def read_self_hp(self):
        # 计算初始地址
        address = self.base_address + 0x1DE4D518
        offsets = [0x8, 0x18, 0x30, 0x238, 0x2C8, 0x30, 0x28, 0x18, 0x38, 0x27C]
        address_bak = self.base_address + 0x1DE7F308
        offsets_bak = [0x8, 0x18, 0x30, 0x268, 0xF0, 0x2B0, 0x2C0, 0x108]

        # 读取最终地址的单精度浮点数值
        value = self.safe_read_float(self.getOffsetAddress(address, offsets))
        if value == -1 or value == 0 or value < 0.0001:
            value = self.safe_read_float(self.getOffsetAddress(address_bak, offsets_bak))
        return value
    def set_self_hp(self, hp):
        # 计算初始地址
        address = self.base_address + 0x1DE4D518
        offsets = [0x8, 0x18, 0x30, 0x238, 0x2C8, 0x30, 0x28, 0x18, 0x38, 0x27C]
        # 读取最终地址的单精度浮点数值
        self.pm.write_float(self.getOffsetAddress(address, offsets), float(hp))
    # 自身棍势
    def read_self_gunshi(self):
        address = self.base_address + 0x1DE4D5D0
        offsets = [0x0, 0x18, 0x30, 0x1A0, 0x28, 0x40, 0x90, 0xB0, 0x10, 0x31C]
        address_bak = self.base_address + 0x1DE4D418
        offsets_bak = [0x0, 0x18, 0x30, 0x260, 0x310, 0x30, 0x28, 0x18, 0x38, 0x31C]
        value = self.safe_read_float(self.getOffsetAddress(address, offsets))
        if value == -1 or (value < 0.0001 and value != 0):
            value = self.safe_read_float(self.getOffsetAddress(address_bak, offsets_bak))
        return value
    
    # 自身体力
    def read_self_tili(self):
        address = self.base_address + 0x1DE7F308
        offsets = [0x8, 0x18, 0x30, 0x230, 0x2C8, 0x30, 0x28, 0x18, 0x38, 0x298]
        address_bak = self.base_address + 0x1DE4D418
        offsets_bak = [0x0, 0x18, 0x30, 0x1A0, 0x28, 0x40, 0x90, 0xB0, 0x10, 0x298]
        value = self.safe_read_float(self.getOffsetAddress(address, offsets))
        if value == -1 or (value < 0.0001 and value != 0):
            value = self.safe_read_float(self.getOffsetAddress(address_bak, offsets_bak))
        return value
    
    # 自身坐标
    def read_self_position(self):
        address = self.base_address + 0x1DC98B98
        offsets = [0x8, 0x190, 0x38, 0x18, 0x88, 0xC0, 0x284]
        zAxisAddress = self.getOffsetAddress(address, offsets)
        yAxisAddress = zAxisAddress - 0x8
        xAxisAddress = zAxisAddress - 0x10
        yAxis = self.safe_read_float(yAxisAddress)
        xAxis = self.safe_read_float(xAxisAddress)
        # return xAxis, yAxis
        return (xAxis, yAxis)
    
    # boss生命
    def read_boss_hp(self):
        boss = self.boss_hp_list.get(self.boss_name)
        address = self.base_address + boss['address']
        value = self.safe_read_float(self.getOffsetAddress(address, boss['offsets']))
        return value
    
    # boss坐标
    def read_boss_position(self):
        boss = self.boss_position_list.get(self.boss_name)
        address = self.base_address + boss['address']
        zAxisAddress = self.getOffsetAddress(address, boss['offsets'])  #5.5  -5.7
        yAxisAddress = zAxisAddress - 0x8  #-6.4 - -6.7
        xAxisAddress = zAxisAddress - 0x10  #6.9-7.1
        zAxis = self.safe_read_float(zAxisAddress)
        yAxis = self.safe_read_float(yAxisAddress)
        xAxis = self.safe_read_float(xAxisAddress)
        return (xAxis, yAxis, zAxis)

    # 攻击段数
    def read_attack_step(self):
        address = self.base_address + 0x1DE4D418
        offsets = [0x0, 0x18, 0xA0, 0xC8, 0x868, 0x78, 0x28, 0x20, 0xB8, 0x30]
        attack_step_add = self.getOffsetAddress(address, offsets)
        attack_step_byte = self.safe_read_bytes(attack_step_add)
        if attack_step_byte == -1:
            return attack_step_byte
        if not attack_step_byte in self.step_list:
            self.step_list.append(attack_step_byte)
        return self.step_list.index(attack_step_byte)

    def get_self_state(self):
        self_hp = self.read_self_hp()
        self_gunshi = self.read_self_gunshi()
        self_tili = self.read_self_tili()
        self_position = self.read_self_position()
        return self_hp, self_gunshi, self_tili, self_position
    
    def get_boss_state(self):
        boss_hp = self.read_boss_hp()
        boss_position = self.read_boss_position()
        return boss_hp, boss_position
    
    def get_custom_features(self):
        self_hp = self.read_self_hp()
        self_gunshi = self.read_self_gunshi()
        self_tili = self.read_self_tili()
        self_pos = self.read_self_position()
        boss_hp = self.read_boss_hp()
        boss_pos = self.read_boss_position()
        return self_hp, self_gunshi, self_tili, boss_hp, self_pos, boss_pos

    def get_game_state(self):
        self_hp = self.read_self_hp()
        boss_hp = self.read_boss_hp()
        if boss_hp > 0 and self_hp > 0:
            self.game_loading_scene = False
            return 1
        elif (self_hp < 0.0001 and self_hp != 0) or (boss_hp < 0.0001 and boss_hp != 0):
            return 3
        else:
            return 2
        # 自身和boss血量同时存在为战斗场景 无需额外操作
        # 否则为战斗结算场景，需要自动化操作重复进入战斗 然后设置loading为true 直到boss和自身血量不为0，其他情况（进入挑战） 手动操作

    def handle_game_restart(self):
        """处理游戏结算与再次挑战逻辑"""
        print('执行再次挑战')
        time.sleep(18)
        self.press_key('e', 0.5)
        self.press_key('e', 0.5)
        self.game_loading_scene = True
        time.sleep(25)

    def handle_game_reload(self):
        """处理游戏结算与加载逻辑"""
        print('执行重新加载')
        time.sleep(18)
        self.press_key(Key.up, 0.5)
        self.press_key('e', 0.5)
        self.press_key(Key.up, 0.5)
        self.press_key('e', 0.5)
        time.sleep(50)
        self.press_key('e', 0.5)
        time.sleep(10)
        self.press_key(Key.up, 0.5)
        self.press_key(Key.up, 0.5)
        self.press_key('e', 0.5)
        self.press_key('e', 0.5)
        time.sleep(1)
        self.press_key('e', 0.5)
        self.press_key('e', 3)
        self.game_loading_scene = True
        # 加载场景
        time.sleep(20)

    def press_key(self, key, press_time=0.1):
        """按下并释放一个键"""
        keyboard.press(key)
        time.sleep(press_time)
        keyboard.release(key)
        
    def handle_w_key(self, distance):
        """处理 W 键的按下和释放"""
        if distance >= 0.025 and not self.w_press:
            keyboard.press('w')
            self.w_press = True
        elif distance < 0.025 and self.w_press:
            keyboard.release('w')
            self.w_press = False

    def game_start_opt(self):
        keyboard.press('1')
        time.sleep(0.1)
        keyboard.release('1')
        mouse.click(Button.middle)
        time.sleep(3)

    def get_current_step(self):
        step_byte = self.read_attack_step()
        if (step_byte in self.step_list) == False: 
            self.step_list.append(step_byte)
        return self.step_list.index(step_byte)
if __name__ == '__main__':
    time.sleep(1)
    state = State('guangzhi')
    state.handle_game_loading()
    # state.press_key(Key.up, 0.5)
    # state = State('guangzhi')
    # while True:
    #     time.sleep(1)
    #     game_state = state.get_game_state()
    #     if (game_state == 2):
    #         state.handle_game_loading()
    #         time.sleep(0.5)
    #         continue
    #     self_hp, self_gunshi, self_tili, self_position = state.get_self_state()
    #     if self_hp is None: continue  # 如果获取失败，跳过本次循环
    #     boss_hp = state.read_boss_hp()
    #     if boss_hp is None: continue  # 如果获取失败，跳过本次循环
    #     print(f'自身生命: {self_hp:.2f}, 自身棍势: {self_gunshi:.2f}, 自身体力: {self_tili:.2f}, 自身坐标: ({self_position[0]:.3f}, {self_position[1]:.3f})')


    #     if (state.game_loading):
    #         print('进入战斗场景')
    #         state.game_loading = False
    #         mouse.click(Button.middle)
    #         state.press_key('1', 1)
            
    #     boss_position = state.read_boss_position()
    #     if boss_position[0] is None: continue  # 如果获取失败，跳过本次循环
    #     print(f'boss生命: {boss_hp:.2f}, boss坐标: ({boss_position[0]:.3f}, {boss_position[1]:.3f})')
        
    #     distance = np.linalg.norm(self_position - boss_position)
    #     print(f"距离boss: {distance:.3f}")
    #     state.handle_w_key(distance)