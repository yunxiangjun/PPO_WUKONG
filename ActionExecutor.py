from tools.action_controller import ActionController
import time
# 动作执行器
class ActionExecutor:
    def __init__(self):
        self.actions = {
            # 0: "FORWARD",
            0: "FORWARD_DODGE",
            1: "LEFT_DODGE",
            2: "ATTACK",
            # 3: "COUNTER_ATTACT",
            3: "FENG_CHUAN_HUA",
            4: "HEALTH",
            # 5: "RIGHT_DODGE",
            # 4: "IDLE"
        }
        self.action_controller = ActionController()
    
    def execute(self, action_idx: int, step) -> str:
        action = self.actions[action_idx]
        self.action_controller.release_all()
        match action:
            case "FORWARD":
                self.action_controller.forward()
            case "FORWARD_DODGE":
                self.action_controller.dodge(step, 'w')
            case "LEFT_DODGE":
                self.action_controller.dodge(step, 'a')
            case "ATTACK":
                self.action_controller.attack(step)
            case "COUNTER_ATTACT":
                self.action_controller.counter_attack()
            case "FENG_CHUAN_HUA":
                self.action_controller.feng_chuan_hua()
            case "IDLE":
                time.sleep(0.3)
            case "HEALTH":
                self.action_controller.health()
            case "RIGHT_DODGE":
                self.action_controller.dodge(step, 'd')
        return action