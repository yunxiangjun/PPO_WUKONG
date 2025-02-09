import pygetwindow as gw

WUKONG_TITLE = 'b1  ' #进程名称
WUKONG_CLASS_NAME = 'UnrealWIndow' #进程类名称

# 矫正窗口
def correction_window():
    try:
        window = gw.getWindowsWithTitle(WUKONG_TITLE)[0]
        if not window.visible or not window.isActive:
            window.restore()    #恢复窗口
            window.activate()   #激活窗口
        window.moveTo(-8, 0)    #移动至左上角
    except:
        print(f'{WUKONG_TITLE} is not find. 或尝试用管理员权限运行')