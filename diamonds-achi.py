import pyautogui
import time

# 预留3秒时间切换到目标窗口
time.sleep(3)
print("开始自动输入，按 Ctrl+C 停止")

try:
    while True:
        pyautogui.typewrite("ZHANGLI")
        # pyautogui.press("enter")  # 需要输入后自动回车则取消注释
        time.sleep(1)
except KeyboardInterrupt:
    print("已停止输入")