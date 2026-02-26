import sys
import os

import cap


# 静默输出重定向
class QuietStream:
    def write(self, text):
        pass

    def flush(self):
        pass


# 屏蔽标准输出和标准错误，拦截库内部日志
sys_stdout = sys.stdout
sys_stderr = sys.stderr
sys.stdout = QuietStream()
sys.stderr = QuietStream()

import cv2
import numpy as np
import serial
import re

# 高版本NumPy兼容补丁
np.int = int
np.float = float
np.bool = bool
import imutils
from hyperlpr import HyperLPR_plate_recognition

# ====================== 全局配置 ======================
CAMERA_INDEX = 1  # USB摄像头索引
FRAME_WIDTH = 600
# 车牌检测参数
MIN_ASPECT_RATIO = 2.5
MAX_ASPECT_RATIO = 6.0
MIN_AREA = 1000
LOWER_BLUE = np.array([95, 30, 30])
UPPER_BLUE = np.array([135, 255, 255])

DETECT_COUNTER_THRESH = 1
# 串口通信配置
SERIAL_PORT = "COM6"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1
# 识别间隔（帧）
RECOGNIZE_INTERVAL = 5



# ====================== 白名单与校验配置 ======================
PLATE_WHITELIST = {"京A84523", "沪B67890", "粤GSB250", "京AD12345"}
PROVINCE_CODES = {"京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", "皖", "鲁",
                  "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋", "蒙", "陕", "吉", "闽",
                  "贵", "粤", "青", "藏", "川", "宁", "琼"}
PLATE_PATTERN = re.compile(r'^(' + '|'.join(PROVINCE_CODES) + r')[A-Z]{1}[A-Z0-9]{5,6}$')
# 串口指令配置
ALLOW_PASS_CMD = "ALLOW"  # 允许通行指令
DENY_PASS_CMD = "DENY"  # 禁止通行指令
SERIAL_DELIMITER = ","  # 指令和车牌号的分隔符

# ====================== 新增：已发送车牌缓存 ======================
SENT_PLATES = set()  # 存储已发送过指令的车牌，确保只发一次
CLEAR_CACHE_INTERVAL = 30  # 缓存清空间隔（秒），防止内存溢出
last_cache_clear = 0  # 上次清空缓存的时间戳


# ====================== 图像预处理函数 ======================
def preprocess_for_ocr(plate_roi):
    if plate_roi is None or plate_roi.size == 0:
        return None
    plate = imutils.resize(plate_roi, width=280)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    _, binary = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)


# ====================== 车牌识别函数 ======================
def recognize_plate_number(processed_plate):
    if processed_plate is None:
        return ""
    try:
        result = HyperLPR_plate_recognition(processed_plate)
        if result:
            plate_num, confidence, _ = result[0]
            plate_num = plate_num.strip().replace(" ", "").upper()
            return plate_num if confidence > 0.5 else ""
        return ""
    except Exception:
        return ""


# ====================== 白名单+格式校验函数 ======================
def is_plate_authorized(plate_str):
    if not plate_str:
        return False
    if not PLATE_PATTERN.match(plate_str):
        return False
    return plate_str in PLATE_WHITELIST


# ====================== 串口发送函数（优化：仅发送未发送过的车牌） ======================
def send_serial_data(serial_port, plate_num, pass_allowed):
    """
    向串口发送车牌号和通行指令（同一车牌仅发送一次）
    :param serial_port: 串口对象
    :param plate_num: 识别到的车牌号
    :param pass_allowed: 是否允许通行（True/False）
    """
    # 校验前置条件：串口可用、车牌有效、未发送过
    if not serial_port or not serial_port.is_open or not plate_num:
        return

    # 核心逻辑：同一车牌只发送一次
    if plate_num in SENT_PLATES:
        safe_print(f" 车牌{plate_num}已发送过指令，跳过")
        return

    # 构造发送数据：指令,车牌号\n
    cmd = ALLOW_PASS_CMD if pass_allowed else DENY_PASS_CMD
    send_data = f"{cmd}{SERIAL_DELIMITER}{plate_num}\n".encode("utf-8")

    try:
        serial_port.write(send_data)
        SENT_PLATES.add(plate_num)  # 发送成功后加入缓存
        safe_print(f" 串口发送：{cmd} {plate_num}")
    except Exception as e:
        safe_print(f" 串口发送失败：{str(e)}")


# ====================== 新增：缓存清理函数 ======================
def clear_sent_plates_cache():
    """定期清空已发送车牌缓存，避免内存占用过大"""
    global last_cache_clear
    current_time = cv2.getTickCount() / cv2.getTickFrequency()  # 获取当前时间（秒）
    # 每隔 CLEAR_CACHE_INTERVAL 秒清空一次缓存
    if current_time - last_cache_clear > CLEAR_CACHE_INTERVAL:
        SENT_PLATES.clear()
        last_cache_clear = current_time
        safe_print(f" 已清空车牌缓存，当前时间：{current_time:.1f}s")


# ====================== 基础图像处理函数 ======================
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)


def filter_blue_plate_region(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_BLUE, UPPER_BLUE)
    return cv2.dilate(mask, None, iterations=2)


# ====================== 车牌轮廓检测 ======================
def find_license_plate_contours(edged, mask, frame):
    combined = cv2.bitwise_and(edged, edged, mask=mask)
    cnts = cv2.findContours(combined.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:20]

    best_plate = None
    best_location = None
    for c in cnts:
        if cv2.contourArea(c) < MIN_AREA:
            continue
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.03 * perimeter, True)
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = w / float(h)
        if 4 <= len(approx) <= 6 and MIN_ASPECT_RATIO <= aspect_ratio <= MAX_ASPECT_RATIO:
            best_plate = frame[y:y + h, x:x + w]
            best_location = (x, y, w, h)
            break
    return best_plate, best_location


# ====================== ROI有效性校验 ======================
def validate_plate(plate_roi):
    if plate_roi is None:
        return False
    h, w = plate_roi.shape[:2]
    if w < 50 or h < 15:
        return False
    gray_plate = cv2.cvtColor(plate_roi, cv2.COLOR_BGR2GRAY)
    _, binary_plate = cv2.threshold(gray_plate, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    white_ratio = np.sum(binary_plate == 255) / binary_plate.size
    return 0.10 < white_ratio < 0.70


# ====================== 打印工具函数：临时恢复输出 ======================
def safe_print(msg):
    sys.stdout = sys_stdout
    sys.stderr = sys_stderr
    print(msg)
    sys.stdout = QuietStream()
    sys.stderr = QuietStream()


# ====================== 主程序 ======================
def main():
    # 初始化全局变量
    global last_cache_clear
    last_cache_clear = cv2.getTickCount() / cv2.getTickFrequency()  # 初始化缓存清空时间

    # 初始化串口
    arduino_serial = None
    try:
        arduino_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        safe_print(f" 串口打开成功：{SERIAL_PORT}")
    except Exception as e:
        safe_print(f" 串口打开失败：{str(e)}")

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # 强制指定摄像头分辨率
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        safe_print(f"错误：无法打开索引为 {CAMERA_INDEX} 的摄像头")
        safe_print(" 请尝试修改 CAMERA_INDEX 为 0、2、3 后重试")
        return

    safe_print("摄像头已启动，按 `q` 键退出程序")
    detect_counter = 0
    frame_counter = 0
    cached_plate_num = ""

    while True:
        # 定期清理已发送车牌缓存
        clear_sent_plates_cache()

        ret, frame = cap.read()
        if not ret:
            safe_print("错误：无法读取画面")
            break

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        output_frame = frame.copy()
        frame_counter += 1

        # 车牌区域检测
        edged = preprocess_image(frame)
        blue_mask = filter_blue_plate_region(frame)
        plate_roi, plate_loc = find_license_plate_contours(edged, blue_mask, frame)
        is_valid = validate_plate(plate_roi)

        # 防抖逻辑
        detect_counter = detect_counter + 1 if is_valid else 0
        is_plate_detected = detect_counter >= DETECT_COUNTER_THRESH

        # 持续识别逻辑
        if is_plate_detected and plate_roi is not None and frame_counter % RECOGNIZE_INTERVAL == 0:
            processed_plate = preprocess_for_ocr(plate_roi)
            current_plate = recognize_plate_number(processed_plate)

            # 只有识别到有效车牌时才处理
            if current_plate:
                cached_plate_num = current_plate

                # 显示ROI
                if processed_plate is not None:
                    cv2.imshow("Plate ROI", processed_plate)
                    cv2.waitKey(1)

                # 校验是否在白名单并发送指令（核心：同一车牌只发一次）
                is_authorized = is_plate_authorized(current_plate)
                send_serial_data(arduino_serial, current_plate, is_authorized)

                # 打印日志
                if is_authorized:
                    safe_print(f" 合法车牌：{current_plate} - 允许通行")
                else:
                    safe_print(f" 非法车牌：{current_plate} - 禁止通行")
            else:
                # 未识别到有效车牌
                cached_plate_num = ""
                if cv2.getWindowProperty("Plate ROI", cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow("Plate ROI")
        else:
            # 未检测到车牌区域
            cached_plate_num = ""
            if cv2.getWindowProperty("Plate ROI", cv2.WND_PROP_VISIBLE) >= 0:
                cv2.destroyWindow("Plate ROI")

        # 界面绘制逻辑
        if is_plate_detected and plate_loc is not None:
            x, y, w, h = plate_loc
            # 根据是否授权显示不同颜色的框和文字
            if cached_plate_num and is_plate_authorized(cached_plate_num):
                color = (0, 255, 0)  # 绿色：允许通行
                text = f"Plate: {cached_plate_num} (ALLOWED)"
            else:
                color = (0, 0, 255)  # 红色：禁止通行
                text = f"Plate: {cached_plate_num} (DENIED)" if cached_plate_num else "Detecting Plate..."

            cv2.rectangle(output_frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        else:
            cached_plate_num = ""
            cv2.putText(output_frame, "No Plate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示主画面
        cv2.imshow("License Plate Recognition", output_frame)
        # 退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 资源释放
    cap.release()
    if arduino_serial and arduino_serial.is_open:
        arduino_serial.close()
        safe_print("串口已关闭")
    cv2.destroyAllWindows()
    safe_print("程序已退出")


if __name__ == "__main__":
    main()