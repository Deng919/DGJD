import sys
import os
# 静默输出重定向：屏蔽所有第三方库（HyperLPR）的调试/坐标打印
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
CAMERA_INDEX = 0
FRAME_WIDTH = 600
# 车牌检测参数
MIN_ASPECT_RATIO = 2.0
MAX_ASPECT_RATIO = 7.0
MIN_AREA = 800
LOWER_BLUE = np.array([80, 70, 70])
UPPER_BLUE = np.array([135, 255, 255])
DETECT_COUNTER_THRESH = 1
# 串口通信配置
SERIAL_PORT = "COM6"
BAUD_RATE = 9600
SERIAL_TIMEOUT = 1
# 性能优化：识别间隔（帧）
RECOGNIZE_INTERVAL = 5

# ====================== 白名单与校验配置 ======================
PLATE_WHITELIST = {"京A12345", "沪B67890", "粤GSB520", "京AD12345"}
PROVINCE_CODES = {"京", "津", "沪", "渝", "冀", "豫", "云", "辽", "黑", "湘", "皖", "鲁",
                  "新", "苏", "浙", "赣", "鄂", "桂", "甘", "晋", "蒙", "陕", "吉", "闽",
                  "贵", "粤", "青", "藏", "川", "宁", "琼"}
PLATE_PATTERN = re.compile(r'^[' + ''.join(PROVINCE_CODES) + r'][A-Z]{1}[A-Z0-9]{5,6}$')

# ====================== 图像预处理函数 ======================
def preprocess_for_ocr(plate_roi):
    if plate_roi is None or plate_roi.size == 0:
        return None
    plate = imutils.resize(plate_roi, width=280)
    gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileSize=(8, 8))
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

# ====================== 基础图像处理函数 ======================
def preprocess_image(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    return cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)

def filter_blue_plate_region(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
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
            best_plate = frame[y:y+h, x:x+w]
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

# ====================== 串口读取工具函数 ======================
def read_serial_response(ser):
    """非阻塞读取串口返回数据，解码并返回有效字符串"""
    if ser and ser.is_open and ser.in_waiting > 0:
        try:
            # 读取一行数据，匹配Arduino换行输出格式
            data = ser.readline().decode('utf-8', errors='ignore').strip()
            if data:
                return data
        except Exception:
            pass
    return None

# ====================== 主程序 ======================
def main():
    # 初始化串口
    arduino_serial = None
    try:
        arduino_serial = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=SERIAL_TIMEOUT)
        safe_print(f"✅ 串口打开成功：{SERIAL_PORT}")
    except Exception:
        safe_print(f"❌ 串口打开失败")
        arduino_serial = None

    # 初始化摄像头
    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        safe_print("错误：无法打开摄像头")
        return

    safe_print("摄像头已启动，按 `q` 键退出程序")
    detect_counter = 0
    frame_counter = 0
    cached_plate_num = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            safe_print("错误：无法读取画面")
            break

        frame = imutils.resize(frame, width=FRAME_WIDTH)
        output_frame = frame.copy()
        frame_counter += 1

        # ========== 新增：实时读取并打印Arduino回传数据 ==========
        response = read_serial_response(arduino_serial)
        if response:
            safe_print(f"📥 Arduino 回传数据：{response}")

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

            if is_plate_authorized(current_plate):
                if processed_plate is not None:
                    cv2.imshow("Plate ROI", processed_plate)
                    cv2.waitKey(1)
                if current_plate != cached_plate_num:
                    safe_print(f"✅ 合法车牌：{current_plate}")
                    cached_plate_num = current_plate
                    # 串口发送数据
                    if arduino_serial and arduino_serial.is_open:
                        try:
                            send_data = f"{cached_plate_num}\n".encode("utf-8")
                            arduino_serial.write(send_data)
                            safe_print(f"📤 串口发送成功：{cached_plate_num}")
                        except Exception:
                            pass
            else:
                if cv2.getWindowProperty("Plate ROI", cv2.WND_PROP_VISIBLE) >= 0:
                    cv2.destroyWindow("Plate ROI")
                cached_plate_num = ""

        # 界面绘制逻辑
        if is_plate_detected and plate_loc is not None:
            x, y, w, h = plate_loc
            cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            text = f"Plate: {cached_plate_num}" if cached_plate_num else "Detecting Plate..."
            cv2.putText(output_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            cached_plate_num = ""
            cv2.putText(output_frame, "No Plate", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示主画面
        cv2.imshow("License Plate Recognition", output_frame)
        # 退出逻辑
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 资源释放
    cap.release()
    if arduino_serial and arduino_serial.is_open:
        arduino_serial.close()
        safe_print("🔌 串口已关闭")
    cv2.destroyAllWindows()
    safe_print("程序已退出")

if __name__ == "__main__":
    main()