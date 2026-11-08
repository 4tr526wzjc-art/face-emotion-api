import time
import uuid
from collections import deque
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, jsonify, request, send_from_directory

from face_analyzer import (
    FaceAnalyzer,
    calculate_activity,
    calculate_anxiety_tension,
    calculate_attention
)

from report_generator import generate_reports


# ==========================================================
# 基础配置
# ==========================================================

app = Flask(__name__)

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

BASELINE_SECONDS = 10
MEASURE_SECONDS = 240
CHECKPOINTS = [60, 180, 240]

SAVE_INTERVAL = 0.2
SMOOTH_SECONDS = 8

CALM_THRESHOLD = 0.06
ACTIVE_THRESHOLD = 0.15

ANXIETY_TENSION_THRESHOLD = 0.35
ANXIETY_DURATION_SECONDS = 15

# 临时保存每位用户的一次测量数据
sessions = {}

# 初始化 MediaPipe 人脸分析器
analyzer = FaceAnalyzer()


# ==========================================================
# 跨域设置
# 方便其他网站前端调用本接口
# ==========================================================

@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Headers"] = (
        "Content-Type"
    )
    response.headers["Access-Control-Allow-Methods"] = (
        "GET, POST, OPTIONS"
    )

    return response


# ==========================================================
# 工具函数
# ==========================================================

def clamp(value, low=0.0, high=1.0):
    return max(low, min(high, value))


def average_of(records, key):
    values = [
        item[key]
        for item in records
        if key in item
    ]

    if not values:
        return 0.0

    return float(np.mean(values))


def build_baseline(samples):
    """根据 10 秒自然状态样本建立个人基线。"""

    baseline = {}

    if not samples:
        return baseline

    for key in samples[0].keys():
        if key == "landmarks":
            continue

        values = [
            item[key]
            for item in samples
            if key in item
        ]

        baseline[key] = float(np.mean(values))

    return baseline


def get_state(activity, attention, tension, anxiety_persistent):
    """
    得到面部行为趋势状态。
    注意：不是心理或医学诊断。
    """

    if anxiety_persistent:
        return "焦虑相关紧张倾向"

    if activity < CALM_THRESHOLD and tension < 0.20:
        return "平静倾向"

    if activity >= ACTIVE_THRESHOLD:
        if tension >= ANXIETY_TENSION_THRESHOLD:
            return "高唤醒/紧张倾向"

        return "唤醒/兴奋倾向"

    if attention < 0.50:
        return "注意力波动倾向"

    return "中性或过渡状态"


def build_checkpoint_summary(records, minute):
    """生成 1、3、4 分钟节点分析。"""

    if not records:
        return {
            "minute": minute,
            "activity": 0.0,
            "attention": 0.0,
            "tension": 0.0,
            "smile": 0.0,
            "state": "数据不足"
        }

    # 最近约 30 秒的数据
    recent_records = records[-min(len(records), 150):]

    activity = average_of(recent_records, "activity")
    attention = average_of(recent_records, "attention")
    tension = average_of(recent_records, "tension")
    smile = average_of(recent_records, "smile")

    anxiety_count = sum(
        1 for item in recent_records
        if item["anxiety_persistent"] == "是"
    )

    anxiety_ratio = anxiety_count / len(recent_records)

    state = get_state(
        activity,
        attention,
        tension,
        anxiety_ratio >= 0.30
    )

    return {
        "minute": minute,
        "activity": round(activity, 4),
        "attention": round(attention, 4),
        "tension": round(tension, 4),
        "smile": round(smile, 4),
        "state": state
    }


def make_suggestions(checkpoints, records):
    """根据整场趋势生成疗愈建议。"""

    overall_activity = average_of(records, "activity")
    overall_attention = average_of(records, "attention")
    overall_tension = average_of(records, "tension")

    anxiety_count = sum(
        1 for item in records
        if item["anxiety_persistent"] == "是"
    )

    anxiety_ratio = anxiety_count / len(records)

    first_state = checkpoints.get(60, {}).get(
        "state",
        "数据不足"
    )

    last_state = checkpoints.get(240, {}).get(
        "state",
        "数据不足"
    )

    suggestions = []

    if anxiety_ratio >= 0.20:
        suggestions.append(
            "检测到持续性紧张相关面部行为。建议优先选择呼吸训练、"
            "自然风景或舒缓音乐类数字疗愈内容，并减少周围环境干扰。"
        )

    if overall_tension >= ANXIETY_TENSION_THRESHOLD:
        suggestions.append(
            "眉部、眼周或唇部紧张相关活动较多。"
            "建议在视频结束后增加 1 至 2 分钟慢呼吸或闭眼放松。"
        )

    if overall_activity >= ACTIVE_THRESHOLD:
        suggestions.append(
            "观看过程中的面部活跃度整体较高。"
            "建议在疗愈视频结束后保留一段安静恢复时间。"
        )

    if overall_attention < 0.55:
        suggestions.append(
            "观看投入度存在一定波动。建议关闭消息提醒，"
            "调整光线、坐姿和屏幕高度。"
        )

    if (
        first_state in [
            "高唤醒/紧张倾向",
            "焦虑相关紧张倾向"
        ]
        and last_state == "平静倾向"
    ):
        suggestions.append(
            "用户从较高紧张或唤醒趋势逐步过渡至相对平静，"
            "当前疗愈视频可能对放松过程具有一定帮助。"
        )

    if last_state in [
        "高唤醒/紧张倾向",
        "焦虑相关紧张倾向"
    ]:
        suggestions.append(
            "结束时仍呈现较高紧张趋势。建议暂时离开屏幕、补充饮水、"
            "进行缓慢深呼吸。若不适感持续并影响生活，可考虑寻求专业支持。"
        )

    if not suggestions:
        suggestions.append(
            "本次面部行为变化整体较稳定。建议在相近环境下多次记录，"
            "比较不同疗愈视频前后的长期变化趋势。"
        )

    return suggestions


# ==========================================================
# 接口 1：开始一次新的疗愈测量
# ==========================================================

@app.route("/api/start-session", methods=["POST", "OPTIONS"])
def start_session():
    if request.method == "OPTIONS":
        return jsonify({"message": "ok"})

    session_id = str(uuid.uuid4())

    sessions[session_id] = {
        "session_id": session_id,
        "stage": "baseline",
        "baseline_start_time": time.time(),
        "measure_start_time": None,
        "baseline_samples": [],
        "baseline": None,
        "records": [],
        "smooth_values": deque(),
        "anxiety_start_time": None,
        "last_save_time": 0.0,
        "checkpoints": {},
        "finished_checkpoints": set()
    }

    return jsonify({
        "success": True,
        "session_id": session_id,
        "stage": "baseline",
        "message": "测量已开始，请保持自然状态，正在进行 10 秒基线校准。",
        "baseline_seconds": BASELINE_SECONDS,
        "measure_seconds": MEASURE_SECONDS
    })


# ==========================================================
# 接口 2：网页上传一张摄像头画面进行分析
# ==========================================================

@app.route("/api/analyze", methods=["POST", "OPTIONS"])
def analyze():
    if request.method == "OPTIONS":
        return jsonify({"message": "ok"})

    session_id = request.form.get("session_id")

    if not session_id:
        return jsonify({
            "success": False,
            "message": "缺少 session_id。"
        }), 400

    if session_id not in sessions:
        return jsonify({
            "success": False,
            "message": "测量会话不存在或已结束。"
        }), 404

    if "image" not in request.files:
        return jsonify({
            "success": False,
            "message": "未接收到摄像头图像。"
        }), 400

    image_file = request.files["image"]
    image_bytes = image_file.read()

    image_array = np.frombuffer(
        image_bytes,
        np.uint8
    )

    frame = cv2.imdecode(
        image_array,
        cv2.IMREAD_COLOR
    )

    if frame is None:
        return jsonify({
            "success": False,
            "message": "图像格式错误，无法读取。"
        }), 400

    session = sessions[session_id]

    timestamp_ms = int(time.monotonic() * 1000)

    features = analyzer.analyze_frame(
        frame,
        timestamp_ms
    )

    if features is None:
        return jsonify({
            "success": True,
            "face_detected": False,
            "stage": session["stage"],
            "message": "未检测到人脸，请正对摄像头并改善光线。"
        })

    current_time = time.time()

    # ======================================================
    # 阶段一：10 秒个人基线校准
    # ======================================================
    if session["stage"] == "baseline":
        session["baseline_samples"].append(features)

        elapsed = current_time - session["baseline_start_time"]

        if elapsed >= BASELINE_SECONDS:
            session["baseline"] = build_baseline(
                session["baseline_samples"]
            )

            session["stage"] = "measurement"
            session["measure_start_time"] = time.time()

            return jsonify({
                "success": True,
                "face_detected": True,
                "stage": "measurement",
                "baseline_completed": True,
                "message": "个人基线校准完成，开始正式测量。",
                "landmarks": features["landmarks"]
            })

        return jsonify({
            "success": True,
            "face_detected": True,
            "stage": "baseline",
            "baseline_completed": False,
            "remaining_seconds": round(
                max(0, BASELINE_SECONDS - elapsed),
                1
            ),
            "message": "正在进行个人自然状态校准。",
            "landmarks": features["landmarks"]
        })

    # ======================================================
    # 阶段二：正式测量
    # ======================================================
    measure_elapsed = (
        current_time - session["measure_start_time"]
    )

    raw_activity = calculate_activity(
        features,
        session["baseline"]
    )

    raw_attention = calculate_attention(features)

    raw_tension = calculate_anxiety_tension(features)

    session["smooth_values"].append((
        measure_elapsed,
        raw_activity,
        raw_attention,
        raw_tension
    ))

    while (
        session["smooth_values"]
        and measure_elapsed - session["smooth_values"][0][0]
        > SMOOTH_SECONDS
    ):
        session["smooth_values"].popleft()

    activity = float(np.mean([
        item[1]
        for item in session["smooth_values"]
    ]))

    attention = float(np.mean([
        item[2]
        for item in session["smooth_values"]
    ]))

    tension = float(np.mean([
        item[3]
        for item in session["smooth_values"]
    ]))

    if tension >= ANXIETY_TENSION_THRESHOLD:
        if session["anxiety_start_time"] is None:
            session["anxiety_start_time"] = measure_elapsed

        anxiety_duration = (
            measure_elapsed -
            session["anxiety_start_time"]
        )

        anxiety_persistent = (
            anxiety_duration >=
            ANXIETY_DURATION_SECONDS
        )
    else:
        session["anxiety_start_time"] = None
        anxiety_duration = 0.0
        anxiety_persistent = False

    state = get_state(
        activity,
        attention,
        tension,
        anxiety_persistent
    )

    # 保存原始数据
    if (
        current_time - session["last_save_time"]
        >= SAVE_INTERVAL
    ):
        session["records"].append({
            "time_seconds": round(measure_elapsed, 2),
            "eye_open": round(features["eye_open"], 4),
            "brow_eye_ratio": round(
                features["brow_eye_ratio"],
                4
            ),
            "mouth_open": round(
                features["mouth_open"],
                4
            ),
            "mouth_corner_score": round(
                features["mouth_corner_score"],
                4
            ),
            "head_yaw": round(
                features["head_yaw"],
                4
            ),
            "head_roll": round(
                features["head_roll"],
                4
            ),
            "smile": round(
                features["smile"],
                4
            ),
            "frown": round(
                features["frown"],
                4
            ),
            "brow_tension": round(
                features["brow_tension"],
                4
            ),
            "eye_squint": round(
                features["eye_squint"],
                4
            ),
            "lip_press": round(
                features["lip_press"],
                4
            ),
            "activity": round(activity, 4),
            "attention": round(attention, 4),
            "tension": round(tension, 4),
            "anxiety_duration": round(
                anxiety_duration,
                2
            ),
            "anxiety_persistent": (
                "是"
                if anxiety_persistent
                else "否"
            ),
            "state": state
        })

        session["last_save_time"] = current_time

    # 自动记录 1、3、4 分钟节点
    for checkpoint in CHECKPOINTS:
        if (
            measure_elapsed >= checkpoint
            and checkpoint not in session["finished_checkpoints"]
        ):
            session["checkpoints"][checkpoint] = (
                build_checkpoint_summary(
                    session["records"],
                    checkpoint // 60
                )
            )

            session["finished_checkpoints"].add(
                checkpoint
            )

    return jsonify({
        "success": True,
        "face_detected": True,
        "stage": "measurement",
        "time_seconds": round(measure_elapsed, 1),
        "measurement_finished": (
            measure_elapsed >= MEASURE_SECONDS
        ),
        "activity": round(activity, 4),
        "attention": round(attention, 4),
        "tension": round(tension, 4),
        "state": state,
        "anxiety_persistent": anxiety_persistent,
        "landmarks": features["landmarks"]
    })


# ==========================================================
# 接口 3：结束测量并生成报告
# ==========================================================

@app.route("/api/end-session", methods=["POST", "OPTIONS"])
def end_session():
    if request.method == "OPTIONS":
        return jsonify({"message": "ok"})

    data = request.get_json(silent=True) or {}
    session_id = data.get("session_id")

    if session_id not in sessions:
        return jsonify({
            "success": False,
            "message": "测量会话不存在或已结束。"
        }), 404

    session = sessions[session_id]
    records = session["records"]

    if not records:
        return jsonify({
            "success": False,
            "message": "没有足够的正式测量数据，无法生成报告。"
        }), 400

    # 如果用户提前结束，也补齐节点结果
    for checkpoint in CHECKPOINTS:
        if checkpoint not in session["checkpoints"]:
            session["checkpoints"][checkpoint] = (
                build_checkpoint_summary(
                    records,
                    checkpoint // 60
                )
            )

    suggestions = make_suggestions(
        session["checkpoints"],
        records
    )

    files = generate_reports(
        session_id,
        records,
        session["checkpoints"],
        suggestions
    )

    # 测量完成后删除内存中的本次会话数据
    del sessions[session_id]

    return jsonify({
        "success": True,
        "message": "测量结束，报告已生成。",
        "checkpoints": session["checkpoints"],
        "suggestions": suggestions,
        "files": files,
        "download_urls": {
            "csv": f"/results/{files['csv_file']}",
            "chart": f"/results/{files['chart_file']}",
            "report": f"/results/{files['report_file']}",
            "excel": f"/results/{files['excel_file']}"
        }
    })


# ==========================================================
# 接口 4：下载生成的结果文件
# ==========================================================

@app.route("/results/<filename>", methods=["GET"])
def download_result(filename):
    return send_from_directory(
        RESULTS_DIR,
        filename,
        as_attachment=True
    )


# ==========================================================
# 接口 5：健康检查
# ==========================================================

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "success": True,
        "message": "人脸情绪反馈接口服务运行正常。"
    })


# ==========================================================
# 启动服务
# ==========================================================

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )