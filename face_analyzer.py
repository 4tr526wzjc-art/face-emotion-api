import math

import cv2
import mediapipe as mp


MODEL_PATH = "face_landmarker.task"


def distance(point_a, point_b):
    """计算两个面部关键点之间的二维距离。"""
    return math.sqrt(
        (point_a.x - point_b.x) ** 2 +
        (point_a.y - point_b.y) ** 2
    )


def safe_ratio(value, denominator, default=0.0):
    """安全除法，避免除以 0。"""
    if abs(denominator) < 0.000001:
        return default

    return value / denominator


def clamp(value, low=0.0, high=1.0):
    """限制数值范围。"""
    return max(low, min(high, value))


def blendshape_score(categories, names):
    """获取 MediaPipe 指定表情类别中的最高分。"""
    if not categories:
        return 0.0

    score_map = {
        item.category_name: item.score
        for item in categories
    }

    scores = [
        score_map.get(name, 0.0)
        for name in names
    ]

    return max(scores, default=0.0)


def calculate_activity(features, baseline):
    """
    根据当前特征与用户个人基线的差异，
    计算面部活跃度。
    """

    if not baseline:
        return 0.0

    eye_change = abs(
        safe_ratio(
            features["eye_open"],
            baseline["eye_open"],
            1.0
        ) - 1
    )

    brow_change = abs(
        safe_ratio(
            features["brow_eye_ratio"],
            baseline["brow_eye_ratio"],
            1.0
        ) - 1
    )

    mouth_change = abs(
        safe_ratio(
            features["mouth_open"],
            baseline["mouth_open"],
            1.0
        ) - 1
    )

    expression_activity = (
        features["smile"] +
        features["frown"] +
        features["brow_tension"] +
        features["eye_squint"] +
        features["lip_press"] +
        features["cheek_activity"] +
        features["jaw_open"]
    ) / 7

    head_activity = (
        features["head_yaw"] +
        features["head_roll"]
    )

    activity = (
        eye_change * 0.20 +
        brow_change * 0.20 +
        mouth_change * 0.22 +
        expression_activity * 0.28 +
        head_activity * 0.10
    )

    return clamp(activity)


def calculate_attention(features):
    """根据头部偏转和倾斜估计观看投入度。"""

    deviation = (
        features["head_yaw"] * 2.3 +
        features["head_roll"] * 1.5 +
        features["eye_symmetry"] * 1.3
    )

    return clamp(1.0 - deviation * 3.2)


def calculate_anxiety_tension(features):
    """
    计算焦虑相关面部紧张指数。

    该指标只表示眉部、眼周、抿嘴和嘴角下压等
    紧张相关面部行为，不构成焦虑诊断。
    """

    tension = (
        features["brow_tension"] * 0.35 +
        features["eye_squint"] * 0.25 +
        features["lip_press"] * 0.25 +
        features["frown"] * 0.15
    )

    return clamp(tension)


class FaceAnalyzer:
    """
    面部分析器：
    接收一张 BGR 格式画面，
    输出面部关键点和面部行为特征。
    """

    def __init__(self):
        base_options = mp.tasks.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = mp.tasks.vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_faces=1,
            output_face_blendshapes=True
        )

        self.landmarker = (
            mp.tasks.vision.FaceLandmarker
            .create_from_options(options)
        )

    def analyze_frame(self, frame_bgr, timestamp_ms):
        """
        分析一张网页摄像头画面。

        参数：
        frame_bgr：OpenCV 读取的 BGR 画面
        timestamp_ms：毫秒时间戳，必须持续递增

        返回：
        未检测到人脸时返回 None；
        检测成功时返回特征字典。
        """

        rgb_frame = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = self.landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        if not result.face_landmarks:
            return None

        landmarks = result.face_landmarks[0]

        if result.face_blendshapes:
            blendshapes = result.face_blendshapes[0]
        else:
            blendshapes = []

        face_width = max(
            distance(landmarks[234], landmarks[454]),
            0.0001
        )

        # 眼睛开合程度
        left_eye_vertical = distance(
            landmarks[159],
            landmarks[145]
        )

        left_eye_horizontal = distance(
            landmarks[33],
            landmarks[133]
        )

        right_eye_vertical = distance(
            landmarks[386],
            landmarks[374]
        )

        right_eye_horizontal = distance(
            landmarks[362],
            landmarks[263]
        )

        left_eye_ratio = safe_ratio(
            left_eye_vertical,
            left_eye_horizontal
        )

        right_eye_ratio = safe_ratio(
            right_eye_vertical,
            right_eye_horizontal
        )

        eye_open = (
            left_eye_ratio + right_eye_ratio
        ) / 2

        # 眉眼距离
        left_brow_eye = distance(
            landmarks[105],
            landmarks[159]
        )

        right_brow_eye = distance(
            landmarks[334],
            landmarks[386]
        )

        brow_eye_ratio = (
            left_brow_eye + right_brow_eye
        ) / (2 * face_width)

        # 嘴巴开合程度
        mouth_vertical = distance(
            landmarks[13],
            landmarks[14]
        )

        mouth_horizontal = distance(
            landmarks[61],
            landmarks[291]
        )

        mouth_open = safe_ratio(
            mouth_vertical,
            mouth_horizontal
        )

        # 嘴角相对位置
        mouth_center_y = (
            landmarks[13].y +
            landmarks[14].y
        ) / 2

        mouth_corner_y = (
            landmarks[61].y +
            landmarks[291].y
        ) / 2

        mouth_corner_score = (
            mouth_center_y - mouth_corner_y
        ) / face_width

        # 左右眼部差异
        eye_symmetry = abs(
            left_eye_ratio - right_eye_ratio
        )

        # 粗略头部方向
        eye_center_x = (
            landmarks[33].x +
            landmarks[263].x
        ) / 2

        head_yaw = abs(
            landmarks[1].x - eye_center_x
        ) / face_width

        head_roll = abs(
            landmarks[33].y - landmarks[263].y
        ) / face_width

        # 网页绘制关键点需要的二维坐标
        landmark_points = []

        for landmark in landmarks:
            landmark_points.append([
                round(landmark.x, 5),
                round(landmark.y, 5)
            ])

        return {
            "eye_open": eye_open,
            "brow_eye_ratio": brow_eye_ratio,
            "mouth_open": mouth_open,
            "mouth_corner_score": mouth_corner_score,
            "eye_symmetry": eye_symmetry,
            "head_yaw": head_yaw,
            "head_roll": head_roll,
            "smile": blendshape_score(
                blendshapes,
                ["mouthSmileLeft", "mouthSmileRight"]
            ),
            "frown": blendshape_score(
                blendshapes,
                ["mouthFrownLeft", "mouthFrownRight"]
            ),
            "brow_tension": blendshape_score(
                blendshapes,
                [
                    "browDownLeft",
                    "browDownRight",
                    "browInnerUp"
                ]
            ),
            "eye_squint": blendshape_score(
                blendshapes,
                ["eyeSquintLeft", "eyeSquintRight"]
            ),
            "lip_press": blendshape_score(
                blendshapes,
                [
                    "mouthPressLeft",
                    "mouthPressRight",
                    "mouthPucker"
                ]
            ),
            "cheek_activity": blendshape_score(
                blendshapes,
                [
                    "cheekSquintLeft",
                    "cheekSquintRight",
                    "cheekPuff"
                ]
            ),
            "jaw_open": blendshape_score(
                blendshapes,
                ["jawOpen"]
            ),
            "landmarks": landmark_points
        }

    def close(self):
        """关闭 MediaPipe 资源。"""
        self.landmarker.close()