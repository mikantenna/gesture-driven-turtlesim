import math
import os
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node


class HandTurtleController(Node):

  def __init__(self):
    super().__init__('hand_turtle_controller')
    self.publisher_ = self.create_publisher(Twist, '/turtle1/cmd_vel', 10)

    # HandLandmarkerの初期化 (Tasks API)
    model_path = os.path.join(os.path.dirname(__file__), 'hand_landmarker.task')
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.HandLandmarkerOptions(
        base_options=base_options,
        num_hands=1,
        min_hand_detection_confidence=0.7,
        min_hand_presence_confidence=0.7,
        min_tracking_confidence=0.7,
    )
    self.detector = vision.HandLandmarker.create_from_options(options)

    # カメラキャプチャ（0番デバイス）
    self.cap = cv2.VideoCapture(0)
    if not self.cap.isOpened():
      self.get_logger().error('Webカメラを開けませんでした。')
      raise RuntimeError('Camera open failed')

    # 30fps周期で処理を実行
    self.timer = self.create_timer(0.033, self.process_frame)

  def process_frame(self):
    ret, frame = self.cap.read()
    if not ret:
      return

    # 鏡像反転とRGB変換
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 推論実行
    detection_result = self.detector.detect(mp_image)

    twist = Twist()

    if detection_result.hand_landmarks:
      landmarks = detection_result.hand_landmarks[0]
      h, w, _ = frame.shape

      # ランドマーク: 5 = 人差し指付け根 (MCP), 8 = 人差し指先端 (TIP)
      x_base = landmarks[5].x * w
      y_base = landmarks[5].y * h
      x_tip = landmarks[8].x * w
      y_tip = landmarks[8].y * h

      # 指のベクトルと角度計算
      dx = x_tip - x_base
      dy = -(y_tip - y_base)  # 画面上方向を正とする
      length = math.hypot(dx, dy)

      # 指が一定以上伸びている場合にコマンド発行 (閾値: 35px)
      if length > 35:
        angle = math.atan2(dy, dx)
        twist.linear.x = 2.0 * math.cos(angle)
        twist.angular.z = 2.5 * math.sin(angle)

        # 画面上に指差し矢印を描画
        cv2.arrowedLine(
            frame,
            (int(x_base), int(y_base)),
            (int(x_tip), int(y_tip)),
            (0, 255, 0),
            3,
        )

      # ランドマーク描画（指先・付け根）
      cv2.circle(frame, (int(x_base), int(y_base)), 5, (0, 0, 255), -1)
      cv2.circle(frame, (int(x_tip), int(y_tip)), 5, (255, 0, 0), -1)

    # 指が非検出の場合は停止
    self.publisher_.publish(twist)

    # 映像表示
    cv2.imshow('Hand Pointer Controller', frame)
    if cv2.waitKey(1) & 0xFF == 27:  # ESCキーで終了
      rclpy.shutdown()

  def destroy_node(self):
    self.cap.release()
    cv2.destroyAllWindows()
    super().destroy_node()


def main(args=None):
  rclpy.init(args=args)
  controller = HandTurtleController()
  try:
    rclpy.spin(controller)
  except KeyboardInterrupt:
    pass
  finally:
    controller.destroy_node()
    if rclpy.ok():
      rclpy.shutdown()


if __name__ == '__main__':
  main()