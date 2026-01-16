import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from ament_index_python.packages import get_package_share_directory
import numpy as np
from tensorflow import keras
import json
import os
import math

class DrivingNode(Node):
    def __init__(self):
        super().__init__('driving_node')
        
        # 모델 불러오기
        self.model = None
        self.params = None
        self.model_load()

        # 기존 학습 패키지에서 사용된 변수 모음
        # 각 변수를 통해 주행 값을 계산함
        self.goal_pose_x = 2.0
        self.goal_pose_y = 1.8
        self.robot_pose_x = 0.0
        self.robot_pose_y = 0.0
        self.start_x = 0.0
        self.start_y = 0.0
        self.scan_ranges = []
        self.front_ranges = []
        self.min_obstacle_distance = 10.0
        self.front_min_obstacle_distance = 10.0
        self.goal_distance = 1.0
        self.goal_angle = 0.0

        self.vel = [1.5, 0.75, 0.0, -0.75, -1.5]

        sensor_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        cmd_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=10
        )
        
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, sensor_qos)
        self.odom_sub = self.create_subscription(Odometry, 'odom', self.odom_callback, sensor_qos)
        self.cmd_pub = self.create_publisher(Twist, 'cmd_vel', cmd_qos)

        self.get_logger().info('================================================')
        self.get_logger().info('=========== turtlebot3 driving start ===========')
        self.get_logger().info('================================================')

    def model_load(self):
        '''
            학습된 모델 로드 함수
            다른 패키지에서 학습된 학습 모델을 불러와 터틀봇의 상황에 맞는 학습 결과를 도출.
        '''
        try:
            dqn_pkg_path = get_package_share_directory('turtlebot3_dqn')
            model_h5_path = os.path.join(dqn_pkg_path, 'saved_model', 'route1_GTA27_best.h5')
            model_json_path = os.path.join(dqn_pkg_path, 'saved_model', 'route1_GTA27_best.json')
            
            self.get_logger().info(f'Loading model from: {model_h5_path}')
            
            # 학습시킨 주행 모델 및 파라미터 로드
            self.model = keras.models.load_model(model_h5_path)
            
            with open(model_json_path, 'r') as f:
                self.params = json.load(f)
            
            self.get_logger().info('================================================')
            self.get_logger().info('=========== MODEL LOAD SUCCESSFULLY ============')
            self.get_logger().info('================================================')
            
        except Exception as e:
            self.get_logger().info('================================================')
            self.get_logger().info('=============== MODEL LOAD FAILED ==============')
            self.get_logger().info('================================================')
            self.get_logger().info(f'error : {e}')
            raise
    
    def odom_callback(self, msg):
        # 로봇 위치 업데이트
        self.robot_pose_x = msg.pose.pose.position.x
        self.robot_pose_y = msg.pose.pose.position.y
        _, _, self.robot_pose_theta = self.euler_from_quaternion(msg.pose.pose.orientation)

        # 골까지의 거리 계산
        goal_distance = math.sqrt(
            (self.goal_pose_x - self.robot_pose_x) ** 2
            + (self.goal_pose_y - self.robot_pose_y) ** 2)
        
        # 골까지의 각도 계산
        path_theta = math.atan2(
            self.goal_pose_y - self.robot_pose_y,
            self.goal_pose_x - self.robot_pose_x)

        goal_angle = path_theta - self.robot_pose_theta
        
        # 각도 정규화 (-π ~ π)
        if goal_angle > math.pi:
            goal_angle -= 2 * math.pi
        elif goal_angle < -math.pi:
            goal_angle += 2 * math.pi

        self.goal_distance = goal_distance
        self.goal_angle = goal_angle

    def scan_callback(self, msg):
        ranges = np.array(msg.ranges, dtype=np.float32)

        # 1. 무한대/결측치 처리
        ranges[np.isinf(ranges)] = 3.5
        ranges[np.isnan(ranges)] = 3.5

        num_ranges = len(ranges) 
        lidar_samples = 24
        
        self.front_ranges = []
        
        step = num_ranges // 48 
        
        # 전방 180도만 계산되도록
        for i in range(-(lidar_samples // 2), lidar_samples // 2):
            idx = int(i * (num_ranges / 48))
            if idx < 0:
                idx = num_ranges + idx
                
            self.front_ranges.append(ranges[idx])

        self.min_obstacle_distance = np.min(ranges)
        self.front_min_obstacle_distance = np.min(self.front_ranges)

        # 현재 상태에 따라 모델이 예측
        state = self.prepare_state()
        action = self.model.predict(state, verbose=0)
        action_idx = int(np.argmax(action))
        
        self.get_logger().info(f'Action Index: {action_idx} | Dist: {self.front_min_obstacle_distance:.2f}')
        self.publish_action(action_idx)

    # 주행 함수 - 학습 데이터를 통해 예측한 angular_z 값을 발행
    def publish_action(self, action_idx):
        twist = Twist()
        if self.vel[action_idx] > 4:
            twist.linear.x = self.vel[action_idx]
            twist.angular.z = 0.0
        else:
            twist.linear.x = 0.15
            twist.angular.z = self.vel[action_idx]
        
        self.cmd_pub.publish(twist)
        
    def prepare_state(self):
        # 학습 데이터에 맞게 데이터 전처리
        state = np.concatenate([[self.goal_distance, self.goal_angle], self.front_ranges]).reshape(1, -1)
        
        return state
    
    def euler_from_quaternion(self, quat):
        # 학습 시, 계산한 각도 함수와 동일하게 구현
        x = quat.x
        y = quat.y
        z = quat.z
        w = quat.w

        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        sinp = 2 * (w * y - z * x)
        pitch = math.asin(sinp)

        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

def main(args=None):
    rclpy.init(args=args)
    node = DrivingNode()
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        twist = Twist()
        node.cmd_pub.publish(twist)
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()