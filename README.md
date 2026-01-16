## 🚀 Project Overview
본 프로젝트는 터틀봇3를 활용하여 장애물 회피와 객체 인식 기반의 경로 주행을 수행하는 지능형 자율주행 시스템입니다. Roboflow를 통해 직접 구축한 **Polygon 데이터셋**으로 YOLOv8 모델을 학습시켰으며, 이를 강화학습 상태(State)와 보상(Reward) 체계에 통합하였습니다.

## 🧠 Core Modules
- **[turtlebot3_dqn_agent]**: TensorFlow 기반 DQN 에이전트 및 경험 재생(Experience Replay) 엔진
- **[turtlebot3_yolo_perception]**: YOLOv8을 활용한 실시간 화살표 탐지 및 방향 추적 모듈
- **[turtlebot3_db_logger]**: MySQL 연동을 통한 주행 로그 및 모델 가중치(H5) 관리 도구

## 📊 Dataset & Training
- 🎯 **[Roboflow Dataset (arrow_detected)](https://app.roboflow.com/123748712/arrow_detected/1)**: 직접 라벨링한 2개 클래스(`blue_left`, `blue_right`) 폴리곤 데이터셋
- 🤖 **YOLOv8 Training**: 고도의 인식을 위해 Roboflow 데이터셋을 활용한 커스텀 학습 수행
- 📈 **DQN Learning**: LiDAR 24채널 정보와 Vision 정보를 결합한 31차원 상태 공간 학습

## 🛠️ Implementation Highlights
- **Vision-Reward Integration**: 로봇의 회전 방향과 화살표의 화면 내 이동량($\Delta$)을 비교하여 주행 정확도 검증
- **Sim-to-Real Deployment**: Gazebo 시뮬레이션에서 검증된 최적의 **H5 가중치 모델** 탑재
- **Data Persistence**: MySQL을 통한 에피소드별 보상 점수 기록 및 성과 분석