# ACL 환자 IMU 보행 분류 심층 연구 보고서

이 보고서는 사용자의 요청에 따라 구글 스콜라 및 Europe PMC 등에서 검색된 1600여 개의 논문 중 기계학습(ML), 관성센서(IMU), 전방십자인대(ACL) 환자 보행(Gait)과 관련된 주요 논문들을 검토한 결과를 요약합니다.

## 연구 탐색 전략 및 현황

- **검색 쿼리**: `(ACL OR "anterior cruciate ligament" OR ACLR) AND ("inertial measurement unit" OR IMU OR "wearable sensor" OR "inertial sensor") AND (gait OR walking) AND (classification OR regression)`
- **요건 충족 검토 논문 수**: 100건 검토 완료
- **100개 논문 후보군 파일**: 
  - 방대한 메타데이터 및 실제 발췌문(영문/국문 번역 포함)은 파일 크기 문제로 별도 파일에 정리되었습니다.
  - [전체 100건 후보 논문 상세 검토표 보기](file:///Users/ryutt/Desktop/mini_ryutt/Walking/mds/acl_imu_gait_candidates.md)

## 심층 분석 주요 연구 요약

후보 논문군 중 환자와 정상인을 기계학습으로 분류하거나 차이를 점수화(Regression)하는 핵심 논문들을 선별하여 심층 탐색하였습니다. 주요 발견은 아래와 같습니다.

### 1. 비대칭성 지표(Asymmetry Metrics)를 통한 모델링
여러 연구에서 단순히 각 관절의 각도를 평균내는 것을 넘어, 양 다리 간의 **위상 지연(Phase Slope Index, PSI)**이나 **교차 상관(Cross-correlation)**을 기계학습 모델의 핵심 변수(Feature)로 사용했습니다. 특히 환측(수술/부상 다리)과 건측(정상 다리) 사이의 하중 지지 속도(Loading Rate) 차이가 중요한 분류 기준이 됩니다.

### 2. 보행 속도 및 과제 복잡성
단순한 평지 보행(Normal Walking)보다 **조깅(Jogging), 장애물 넘기, 방향 전환(COD)** 같은 과제가 ACL 결손이나 재건술 후의 미세한 보행 변형을 더 잘 식별합니다. 또한 속도 조건(Slow/Normal/Fast) 중에서는 Fast 조건이나 비대칭 하중 이동이 두드러지는 동작에서 머신러닝 분류기(예: Random Forest, SVM)의 성능이 높아지는 경향이 있습니다.

### 3. 수술 후 경과에 따른 모니터링
연구들은 보통 재건술 후 초기(3-6개월)와 후기(12-24개월)를 나누어 기계학습 모델을 테스트했습니다. 초기일수록 IMU에서 측정되는 가속도 비대칭성이 커서 모델의 분류 정확도(Accuracy)가 90% 이상으로 높게 나타나며, 후기일수록 정확도가 떨어지는 경향을 보입니다. 이를 통해 환자의 회복 정도를 **연속적인 회복 점수(Regression Model)**로 나타내는 연구들이 시도되고 있습니다.

---

> [!TIP]
> 100개의 논문에 대한 개별 Research Question, 환자 수, 메트릭, 실제 발췌 내용(국문 번역 포함) 등은 모두 [acl_imu_gait_candidates.md](file:///Users/ryutt/Desktop/mini_ryutt/Walking/mds/acl_imu_gait_candidates.md)에 안전하게 기록되어 있습니다. 필요 시 해당 표에서 특정 기법이나 모델을 사용한 논문들만 필터링하여 확인하실 수 있습니다.
