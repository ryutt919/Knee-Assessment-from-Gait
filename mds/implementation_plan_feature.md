# 📌 보행 모델 임베딩 최적화 피처 선택(Feature Selection) 계획 (Ver. 2.0)

## 1. 개요 (Goal Description)

본 계획은 Xsens MVNX 파서를 통해 추출된 고차원 시계열 피처셋(693 Dimensions)을 최적화하여, 딥러닝 모델의 학습 효율을 높이고 데이터 노이즈를 제어하기 위함입니다. 보행 패턴 분석의 핵심인 하체 동역학(Lower Body Dynamics)에 집중함으로써 **차원의 저주(Curse of Dimensionality)** 문제를 해결하고 모델의 군집화(Clustering) 성능을 극대화합니다.

---

## 2. ⭐ 수정됨: 신체 세그먼트 기반 필터링 상세 (Lower Body Exclusion)

사용자의 결정에 따라 하체 동역학에 불필요한 상체 세그먼트(Index 1~14)를 전면 제거합니다.

| 분류           | 세그먼트 인덱스 (Index) | 세그먼트 명칭 (Label)                     | 비고              |
| :------------- | :---------------------- | :---------------------------------------- | :---------------- |
| **보존** | **0**             | Pelvis                                    | 하체 중심점       |
| **삭제** | **1 ~ 4**         | L5, L3, T12, T8                           | 척추 및 하부 흉추 |
| **삭제** | **5 ~ 6**         | Neck, Head                                | 목 및 머리        |
| **삭제** | **7 ~ 10**        | R-Shoulder, R-UpperArm, R-Forearm, R-Hand | 우측 상지         |
| **삭제** | **11 ~ 14**       | L-Shoulder, L-UpperArm, L-Forearm, L-Hand | 좌측 상지         |
| **보존** | **15 ~ 18**       | R-UpperLeg, R-LowerLeg, R-Foot, R-Toe     | 우측 하지         |
| **보존** | **19 ~ 22**       | L-UpperLeg, L-LowerLeg, L-Foot, L-Toe     | 좌측 하지         |

---

## 3. ⭐ 수정됨: 속성별 컬럼 삭제 리스트 (Technical Specifications)

### 🧺 그룹 1: 미분 속성 및 파생 지표 (276 Columns 삭제)

시계열 모델(LSTM, CNN)의 자체 특징 추출 기능을 활용하기 위해 중복 도함수 데이터를 제거합니다.

- **Velocity (69):** `velocity_0` ~ `velocity_68` (23 Segments × 3 Axis)
- **Acceleration (69):** `acceleration_0` ~ `acceleration_68`
- **Angular Velocity (69):** `angularVelocity_0` ~ `angularVelocity_68`
- **Angular Acceleration (69):** `angularAcceleration_0` ~ `angularAcceleration_68`

### 🧺 그룹 2: 변환 행렬 및 에고 노드 (102 Columns 삭제)

기본 `jointAngle`과 중핵 정보가 겹치는 좌표계 변환 지표를 제거합니다.

- **JointAngle XZY (66):** `jointAngleXZY_0` ~ `jointAngleXZY_65` (22 Joints × 3 Axis)
- **Ergo Values (36):** `jointAngleErgo_0 ~ 17`, `jointAngleErgoXZY_0 ~ 17`

### 🧺 ⭐ 수정됨: 그룹 3: 상체 세그먼트 대응 데이터 (98 Columns 삭제)

섹션 2의 삭제 세그먼트(14개) 가중치를 물리적으로 제거합니다.

- **Orientation (56):** `orientation_4` ~ `orientation_59` (14 Segments [1-14] × 4 Quaternions)
- **Position (42):** `position_3` ~ `position_44` (14 Segments [1-14] × 3 Axis Vector)

### 🧺 ⭐ 수정됨: 그룹 4: 상체 조인트 및 센서 데이터 (Label-based 삭제)

- **JointAngle:** 상체 관절(Shoulder, Elbow, Wrist, Neck 등)에 해당하는 인덱스 데이터 삭제.
- **Sensor Data:** 상체 부착 센서(Head, Sternum, Shoulder 등 7개 중 해당 센서) 데이터 삭제.

---

## 4. ⭐ 수정됨: 데이터셋 아키텍처 변화 (As-is vs To-be)

| 구분                  | As-is (Raw/Master)              | To-be (Lower Body Only)               |
| :-------------------- | :------------------------------ | :------------------------------------ |
| **총 컬럼 수**  | 693 Columns                     | **315 Columns** (약 54% 축소)   |
| **핵심 피처**   | 전신 포지션 및 속도 포함        | Pelvis + 하지 8개 세그먼트 집중       |
| **데이터셋 명** | `Master_Gait_Dataset.parquet` | `Master_Gait_Dataset_lower.parquet` |
| **특이 사항**   | 상체 노이즈 포함                | 임베딩 공간 내 하지 변위 밀도 상승    |

---

## 5. 실행 및 검증 계획 (Action Plan)

1. **Script:** `agent_temp/make_lower_body_parquet.py`를 정교화하여 위 명세에 따른 하드 필터링 수행.
2. **Verification:**

   - [ ] `df.columns`에서 'Neck', 'Shoulder', 'Hand' 관련 연산 인덱스가 잔존하는지 전수 조사.
   - [ ] `lower` 데이터셋 기반 CNN 임베딩 재학습 후 **Silhouette Score** 개선 폭(Delta) 측정.
   - [ ] 동일 피험자 내 Trial간 유클리드 거리 응집도 평가

   ---

# data_description.csv 컬럼 구조 정리

> **총 컬럼 수**: 693개 | 소스: Xsens MVNX 파서 출력 (Xsens Awinda 풀바디 IMU)

---

## 1. orientation — 세그먼트 방향 쿼터니언 (92컬럼)

23개 신체 세그먼트 × 4 quaternion 성분 (q0, q1, q2, q3)

| 세그먼트        | 컬럼 인덱스                             | 상/하체        |
| --------------- | --------------------------------------- | -------------- |
| Pelvis          | `orientation_0` ~ `orientation_3`   | 중심           |
| L5              | `orientation_4` ~ `orientation_7`   | 상체           |
| L3              | `orientation_8` ~ `orientation_11`  | 상체           |
| T12             | `orientation_12` ~ `orientation_15` | 상체           |
| T8              | `orientation_16` ~ `orientation_19` | 상체           |
| Neck            | `orientation_20` ~ `orientation_23` | 상체           |
| Head            | `orientation_24` ~ `orientation_27` | 상체           |
| Right Shoulder  | `orientation_28` ~ `orientation_31` | 상체           |
| Right Upper Arm | `orientation_32` ~ `orientation_35` | 상체           |
| Right Forearm   | `orientation_36` ~ `orientation_39` | 상체           |
| Right Hand      | `orientation_40` ~ `orientation_43` | 상체           |
| Left Shoulder   | `orientation_44` ~ `orientation_47` | 상체           |
| Left Upper Arm  | `orientation_48` ~ `orientation_51` | 상체           |
| Left Forearm    | `orientation_52` ~ `orientation_55` | 상체           |
| Left Hand       | `orientation_56` ~ `orientation_59` | 상체           |
| Right Upper Leg | `orientation_60` ~ `orientation_63` | **하체** |
| Right Lower Leg | `orientation_64` ~ `orientation_67` | **하체** |
| Right Foot      | `orientation_68` ~ `orientation_71` | **하체** |
| Right Toe       | `orientation_72` ~ `orientation_75` | **하체** |
| Left Upper Leg  | `orientation_76` ~ `orientation_79` | **하체** |
| Left Lower Leg  | `orientation_80` ~ `orientation_83` | **하체** |
| Left Foot       | `orientation_84` ~ `orientation_87` | **하체** |
| Left Toe        | `orientation_88` ~ `orientation_91` | **하체** |

---

## 2. position — 세그먼트 위치 (69컬럼)

23개 세그먼트 × 3축 (x, y, z), 단위: m

| 세그먼트        | x               | y               | z               | 상/하체        |
| --------------- | --------------- | --------------- | --------------- | -------------- |
| Pelvis          | `position_0`  | `position_1`  | `position_2`  | 중심           |
| L5              | `position_3`  | `position_4`  | `position_5`  | 상체           |
| L3              | `position_6`  | `position_7`  | `position_8`  | 상체           |
| T12             | `position_9`  | `position_10` | `position_11` | 상체           |
| T8              | `position_12` | `position_13` | `position_14` | 상체           |
| Neck            | `position_15` | `position_16` | `position_17` | 상체           |
| Head            | `position_18` | `position_19` | `position_20` | 상체           |
| Right Shoulder  | `position_21` | `position_22` | `position_23` | 상체           |
| Right Upper Arm | `position_24` | `position_25` | `position_26` | 상체           |
| Right Forearm   | `position_27` | `position_28` | `position_29` | 상체           |
| Right Hand      | `position_30` | `position_31` | `position_32` | 상체           |
| Left Shoulder   | `position_33` | `position_34` | `position_35` | 상체           |
| Left Upper Arm  | `position_36` | `position_37` | `position_38` | 상체           |
| Left Forearm    | `position_39` | `position_40` | `position_41` | 상체           |
| Left Hand       | `position_42` | `position_43` | `position_44` | 상체           |
| Right Upper Leg | `position_45` | `position_46` | `position_47` | **하체** |
| Right Lower Leg | `position_48` | `position_49` | `position_50` | **하체** |
| Right Foot      | `position_51` | `position_52` | `position_53` | **하체** |
| Right Toe       | `position_54` | `position_55` | `position_56` | **하체** |
| Left Upper Leg  | `position_57` | `position_58` | `position_59` | **하체** |
| Left Lower Leg  | `position_60` | `position_61` | `position_62` | **하체** |
| Left Foot       | `position_63` | `position_64` | `position_65` | **하체** |
| Left Toe        | `position_66` | `position_67` | `position_68` | **하체** |

---

## 3. velocity — 세그먼트 선속도 (69컬럼)

23개 세그먼트 × 3축 (x, y, z), 단위: m/s
인덱스 구조는 position과 동일 (`velocity_0`~`velocity_68`)

| 세그먼트        | x                                | y               | z               | 상/하체        |
| --------------- | -------------------------------- | --------------- | --------------- | -------------- |
| Pelvis          | `velocity_0`                   | `velocity_1`  | `velocity_2`  | 중심           |
| L5 ~ Left Hand  | `velocity_3` ~ `velocity_44` | —              | —              | 상체           |
| Right Upper Leg | `velocity_45`                  | `velocity_46` | `velocity_47` | **하체** |
| Right Lower Leg | `velocity_48`                  | `velocity_49` | `velocity_50` | **하체** |
| Right Foot      | `velocity_51`                  | `velocity_52` | `velocity_53` | **하체** |
| Right Toe       | `velocity_54`                  | `velocity_55` | `velocity_56` | **하체** |
| Left Upper Leg  | `velocity_57`                  | `velocity_58` | `velocity_59` | **하체** |
| Left Lower Leg  | `velocity_60`                  | `velocity_61` | `velocity_62` | **하체** |
| Left Foot       | `velocity_63`                  | `velocity_64` | `velocity_65` | **하체** |
| Left Toe        | `velocity_66`                  | `velocity_67` | `velocity_68` | **하체** |

---

## 4. acceleration — 세그먼트 선가속도 (69컬럼)

23개 세그먼트 × 3축 (x, y, z), 단위: m/s²
인덱스 구조는 position/velocity와 동일 (`acceleration_0`~`acceleration_68`)

| 세그먼트        | x                                        | y                   | z                   | 상/하체        |
| --------------- | ---------------------------------------- | ------------------- | ------------------- | -------------- |
| Pelvis          | `acceleration_0`                       | `acceleration_1`  | `acceleration_2`  | 중심           |
| L5 ~ Left Hand  | `acceleration_3` ~ `acceleration_44` | —                  | —                  | 상체           |
| Right Upper Leg | `acceleration_45`                      | `acceleration_46` | `acceleration_47` | **하체** |
| Right Lower Leg | `acceleration_48`                      | `acceleration_49` | `acceleration_50` | **하체** |
| Right Foot      | `acceleration_51`                      | `acceleration_52` | `acceleration_53` | **하체** |
| Right Toe       | `acceleration_54`                      | `acceleration_55` | `acceleration_56` | **하체** |
| Left Upper Leg  | `acceleration_57`                      | `acceleration_58` | `acceleration_59` | **하체** |
| Left Lower Leg  | `acceleration_60`                      | `acceleration_61` | `acceleration_62` | **하체** |
| Left Foot       | `acceleration_63`                      | `acceleration_64` | `acceleration_65` | **하체** |
| Left Toe        | `acceleration_66`                      | `acceleration_67` | `acceleration_68` | **하체** |

---

## 5. angularVelocity — 세그먼트 각속도 (69컬럼)

23개 세그먼트 × 3축 (x, y, z), 단위: rad/s
인덱스 구조 동일 (`angularVelocity_0`~`angularVelocity_68`)

| 세그먼트        | x                                              | y                      | z                      | 상/하체        |
| --------------- | ---------------------------------------------- | ---------------------- | ---------------------- | -------------- |
| Pelvis          | `angularVelocity_0`                          | `angularVelocity_1`  | `angularVelocity_2`  | 중심           |
| L5 ~ Left Hand  | `angularVelocity_3` ~ `angularVelocity_44` | —                     | —                     | 상체           |
| Right Upper Leg | `angularVelocity_45`                         | `angularVelocity_46` | `angularVelocity_47` | **하체** |
| Right Lower Leg | `angularVelocity_48`                         | `angularVelocity_49` | `angularVelocity_50` | **하체** |
| Right Foot      | `angularVelocity_51`                         | `angularVelocity_52` | `angularVelocity_53` | **하체** |
| Right Toe       | `angularVelocity_54`                         | `angularVelocity_55` | `angularVelocity_56` | **하체** |
| Left Upper Leg  | `angularVelocity_57`                         | `angularVelocity_58` | `angularVelocity_59` | **하체** |
| Left Lower Leg  | `angularVelocity_60`                         | `angularVelocity_61` | `angularVelocity_62` | **하체** |
| Left Foot       | `angularVelocity_63`                         | `angularVelocity_64` | `angularVelocity_65` | **하체** |
| Left Toe        | `angularVelocity_66`                         | `angularVelocity_67` | `angularVelocity_68` | **하체** |

---

## 6. angularAcceleration — 세그먼트 각가속도 (69컬럼)

23개 세그먼트 × 3축 (x, y, z), 단위: rad/s²
인덱스 구조 동일 (`angularAcceleration_0`~`angularAcceleration_68`)

| 세그먼트        | x                                                      | y                          | z                          | 상/하체        |
| --------------- | ------------------------------------------------------ | -------------------------- | -------------------------- | -------------- |
| Pelvis          | `angularAcceleration_0`                              | `angularAcceleration_1`  | `angularAcceleration_2`  | 중심           |
| L5 ~ Left Hand  | `angularAcceleration_3` ~ `angularAcceleration_44` | —                         | —                         | 상체           |
| Right Upper Leg | `angularAcceleration_45`                             | `angularAcceleration_46` | `angularAcceleration_47` | **하체** |
| Right Lower Leg | `angularAcceleration_48`                             | `angularAcceleration_49` | `angularAcceleration_50` | **하체** |
| Right Foot      | `angularAcceleration_51`                             | `angularAcceleration_52` | `angularAcceleration_53` | **하체** |
| Right Toe       | `angularAcceleration_54`                             | `angularAcceleration_55` | `angularAcceleration_56` | **하체** |
| Left Upper Leg  | `angularAcceleration_57`                             | `angularAcceleration_58` | `angularAcceleration_59` | **하체** |
| Left Lower Leg  | `angularAcceleration_60`                             | `angularAcceleration_61` | `angularAcceleration_62` | **하체** |
| Left Foot       | `angularAcceleration_63`                             | `angularAcceleration_64` | `angularAcceleration_65` | **하체** |
| Left Toe        | `angularAcceleration_66`                             | `angularAcceleration_67` | `angularAcceleration_68` | **하체** |

---

## 7. footContacts — 발 접촉 이벤트 (4컬럼)

| 컬럼명             | 설명                    |
| ------------------ | ----------------------- |
| `footContacts_0` | Right Foot Heel Contact |
| `footContacts_1` | Right Foot Toe Contact  |
| `footContacts_2` | Left Foot Heel Contact  |
| `footContacts_3` | Left Foot Toe Contact   |

---

## 8. jointAngle — 역운동학 관절각 (66컬럼)

22개 관절 × 3축 (x: 굴곡/신전, y: 내전/외전, z: 내회전/외회전)

| 관절                     | x (굴곡/신전)               | y (내전/외전)               | z (내/외회전)               | 상/하체        |
| ------------------------ | --------------------------- | --------------------------- | --------------------------- | -------------- |
| jL5S1                    | `jointAngle_0`            | `jointAngle_1`            | `jointAngle_2`            | 상체           |
| jL4L3                    | `jointAngle_3`            | `jointAngle_4`            | `jointAngle_5`            | 상체           |
| jL1T12                   | `jointAngle_6`            | `jointAngle_7`            | `jointAngle_8`            | 상체           |
| jT9T8                    | `jointAngle_9`            | `jointAngle_10`           | `jointAngle_11`           | 상체           |
| jT1C7                    | `jointAngle_12`           | `jointAngle_13`           | `jointAngle_14`           | 상체           |
| jC1Head                  | `jointAngle_15`           | `jointAngle_16`           | `jointAngle_17`           | 상체           |
| jRightC7Shoulder         | `jointAngle_18`           | `jointAngle_19`           | `jointAngle_20`           | 상체           |
| jRightShoulder           | `jointAngle_21`           | `jointAngle_22`           | `jointAngle_23`           | 상체           |
| jRightElbow              | `jointAngle_24`           | `jointAngle_25`           | `jointAngle_26`           | 상체           |
| jRightWrist              | `jointAngle_27`           | `jointAngle_28`           | `jointAngle_29`           | 상체           |
| jLeftC7Shoulder          | `jointAngle_30`           | `jointAngle_31`           | `jointAngle_32`           | 상체           |
| jLeftShoulder            | `jointAngle_33`           | `jointAngle_34`           | `jointAngle_35`           | 상체           |
| jLeftElbow               | `jointAngle_36`           | `jointAngle_37`           | `jointAngle_38`           | 상체           |
| jLeftWrist               | `jointAngle_39`           | `jointAngle_40`           | `jointAngle_41`           | 상체           |
| **jRightHip**      | **`jointAngle_42`** | **`jointAngle_43`** | **`jointAngle_44`** | **하체** |
| **jRightKnee**     | **`jointAngle_45`** | **`jointAngle_46`** | **`jointAngle_47`** | **하체** |
| **jRightAnkle**    | **`jointAngle_48`** | **`jointAngle_49`** | **`jointAngle_50`** | **하체** |
| **jRightBallFoot** | **`jointAngle_51`** | **`jointAngle_52`** | **`jointAngle_53`** | **하체** |
| **jLeftHip**       | **`jointAngle_54`** | **`jointAngle_55`** | **`jointAngle_56`** | **하체** |
| **jLeftKnee**      | **`jointAngle_57`** | **`jointAngle_58`** | **`jointAngle_59`** | **하체** |
| **jLeftAnkle**     | **`jointAngle_60`** | **`jointAngle_61`** | **`jointAngle_62`** | **하체** |
| **jLeftBallFoot**  | **`jointAngle_63`** | **`jointAngle_64`** | **`jointAngle_65`** | **하체** |

---

## 9. jointAngleXZY — 관절각 (XZY 회전 순서, 66컬럼)

jointAngle과 동일 22개 관절, 축 순서만 다름 (x, z, y 순)
인덱스 구조 동일 (`jointAngleXZY_0`~`jointAngleXZY_65`)

| 관절                      | x                              | z                              | y                              | 상/하체        |
| ------------------------- | ------------------------------ | ------------------------------ | ------------------------------ | -------------- |
| jL5S1 ~ jLeftWrist (0~41) | `jointAngleXZY_0~11`         | —                             | —                             | 상체           |
| **jRightHip**       | **`jointAngleXZY_42`** | **`jointAngleXZY_43`** | **`jointAngleXZY_44`** | **하체** |
| **jRightKnee**      | **`jointAngleXZY_45`** | **`jointAngleXZY_46`** | **`jointAngleXZY_47`** | **하체** |
| **jRightAnkle**     | **`jointAngleXZY_48`** | **`jointAngleXZY_49`** | **`jointAngleXZY_50`** | **하체** |
| **jRightBallFoot**  | **`jointAngleXZY_51`** | **`jointAngleXZY_52`** | **`jointAngleXZY_53`** | **하체** |
| **jLeftHip**        | **`jointAngleXZY_54`** | **`jointAngleXZY_55`** | **`jointAngleXZY_56`** | **하체** |
| **jLeftKnee**       | **`jointAngleXZY_57`** | **`jointAngleXZY_58`** | **`jointAngleXZY_59`** | **하체** |
| **jLeftAnkle**      | **`jointAngleXZY_60`** | **`jointAngleXZY_61`** | **`jointAngleXZY_62`** | **하체** |
| **jLeftBallFoot**   | **`jointAngleXZY_63`** | **`jointAngleXZY_64`** | **`jointAngleXZY_65`** | **하체** |

---

## 10. jointAngleErgo / jointAngleErgoXZY — 에르고노믹 관절각 (36컬럼)

| 컬럼 그룹             | 인덱스 범위                                        | 컬럼 수 | 비고                 |
| --------------------- | -------------------------------------------------- | ------- | -------------------- |
| `jointAngleErgo`    | `jointAngleErgo_0` ~ `jointAngleErgo_17`       | 18      | 기본 에르고 관절각   |
| `jointAngleErgoXZY` | `jointAngleErgoXZY_0` ~ `jointAngleErgoXZY_17` | 18      | XZY 순 에르고 관절각 |

---

## 11. centerOfMass — 질량중심 (9컬럼)

| 컬럼명             | 물리량     | 축 |
| ------------------ | ---------- | -- |
| `centerOfMass_0` | CoM 위치   | x  |
| `centerOfMass_1` | CoM 위치   | y  |
| `centerOfMass_2` | CoM 위치   | z  |
| `centerOfMass_3` | CoM 속도   | x  |
| `centerOfMass_4` | CoM 속도   | y  |
| `centerOfMass_5` | CoM 속도   | z  |
| `centerOfMass_6` | CoM 가속도 | x  |
| `centerOfMass_7` | CoM 가속도 | y  |
| `centerOfMass_8` | CoM 가속도 | z  |

---

## 12. sensorFreeAcceleration — IMU 센서 중력 제거 가속도 (21컬럼)

7개 부착 센서 × 3축 (x, y, z), 단위: m/s²

| 센서 위치              | x                             | y                             | z                             |
| ---------------------- | ----------------------------- | ----------------------------- | ----------------------------- |
| Pelvis Sensor          | `sensorFreeAcceleration_0`  | `sensorFreeAcceleration_1`  | `sensorFreeAcceleration_2`  |
| Right Upper Leg Sensor | `sensorFreeAcceleration_3`  | `sensorFreeAcceleration_4`  | `sensorFreeAcceleration_5`  |
| Right Lower Leg Sensor | `sensorFreeAcceleration_6`  | `sensorFreeAcceleration_7`  | `sensorFreeAcceleration_8`  |
| Right Foot Sensor      | `sensorFreeAcceleration_9`  | `sensorFreeAcceleration_10` | `sensorFreeAcceleration_11` |
| Left Upper Leg Sensor  | `sensorFreeAcceleration_12` | `sensorFreeAcceleration_13` | `sensorFreeAcceleration_14` |
| Left Lower Leg Sensor  | `sensorFreeAcceleration_15` | `sensorFreeAcceleration_16` | `sensorFreeAcceleration_17` |
| Left Foot Sensor       | `sensorFreeAcceleration_18` | `sensorFreeAcceleration_19` | `sensorFreeAcceleration_20` |

---

## 13. sensorMagneticField — IMU 센서 자기장 (21컬럼)

7개 부착 센서 × 3축 (x, y, z), 단위: μT
인덱스 구조는 sensorFreeAcceleration과 동일 (`sensorMagneticField_0`~`sensorMagneticField_20`)

---

## 14. sensorOrientation — IMU 센서 방향 쿼터니언 (28컬럼)

7개 부착 센서 × 4 quaternion 성분 (q0, q1, q2, q3)

| 센서 위치              | q0                       | q1                       | q2                       | q3                       |
| ---------------------- | ------------------------ | ------------------------ | ------------------------ | ------------------------ |
| Pelvis Sensor          | `sensorOrientation_0`  | `sensorOrientation_1`  | `sensorOrientation_2`  | `sensorOrientation_3`  |
| Right Upper Leg Sensor | `sensorOrientation_4`  | `sensorOrientation_5`  | `sensorOrientation_6`  | `sensorOrientation_7`  |
| Right Lower Leg Sensor | `sensorOrientation_8`  | `sensorOrientation_9`  | `sensorOrientation_10` | `sensorOrientation_11` |
| Right Foot Sensor      | `sensorOrientation_12` | `sensorOrientation_13` | `sensorOrientation_14` | `sensorOrientation_15` |
| Left Upper Leg Sensor  | `sensorOrientation_16` | `sensorOrientation_17` | `sensorOrientation_18` | `sensorOrientation_19` |
| Left Lower Leg Sensor  | `sensorOrientation_20` | `sensorOrientation_21` | `sensorOrientation_22` | `sensorOrientation_23` |
| Left Foot Sensor       | `sensorOrientation_24` | `sensorOrientation_25` | `sensorOrientation_26` | `sensorOrientation_27` |

---

## 전체 컬럼 수 요약

| 컬럼 그룹              | 컬럼 수       | 세그먼트/관절 수 | 채널                           |
| ---------------------- | ------------- | ---------------- | ------------------------------ |
| orientation            | 92            | 23 세그먼트      | q0~q3                          |
| position               | 69            | 23 세그먼트      | x, y, z                        |
| velocity               | 69            | 23 세그먼트      | x, y, z                        |
| acceleration           | 69            | 23 세그먼트      | x, y, z                        |
| angularVelocity        | 69            | 23 세그먼트      | x, y, z                        |
| angularAcceleration    | 69            | 23 세그먼트      | x, y, z                        |
| footContacts           | 4             | 2 발             | Heel/Toe                       |
| jointAngle             | 66            | 22 관절          | x(r/e), y(abd/add), z(int/ext) |
| jointAngleXZY          | 66            | 22 관절          | x, z, y                        |
| jointAngleErgo         | 18            | 에르고 관절      | —                             |
| jointAngleErgoXZY      | 18            | 에르고 관절      | —                             |
| centerOfMass           | 9             | CoM              | pos/vel/acc × xyz             |
| sensorFreeAcceleration | 21            | 7 IMU 센서       | x, y, z                        |
| sensorMagneticField    | 21            | 7 IMU 센서       | x, y, z                        |
| sensorOrientation      | 28            | 7 IMU 센서       | q0~q3                          |
| **합계**         | **693** |                  |                                |
