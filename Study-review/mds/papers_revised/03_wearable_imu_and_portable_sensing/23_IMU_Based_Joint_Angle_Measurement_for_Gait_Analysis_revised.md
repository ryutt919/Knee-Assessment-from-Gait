# IMU-Based Joint Angle Measurement for Gait Analysis

Seel, T., Raisch, J., & Schauer, T. (2014). IMU-Based Joint Angle Measurement for Gait Analysis. Sensors, 14(4), 6891-6909. https://doi.org/10.3390/s140406891

## 서지정보

- 저자: Thomas Seel, Jörg Raisch and Thomas Schauer
- 연도: 2014
- 저널: Sensors
- DOI: 10.3390/s140406891
- 원본 파일: /Users/ryutt/Desktop/mini_ryutt/Walking/docs/ref_papers/03_wearable_imu_and_portable_sensing/IMU-Based Joint Angle Measurement for Gait Analysis.pdf
- 분석 provider: antigravity

## 연구 목적

- 센서가 인체 세그먼트에 대해 부착된 특정 방향을 가정하지 않고 관성 측정 데이터를 기초로 관절 각도를 계산하는 방법을 제안한다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “This contribution is concerned with joint angle calculation based on inertial
measurement data in the context of human motion analysis. Unlike most robotic
devices, the human body lacks even surfaces and right angles. Therefore, we focus
on methods that avoid assuming certain orientations in which the sensors are mounted
with respect to the body segments.”

## 연구 설계와 대상

- 대퇴부 절단 환자의 보행 시험 데이터를 활용하여 광학식 3차원 동작 포착 시스템과 관성 측정 장치(IMU) 기반 방법을 비교하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “We provide results from gait trials of a transfemoral amputee in which we
compare the inertial measurement unit (IMU)-based methods to an optical 3D motion
capture system.”
- 대퇴부 절단 환자의 양쪽 다리(보철물 측 및 대조측 다리)의 대퇴부, 하퇴부, 발에 탄성 몸체 스트랩을 사용하여 각 세그먼트 당 1개의 IMU(Xsens MTw)를 위치나 방향 제한 없이 부착하였다. _(근거: PAGE 14, Section 4. Experimental Results and Discussion)_
  - 근거 원문: “Furthermore, we use elastic body straps to equip the upper and lower leg, as well as the foot, of both the prosthesis and the contralateral leg with one inertial measurement unit (Xsens MTw [1]) each, as depicted in Figure 7.”

## 방법

- 비선형 최소제곱 오차 함수를 최소화하는 가우스-뉴턴 알고리즘 또는 기타 표준 최적화 방식을 사용하여 관절 축의 방향 및 위치 좌표를 식별한다. _(근거: PAGE 8, Section 3.1.1)_
  - 근거 원문: “This optimization might be implemented
using a Gauss-Newton algorithm, as further described in [12], or any other standard optimization
method [31].”
- 관절 위치 벡터를 찾기 위해 비선형 최소제곱 기법인 가우스-뉴턴 알고리즘을 사용해 오차를 최소화한다. _(근거: PAGE 10, Section 3.1.3)_
  - 근거 원문: “We minimize Ψ̃(o1,o2) over its arguments via a Gauss-Newton algorithm, the
implementation of which is described in [12].”
- 자이로스코프 기반 각도와 가속도 기반 각도를 결합하기 위해 상보 필터나 칼만 필터와 같은 센서 융합 도구를 사용한다. _(근거: PAGE 12, Section 3.2.2)_
  - 근거 원문: “Therefore, it is advantageous to combine both angles using a standard tool of sensor fusion, e.g., a complementary filter [32] or a Kalman filter.”

## 핵심 결과

- 모든 시험에서 두 가지 IMU 기반 방법은 관성 데이터를 완전히 다른 방식으로 사용함에도 불구하고 유사한 값을 도출했다. _(근거: PAGE 15, Section 4. Experimental Results and Discussion)_
  - 근거 원문: “In all trials, both IMU-based approaches yield similar values, although they use the inertial data
in completely different ways.”
- 인체 다리에서의 무릎 각도 측정 오차는 보철물 측 오차보다 약 4배 더 컸다. _(근거: PAGE 16, Section 4. Experimental Results and Discussion)_
  - 근거 원문: “It is important to note that the errors on the human leg are about four times larger than on
the prosthesis.”
- > **[AS-IS]** 보철물 측과 대조측 모두 발목관절 저측/배측 굴곡 각도 측정에서 편차가 약 1도 내외였다. _(근거: PAGE 15, Figure 9 Caption)_
>
> **[TO-BE]** 발목 저측/배측 굴곡 각도 측정에서 보철물 측 평균 RMSE는 0.81도, 대조측 평균 RMSE는 1.62도였으며, 본문과 그림 설명에서는 전체적으로 약 1도 수준의 편차로 설명되었다.
>
> _(사실검증 — 수치오류/경미: Figure 9 캡션은 양쪽 모두 약 1도라고 표현하지만, Table 1의 6회 시험 평균은 보철물 0.81도, 대조측 1.62도로 제시된다. '대조측도 약 1도 내외'라고 단정하면 표의 평균값을 충분히 반영하지 못한다.)_
  - 근거 원문: “Both on the prosthesis side and on the contralateral side, the deviation is about 1◦
.”

## 저자 결론

- 관절의 운동학적 구속조건을 활용하여 임의의 동작 데이터를 통해 관절 축의 방향 및 위치 좌표를 추정하는 방식은 이전에 제안된 보정 자세나 동작이 필요한 방식보다 실용적이고 강건하다. _(근거: PAGE 16, Section 5. Conclusions)_
  - 근거 원문: “We proposed a set of methods that allow us to determine the local joint axis and position coordinates
from arbitrary motions by exploitation of the kinematic constraints of the joint.”
- 가속도계와 자이로스코프만 사용하는 방법은 지자기 센서를 배제하므로 실내 환경이나 자기 왜곡이 있는 곳에서도 사용 가능하다. _(근거: PAGE 16, Section 5. Conclusions)_
  - 근거 원문: “The second
and novel method employs only accelerometer and gyroscope readings. Since the use of magnetometers
is avoided, it can be used indoors and in the proximity of magnetic disturbances.”

## 연구의 한계

- 인체 다리에서는 근육과 피부의 운동으로 인해 관성 센서와 마커가 서로에 대해 상대적으로 움직이며, 이는 측정 오차의 원인이 된다. _(근거: PAGE 16, Section 4. Experimental Results and Discussion)_
  - 근거 원문: “However, on the human leg, the inertial sensors and the markers move relative to each other as a result
of muscle and skin motions.”
- 향후 연구는 연부 조직 움직임으로 인한 오차 효과를 어떻게 보상하거나 최소화할 것인가에 집중될 것이다. _(근거: PAGE 17, Section 5. Conclusions)_
  - 근거 원문: “Future research will be dedicated to the question of how these effects can be
compensated for or minimized.”

## 생각해볼 내용

- 선행 연구들이 광학 마커를 IMU에 직접 부착하여 연부조직 아티팩트를 우회적으로 피했던 것과 달리, 본 논문은 실제 해부학적 랜드마크에 마커를 배치함으로써 더 현실적이고 가혹한 환경에서 정밀도를 검증하였다. _(근거: PAGE 1, Abstract)_
  - 근거 원문: “Unlike most authors, we place the optical markers on anatomical landmarks instead of
attaching them to the IMUs.”
- > **[AS-IS]** 마그네토미터를 제외하고 자이로스코프와 가속도계만 사용해 구속조건 식별 및 각도 측정을 성공적으로 완수한 설계 방식은 실생활이나 자성 왜곡이 흔한 병원 실내 임상 시험 환경에서 대단히 유용할 것으로 평가된다. _(근거: PAGE 1, Abstract)_
>
> **[TO-BE]** 마그네토미터를 제외하고 자이로스코프와 가속도계만 사용하도록 설계되어, 균질한 자기장에 의존하지 않는다는 점이 장점으로 제시된다.
>
> _(사실검증 — 과장/경미: 원문은 자이로스코프와 가속도계만 사용하여 균질한 자기장에 의존하지 않는 방법을 제안한다고 설명한다. '실생활이나 자성 왜곡이 흔한 병원 실내 임상 시험 환경에서 대단히 유용'하다는 평가는 원문보다 적용 맥락과 효용을 강하게 확장한 해석이다.)_
  - 근거 원문: “In particular, we propose methods that use
only gyroscopes and accelerometers and, therefore, do not rely on a homogeneous magnetic
field.”

## 이 연구가 지적한 선행연구의 문제점

- 일부 문헌에서는 관성 센서가 미리 정의된 방향으로 정밀하게 정렬되어 부착될 수 있다고 현실과 동떨어진 가정을 하거나 정렬 오차 문제를 아예 무시한다. _(근거: PAGE 3, Section 1.3)_
  - 근거 원문: “First, we shall
note that in some publications, this problem is ignored completely by assuming that the IMUs can be
mounted precisely in a predefined orientation towards the joint; see, e.g., [9,10].”
- 센서 부착 방향 및 위치 정보를 수동으로 계측하는 방식은 3차원 공간에서 매우 번거로우며 오차가 발생하기 쉽다. _(근거: PAGE 3, Section 1.3)_
  - 근거 원문: “Both quantities might be
measured manually, but in three-dimensional space, this is a cumbersome task that yields low accuracy
results, as demonstrated, e.g., in [9,12].”
- 자세 보정이나 특정 교정 움직임을 활용하는 방식은 피험자가 지시 동작을 수행하는 정밀도에 의해서 그 보정의 정확도가 제한을 받는다. _(근거: PAGE 4, Section 1.3)_
  - 근거 원문: “However, it
is important to note that, both in calibration postures and calibration motions, the accuracy is limited by
the precision with which the subject can perform the postures or motions.”
- 지자기 센서 데이터는 강자성 물질 등에 의한 자기 왜곡에 의해 방위각(heading) 추정 시 정확도 저하가 발생할 수 있다. _(근거: PAGE 2, Section 1.1)_
  - 근거 원문: “Therefore, the presence of magnetic disturbances (as induced, e.g., by ferromagnetic material) may limit the accuracy of the orientation estimates, as demonstrated in [5,6].”

## 이 연구의 해결 방식과 기여

- 관절의 기하학적/운동학적 제약조건을 비용 함수 최소화에 활용하여 특정 정렬이나 보정 동작 없이도 임의의 움직임 데이터만으로 관절 축 및 위치 좌표를 자동 식별하는 기법을 개발했다. _(근거: PAGE 7, Section 3.1)_
  - 근거 원문: “However, these coordinates can
be identified from the measurement data of arbitrary motions by exploiting kinematic constraints, as
explained in [12].”
- 제안된 모든 방법이 마그네토미터 데이터를 일체 사용하지 않도록 하여 왜곡이 빈번한 환경에서도 신뢰성을 유지할 수 있도록 하였다. _(근거: PAGE 7, Section 3)_
  - 근거 원문: “All of the methods that we will introduce use only angular rates and accelerations, while the use of magnetometer readings is completely avoided.”
- > **[AS-IS]** 복잡한 수동 측정이나 캘리브레이션 자세/동작 없이 센서를 부착하고 간단히 다리를 움직이기만 하면 실시간 각도를 출력하는 플러그 앤 플레이 방식의 혁신적 보행 분석 환경을 제공한다. _(근거: PAGE 17, Section 5. Conclusions)_
>
> **[TO-BE]** 복잡한 수동 측정이나 정밀한 캘리브레이션 자세/동작을 대체할 수 있어, 센서를 부착하고 몇 초간 다리를 움직인 뒤 실시간 관절각을 얻는 플러그 앤 플레이 보행 분석으로 이어질 가능성을 제시한다.
>
> _(사실검증 — 과장/경미: 원문은 이러한 방법이 플러그 앤 플레이 보행 분석의 가능성을 열며, 온라인 사용 구현과 실시간 측정은 향후 연구 주제라고 서술한다. 요약의 '제공한다'와 '혁신적'은 현재 완성된 환경을 이미 제공한 것처럼 표현해 원문보다 강하다.)_
  - 근거 원문: “these new methods open the door to a plug-and-play gait analysis, in which one simply attaches the IMUs, moves the legs for a few seconds and then receives joint angle measurements in real time.”

## 레퍼런스할 수 있는 내용

### 1. IMU 기반 힌지 관절 각도 측정의 가능성

- 원문 발췌: “It has been demonstrated in many publications, e.g., [7] and the references therein, that inertial measurement data can be used to calculate hinge joint angles when at least one IMU is attached to each side of the joint.”
- 한국어 번역: 관절의 각 측면에 적어도 하나의 IMU가 부착되었을 때 관성 측정 데이터를 사용하여 힌지 관절 각도를 계산할 수 있음이 [7] 및 그 안의 참고문헌 등 많은 문헌에서 입증되었다.
- 원문 위치: PAGE 2, Section 1.2
- 원문 내 인용표기: [7]
- 해당 선행문헌: 7. Cheng, P.; Oelmann, B. Joint-Angle Measurement Using Accelerometers and Gyroscopes—A
Survey. IEEE Trans. Instrum. Meas. 2010, 59, 404–414.
- 주장 유형: background_citation
- 활용 맥락과 주의: 관절 양측에 IMU를 부착하여 힌지 관절 각도를 계산하는 선행 연구들의 일반적인 배경으로 인용 가능함. 원인용 문헌인 Cheng & Oelmann (2010)을 직접 검토하여 2차 인용에 주의할 것.

### 2. 지자기 센서 오차 요인으로서의 자기 교란

- 원문 발췌: “Therefore, the presence of magnetic disturbances (as induced, e.g., by ferromagnetic material) may limit the accuracy of the orientation estimates, as demonstrated in [5,6].”
- 한국어 번역: 따라서 강자성 물질 등에 의해 유발되는 자기 교란의 존재는 [5,6]에서 입증된 바와 같이 방향 추정의 정확도를 제한할 수 있다.
- 원문 위치: PAGE 2, Section 1.1
- 원문 내 인용표기: [5,6]
- 해당 선행문헌: 5. Bachmann, E.; Yun, X.; Brumfield, A. Limitations of Attitude Estimation Algorithms for
Inertial/Magnetic Sensor Modules. IEEE Robot. Autom. Mag. 2007, 14, 76–87.
6. De Vries, W.H.; Veeger, H.E.; Baten, C.T.; van der Helm, F.C. Magnetic distortion in motion labs,
implications for validating inertial magnetic sensors. Gait Posture 2009, 29, 535–541.
- 주장 유형: background_citation
- 활용 맥락과 주의: 자기 교란이 지자기 센서 기반 방향 추정(특히 azimuth/heading)의 오차를 초래할 수 있음을 서술하는 선행 연구 배경으로 인용 가능함. 원인용 문헌인 Bachmann (2007) 또는 De Vries (2009)를 참고할 것.

### 3. 제안된 무릎 굴곡/신전 각도 오차 결과

- 원문 발췌: “Root mean square errors of the knee flexion/extension angles
are found to be less than 1◦
on the prosthesis and about 3◦
on the human leg.”
- 한국어 번역: 무릎 굴곡/신전 각도의 제곱평균제곱근 오차는 보철물에서 1도 미만, 실제 사람 다리에서 약 3도이다.
- 원문 위치: PAGE 1, Abstract
- 원문 내 인용표기: 해당 없음
- 해당 선행문헌: 해당 없음
- 주장 유형: analyzed_paper_finding
- 활용 맥락과 주의: 본 논문의 핵심 실험 결과로, 제안한 알고리즘의 적용 시 보철물 다리와 생체 다리에서의 무릎 각도 오차 범위를 지칭할 때 인용할 수 있음.
