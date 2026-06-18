# ACL/ACLR 보행 연구에서 전체 waveform 분석 필요성 근거 정리

작성일: 2026-06-18

## 결론

`docs/ref_papers` 전체 사전연구를 확인한 결과, **전체 waveform 또는 연속적인 gait-cycle/stance-phase 분석이 필요하다고 직접 주장하거나, 실제 분석 방법으로 채택해 그 필요성을 설명한 논문이 있다.**

가장 강한 근거는 다음 세 논문이다.

1. **Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls**
2. **Gait asymmetries are exacerbated at faster walking speeds in individuals with acute anterior cruciate ligament reconstruction**
3. **Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls**

이 논문들은 공통적으로 peak knee flexion angle, peak moment, 특정 시점의 GRF처럼 **단일 peak 또는 discrete time point만 보는 접근이 mid-stance, late stance, stance 전반의 이상 패턴을 놓칠 수 있다**고 본다. 따라서 ACL/ACLR 환자의 보행 회복 또는 비정상 패턴을 설명할 때는 전체 waveform, 즉 시간 정규화된 보행 곡선 전체를 분석하는 것이 더 타당하다는 근거가 된다.

다만 논문별로 표현은 다르다. 일부는 "전체 gait cycle"보다 **stance phase 전체**를 대상으로 한다. 따라서 보고서에서 쓸 때는 "전체 보행 주기 waveform"이라고 일반화하기보다, **"적어도 stance phase 전반의 연속 waveform 분석이 필요하며, 본 데이터처럼 0-100% gait cycle waveform을 보유한 경우 전체 cycle 기반 분석으로 확장할 수 있다"**고 쓰는 것이 정확하다.

## 검색 범위와 판정 기준

- 검색 범위: `docs/ref_papers` 하위 PDF 53개 텍스트 추출본
- 중복 파일: `99_duplicates_or_alternate_versions`의 중복본은 독립 근거로 중복 계산하지 않음
- 주요 검색어: `waveform`, `entire stance`, `entire gait cycle`, `throughout stance`, `SPM`, `SPM1D`, `functional waveform`, `discrete variables`, `peak values`, `phase-specific`
- 직접 근거 판정: 연구 목적, 방법, 서론, 제한점에서 discrete/peak 분석의 한계를 언급하고 waveform/continuous/whole-stance 분석을 정당화한 경우
- 보조 근거 판정: 리뷰 또는 메타분석에서 peak/discrete 변수 중심 문헌의 한계, phase-specific metric 부족, 특정 gait phase 미탐색을 언급한 경우

## 직접 근거 논문

| 근거 강도 | 논문 | 분석 대상 | 핵심 내용 | 본 연구에 주는 의미 |
|---|---|---|---|---|
| 매우 강함 | Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls | vGRF, KFA, KEM, KAM의 stance-phase waveform | peak magnitude 대신 stance 전체 waveform을 분석해야 더 큰 통찰을 얻을 수 있다고 직접 설명 | ACLR 환자의 stiffened-knee, less dynamic waveform, bilateral adaptation을 peak 하나로 축약하면 정보 손실이 큼 |
| 매우 강함 | Gait asymmetries are exacerbated at faster walking speeds in individuals with acute anterior cruciate ligament reconstruction | vertical/posterior-anterior GRF waveform | SPM으로 GRF waveform 전체를 평가해야 single discrete time point가 놓치는 보행 특성을 볼 수 있다고 설명 | 속도 변화에 따른 ACLR 비대칭은 특정 peak가 아니라 stance 중 여러 구간에서 나타남 |
| 강함 | Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls | vGRF, KFA, KEM의 stance-phase waveform | functional ANOVA로 stance 전반을 비교했고, mid/late stance 차이가 중요하다고 설명 | 6-12개월 ACLR의 "대칭성 회복"이 실제 정상화가 아닐 수 있음을 전체 waveform으로 확인 |
| 강함 | Linking Gait Biomechanics and Daily Steps Post ACL-Reconstruction | vGRF, KEM, KFA stance waveform | daily steps와 관련된 보행 이상을 functional waveform 분석으로 비교 | 활동량과 보행 질의 관계도 peak 하나보다 stance 전반의 loading strategy로 해석해야 함 |
| 중간-강함 | Whether Patients with Anterior Cruciate Ligament Reconstruction Walking at a Fast Speed Show more Kinematic Asymmetries | 6-DOF knee kinematics over one gait cycle | SPM1D로 gait cycle의 어느 구간에서 ACLR limb와 intact limb가 다른지 분석 | fast walking에서 transverse-plane, tibial translation asymmetry가 특정 gait-cycle 구간에 국소적으로 나타남 |
| 중간 | Gait Biomechanics in Anterior Cruciate Ligament-Reconstructed Knees at Different Time Frames Postsurgery | walking/jogging의 1-100% gait cycle | 1% 단위로 전체 gait cycle을 비교 | 회복 시점별 차이가 peak가 아니라 여러 gait-cycle region에 분포할 수 있음을 보여줌 |

## 논문별 핵심 정리

### Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls

**원문 근거 발췌**

- "Functional mixed effects models"
- "throughout the stance phase"
- "entire stance phase instead of just discrete points"

**해석**

이 논문은 가장 직접적인 근거다. 저자들은 기존 ACLR 보행 연구가 peak vGRF, peak KFA 같은 discrete feature에 집중했다고 지적한다. 그러나 ACLR 환자의 차이는 peak에서만 나타나는 것이 아니라 stance phase 여러 구간에 분포한다. 특히 mid-stance와 late stance에서 KFA, KEM, vGRF 패턴이 다르게 나타나고, 이러한 차이는 cartilage composition이나 KOA risk와 연결될 수 있다고 설명한다.

**결과 측면**

- ACLR limb는 control보다 early stance에서 vGRF, KEM, KAM이 낮고, mid-late stance에서 vGRF와 KFA가 높게 나타났다.
- uninvolved limb도 control과 다르게 나타나 단순 contralateral 비교는 정상화 판단을 왜곡할 수 있다.
- ACLR group의 biomechanical waveform은 전반적으로 "less dynamic", 즉 peak가 낮고 곡선이 평평한 형태로 설명된다.

**의의**

이 논문은 "전체 waveform 분석이 왜 필요한가"에 대해 가장 쓰기 좋은 문헌이다. 우리 연구에서 101-point waveform 전체를 입력으로 쓰는 논리와 잘 맞는다. 특히 peak knee flexion angle 하나만으로는 ACLR 환자의 전체 stiffened-knee strategy, mid-stance loading, bilateral compensation을 충분히 설명할 수 없다는 배경 근거가 된다.

**참고 표기**

Büttner C. et al. *Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls*. Journal of Orthopaedic Research.

### Gait asymmetries are exacerbated at faster walking speeds in individuals with acute anterior cruciate ligament reconstruction

**원문 근거 발췌**

- "entirety of the GRF waveform"
- "single discrete time points"
- "across GRF waveforms"

**해석**

이 논문은 discrete predetermined time point로 GRF를 평가하는 기존 접근을 명확히 언급한 뒤, SPM을 사용해 GRF waveform 전체를 평가했다고 설명한다. 특히 저자들은 stance phase 전체의 GRF를 고려해야 single time point로 포착되지 않는 gait characteristic을 더 robust하게 평가할 수 있다고 본다.

**결과 측면**

- ACLR group은 walking speed가 증가할수록 vertical GRF asymmetry가 커졌다.
- vertical GRF 차이는 stance phase 대부분에서 나타났고, fast speed 조건에서 injured limb와 uninjured limb의 loading modulation 차이가 커졌다.
- 저자들은 fast walking이 기존 gait asymmetry를 드러내는 과제로 유용할 수 있다고 해석했다.

**의의**

이 논문은 "전체 waveform 분석"뿐 아니라 "속도 조건을 바꾸면 숨겨진 ACLR 보행 이상이 드러난다"는 근거도 준다. 우리 데이터가 slow/normal/fast 조건을 포함한다면, 단일 속도 또는 단일 peak feature보다 waveform x speed 조합이 더 설득력 있는 분석 단위가 된다.

**참고 표기**

Garcia S. A. et al. *Gait asymmetries are exacerbated at faster walking speeds in individuals with acute anterior cruciate ligament reconstruction*. Journal of Orthopaedic Research.

### Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls

**원문 근거 발췌**

- "throughout stance"
- "Functional analyses of variance"
- "entirety of stance"

**해석**

이 논문은 6개월과 12개월 ACLR 환자의 involved limb, contralateral limb, healthy control limb를 비교했다. 주요 방법은 functional ANOVA이며, vGRF, KFA, KEM을 stance 전체에서 비교했다. 저자들은 기존 연구가 early stance peak magnitude에 많이 집중했지만, mid-stance와 late-stance의 변화가 충분히 연구되지 않았다고 설명한다.

**결과 측면**

- ACLR 환자는 6개월과 12개월에 involved limb와 contralateral limb가 서로 더 비슷해졌지만, 두 limb 모두 control과는 달랐다.
- involved limb는 early stance에서 KFA가 낮고, second half of stance에서 KFA가 높아 knee flexion excursion이 감소한 패턴을 보였다.
- contralateral limb도 control과 다른 vGRF 변화를 보여, symmetry가 recovery를 뜻하지 않을 수 있음을 보였다.

**의의**

이 논문은 "대칭성"만 보면 회복처럼 보일 수 있지만, control waveform과 비교하면 양쪽 limb 모두 비정상일 수 있음을 보여준다. 따라서 ACLR 회복 분석에서는 peak symmetry 또는 LSI만으로 부족하고, limb-control 비교와 waveform region 분석이 필요하다.

**참고 표기**

Davis-Wilson H. C. et al. *Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls*. Medicine & Science in Sports & Exercise.

### Linking Gait Biomechanics and Daily Steps Post ACL-Reconstruction

**원문 근거 발췌**

- "functional waveform gait analyses"
- "throughout stance"
- "less dynamic vGRF waveform"

**해석**

이 논문은 ACLR 후 6-12개월 환자에서 daily step 수와 stance-phase gait biomechanics의 관련성을 보았다. peak 하나가 아니라 vGRF, KEM, KFA waveform을 stance 전체에서 분석했다.

**결과 측면**

- daily steps가 가장 적은 그룹은 weight acceptance에서 vGRF와 KEM이 낮고, KFA는 stance 전체에서 낮았다.
- 적은 활동량은 단순히 보행 횟수 문제가 아니라, ACLR limb loading strategy 자체의 변화와 함께 나타났다.

**의의**

보행 회복은 "얼마나 많이 걷는가"와 "어떻게 걷는가"가 함께 봐야 한다. 이때 "어떻게 걷는가"는 특정 peak보다 stance 전체의 force/angle/moment waveform으로 더 잘 설명된다.

**참고 표기**

Lisee C. et al. *Linking Gait Biomechanics and Daily Steps Post ACL-Reconstruction*. Medicine & Science in Sports & Exercise.

### Whether Patients with Anterior Cruciate Ligament Reconstruction Walking at a Fast Speed Show more Kinematic Asymmetries

**원문 근거 발췌**

- "SPM1D"
- "during one gait cycle"
- "exact affected parts"

**해석**

이 논문은 ACLR 환자의 6-DOF knee kinematics를 slow, normal, fast walking에서 측정하고 SPM1D로 분석했다. sagittal flexion/extension뿐 아니라 adduction/abduction, internal/external rotation, tibial translation까지 gait cycle 전반에서 어느 구간이 다른지 확인했다.

**결과 측면**

- fast walking에서 ACLR knee는 intact knee보다 external rotation, proximal tibial translation, posterior tibial translation asymmetry가 더 크게 나타났다.
- 차이는 2-6%, 7-8%, 38-43%, 50-61%, 92-96% gait cycle처럼 특정 구간에 분포했다.

**의의**

이 논문은 peak flexion angle만 보아서는 transverse-plane과 tibial translation의 구간별 이상을 놓칠 수 있음을 보여준다. 우리 연구가 다축 IMU waveform 또는 3D joint trajectory를 다룬다면, 전체 waveform 분석의 필요성을 kinematics 관점에서 뒷받침한다.

**참고 표기**

*Whether Patients with Anterior Cruciate Ligament Reconstruction Walking at a Fast Speed Show more Kinematic Asymmetries*. Orthopaedic Surgery.

### Gait Biomechanics in Anterior Cruciate Ligament-Reconstructed Knees at Different Time Frames Postsurgery

**원문 근거 발췌**

- "across the gait cycle"
- "1% to 100%"
- "regions of the gait cycle"

**해석**

이 논문은 walking과 jogging에서 kinematic/kinetic 변수를 1-100% gait cycle로 정규화하고, confidence interval curve를 이용해 차이가 나타나는 gait-cycle region을 찾았다. 최신 SPM이나 functional mixed model은 아니지만, peak 하나가 아니라 곡선의 어느 구간에서 차이가 생기는지 보는 접근이다.

**결과 측면**

- early ACLR group, mid ACLR group, late ACLR group의 보행 차이가 서로 다르게 나타났다.
- early ACLR은 loading을 줄이는 protective adaptation을 보였고, late ACLR은 frontal-plane loading 증가 등 다른 양상을 보였다.

**의의**

ACLR 후 시간 경과에 따른 보행 이상은 한두 개 peak 값으로 단순화하기 어렵다. gait cycle region별 변화가 다르기 때문에 전체 waveform 기반 비교가 회복 단계 해석에 유리하다.

**참고 표기**

Capin J. J. et al. *Gait Biomechanics in Anterior Cruciate Ligament-Reconstructed Knees at Different Time Frames Postsurgery*. Medicine & Science in Sports & Exercise.

## 보조 근거: 기존 peak/discrete 분석의 한계

| 논문 | 한계 언급 | waveform 분석 필요성과의 연결 |
|---|---|---|
| Movement Patterns of the Knee During Gait Following ACL Reconstruction - A Systematic Review and Meta-Analysis | review가 peak angle/moment 등 discrete variables에 제한되어 다른 gait phase를 보지 못했다고 명시 | 기존 evidence base 자체가 peak 중심이라 phase-specific abnormality를 놓칠 수 있음 |
| Time to normalization of gait following ACL reconstruction compared with healthy controls - A systematic review and meta-analysis | walking speed, peak KFA, peak KFM 중심이며 phase-specific metrics 부족을 limitation으로 언급 | 회복 시점 분석도 peak 중심이면 motor control mechanism을 충분히 설명하기 어려움 |
| Knee kinematics and joint moments during gait following anterior cruciate ligament reconstruction - A systematic review and meta-analysis | sagittal-plane 중심, transverse-plane 표준화 부족, 일부 변수는 peak 중심 | 다축/다평면 waveform 분석이 향후 연구에서 더 필요함 |
| Progressive Changes in Walking Kinematics and Kinetics After Anterior Cruciate Ligament Injury and Reconstruction - A Review and Meta-Analysis | time from injury/reconstruction, GRF 등 추가 고려 필요 | longitudinal recovery를 더 포괄적으로 보려면 waveform과 GRF 포함이 유리함 |

### Movement Patterns of the Knee During Gait Following ACL Reconstruction - A Systematic Review and Meta-Analysis

**핵심**

이 리뷰는 meta-analysis가 discrete variables, 즉 peak angles와 peak moments 중심이었다고 밝힌다. 제한점에서는 다른 gait-cycle phase의 biomechanical variable을 탐색하지 못했다고 쓴다.

**우리 연구에 주는 의미**

기존 리뷰 수준의 근거가 peak 중심이라는 점 자체가 새로운 waveform 분석의 필요성을 만든다. 즉 "기존 문헌은 ACLR 보행 이상을 주로 peak로 요약했기 때문에, waveform 기반 접근은 누락된 phase-specific 정보를 보완한다"고 쓸 수 있다.

**참고 표기**

Kaur M. et al. *Movement Patterns of the Knee During Gait Following ACL Reconstruction: A Systematic Review and Meta-Analysis*.

### Time to normalization of gait following ACL reconstruction compared with healthy controls - A systematic review and meta-analysis

**핵심**

이 논문은 walking speed, peak knee flexion angle, peak knee flexion moment의 정상화 시점을 추정했다. 동시에 phase-specific metric의 표준화된 보고 부족이 motor control mechanism을 더 포괄적으로 탐색하는 데 제한이라고 설명한다.

**우리 연구에 주는 의미**

peak KFA와 peak KFM은 중요한 대표 지표지만, ACLR 회복의 전체 보행 전략을 설명하기에는 부족하다. 따라서 이 논문은 직접적인 waveform 주장 논문은 아니지만, peak 중심 정상화 분석의 빈틈을 보여주는 배경 근거로 적합하다.

**참고 표기**

Chen S. et al. *Time to normalization of gait following ACL reconstruction compared with healthy controls: A systematic review and meta-analysis*.

## IMU/ML 관점의 관련 근거

| 논문 | 관련 내용 | 해석 |
|---|---|---|
| The COMPWALK-ACL - A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients | raw time-series, joint-specific waveforms, 101-point time-normalized trajectories 제공 | 전체 waveform 분석을 실제로 수행할 수 있는 데이터 구조를 제시 |
| A Deep Learning-Based Framework Oriented to Pathological Gait Recognition with Inertial Sensors | DL은 raw data에서 직접 학습하여 manual feature extraction을 줄일 수 있다고 설명 | ACL 특이 근거는 아니지만, IMU time-series 전체를 feature engineering 없이 활용하는 ML/DL 논리와 연결 |
| Learning based lower limb joint kinematic estimation using open source IMU data | continuous recordings 기반 kinematics estimation | IMU 기반 연속 신호가 joint kinematics 추정에 유용하다는 기술적 배경 |

### The COMPWALK-ACL - A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients

**원문 근거 발췌**

- "joint-specific waveforms"
- "time-normalized to 101 points"
- "comprehensive time-series data"

**해석**

이 논문은 "전체 waveform을 분석해야 한다"는 주장을 직접 전개하는 연구라기보다, ACLD/ACLR/healthy cohort에서 multi-pace IMU gait kinematics를 제공하고, joint angle trajectory를 101-point time-normalized waveform으로 분석할 수 있음을 보여준다.

**의의**

우리 데이터 및 분석 방향과 가장 직접적으로 맞닿는 데이터셋 근거다. 특히 slow/normal/fast 속도, injured side 정보, raw sensor signal, joint kinematics가 함께 제공되므로, peak feature extraction보다 waveform 기반 분석을 정당화하기 좋다.

**참고 표기**

*The COMPWALK-ACL: A Dataset of Multi-pace IMU Gait Kinematics in Adolescents, Adults, and ACL Injured Patients*.

## 보고서에 바로 쓸 수 있는 문장

1. 기존 ACLR 보행 연구들은 peak knee flexion angle, peak knee flexion moment, peak GRF와 같은 discrete features를 자주 사용했지만, 최근 연구들은 ACLR 환자의 보행 이상이 stance phase의 여러 구간에 분포한다고 보고하였다.

2. Büttner et al.은 ACLR 환자의 vGRF, KFA, KEM, KAM을 stance phase 전반의 waveform으로 분석해야 discrete point만으로는 포착하기 어려운 bilateral compensation과 less dynamic gait profile을 설명할 수 있다고 제시하였다.

3. Garcia et al.은 GRF의 특정 시점만 평가하는 대신 SPM을 이용해 waveform 전체를 분석했으며, 빠른 보행 조건에서 ACLR 환자의 asymmetry가 더 뚜렷하게 나타남을 보였다.

4. 따라서 본 연구에서 101-point gait waveform을 사용한 분석은 단순 feature 확장이 아니라, ACLR 보행 이상이 시간적으로 국소화되고 phase-specific하게 나타난다는 선행연구의 문제의식에 근거한다.

5. 특히 peak KFA가 정상화되거나 통계적으로 차이가 작아 보이는 경우에도, mid-stance 또는 late-stance의 waveform shape, tibial rotation/translation, GRF loading pattern은 여전히 비정상일 수 있다.

## 최종 판단

**있다.** 전체 waveform 또는 stance/gait-cycle 전반의 연속 분석 필요성을 가장 명확하게 주장한 논문은 **Bilateral waveform analysis of gait biomechanics presurgery to 12 months following ACL reconstruction compared to controls**와 **Gait asymmetries are exacerbated at faster walking speeds in individuals with acute anterior cruciate ligament reconstruction**이다.

그리고 **Bilateral Gait Six and Twelve Months Post-ACL Reconstruction Compared to Controls**, **Linking Gait Biomechanics and Daily Steps Post ACL-Reconstruction**, **Whether Patients with Anterior Cruciate Ligament Reconstruction Walking at a Fast Speed Show more Kinematic Asymmetries**는 같은 방향의 직접 근거를 제공한다.

리뷰/메타분석 논문들은 "전체 waveform을 반드시 분석해야 한다"는 직접 주장보다는, 기존 연구가 peak/discrete variable 중심이어서 다른 gait phase나 phase-specific mechanism을 충분히 설명하지 못했다는 보조 근거로 사용하는 것이 적절하다.
