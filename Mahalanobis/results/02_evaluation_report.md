# Mahalanobis Impairment Score — 교차 검증 리포트
## 파라미터
```json
{
  "scaling": "zscore",
  "pca_k_method": "kaiser",
  "pca_variance_ratio": 0.9,
  "pca_fixed_k": 10,
  "speed_filter": "all",
  "distance_metric": "mahalanobis",
  "mcd_support_fraction": 0.75
}
```
## 전체 OOF AUC-ROC
`0.4991`
## Fold별 AUC
- Fold 1: `0.6023`
- Fold 2: `0.2850`
- Fold 3: `0.5733`
- Fold 4: `0.6783`
- Fold 5: `0.5219`

## 그룹별 Impairment Score 통계
- **ACLD**: D_M mean=41.289 ± 32.143, Impairment=0.437 ± 0.493
- **ACLR**: D_M mean=58.445 ± 35.104, Impairment=0.622 ± 0.533
- **HA**: D_M mean=84.475 ± 182.867, Impairment=0.826 ± 2.161
