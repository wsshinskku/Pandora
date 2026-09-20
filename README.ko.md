# Pandora — 한국어 안내

**독립적인 Open RAN xApp들을 개인화된 연합 계약으로 조정하는 실행 가능한 연구용 구현입니다.**

[English README](README.md) · [논문–코드 대응표](docs/paper-to-code.md) · [실험 재현 절차](docs/reproducibility.md) · [외부 RAN 연동](docs/integration.md)

이 코드는 제공된 익명 논문 *Pandora: Personalized Adaptive Network-Driven Open RAN Orchestration with Federated Contracts*를 바탕으로 새로 작성한 reference implementation입니다. 논문 저자의 공식 원본 코드라는 의미는 아닙니다.

핵심 알고리즘은 구현되어 있지만, 함께 제공되는 네트워크 환경은 **분석적 샌드박스 시뮬레이터**입니다. 원본 ns-3 수정 코드, QuaDRiGa 채널 트레이스, 학습 데이터가 제공되지 않았으므로 이 저장소에서 생성한 결과를 논문의 표 II–IV 재현 결과로 표시하면 안 됩니다.

## 핵심 아이디어

1. 자원 할당·트래픽 조향·QoS xApp이 각각 행동을 제안합니다.
2. 각 사이트의 공유 MLP와 개인화 어댑터가 후보 행동의 KPI 및 SLA 위험을 예측합니다.
3. 관측 데이터의 지원 영역과 보정된 위험 임계값을 통과한 후보들로 선형 계약을 구성합니다.
4. 계약 내부에서 검증 행동을 뽑아 확인하고, 필요하면 영역을 축소하거나 fallback을 사용합니다.
5. OSQP로 제안에 가장 가까운 실행 가능 행동을 구합니다.
6. 공유 가중치와 표본 수를 연합 평균하고, 개인화 파라미터와 원시 데이터는 사이트에 남깁니다.

## 설치 및 첫 실행

Python 3.11 이상을 사용합니다. GPU 없이 실행할 수 있습니다.

```bash
git clone https://github.com/wsshinskku/Pandora.git
cd Pandora
python -m venv .venv
```

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
source .venv/bin/activate
```

설치하고 테스트한 뒤 작은 실험을 실행합니다.

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
pandora run --config configs/smoke.yaml --output runs/first
```

실행하면 학습 4개·보정 1개·평가 1개 에피소드를 분리하여 수집하고, 학습·보정·계약 생성·투영·평가·그래프 생성을 수행합니다. 기존 실험을 덮어쓰지 않도록 매번 새로운 출력 경로를 사용합니다.

## 제공 기능

| 기능 | 내용 |
|---|---|
| 행동 공간 | 자원 비율 `rho`, UE별 합이 1인 조향 가중치 `nu`, QoS 우선순위 `omega` |
| 예측 모델 | 128–128–64 MLP, KPI 회귀 및 위험 분류 head |
| 개인화·FL | 사이트별 affine adapter, 표본 수 가중 FedAvg |
| 보정 | 표준화 최근접 거리, leave-one-out 95% 분위수, 단측 잔차 보정 |
| 계약 생성 | 128개 후보, 분위수 box와 결합 제약, 64개 검증 행동, 최대 3회 축소 |
| 투영 | 블록 가중치 1.0/1.5/2.0의 OSQP 최적화, 비어 있지 않은 fallback |
| 평가 | 평균·상위/하위 10%, SLA 초과, 개입률, AUROC/Brier/ECE, paired bootstrap |
| 외부 연동 | 체크포인트, JSONL 제어/실행확인 프로토콜, SINR CSV 변환 |

## 실험 명령

```bash
# 참고 비교 방법 전체
pandora run --config configs/smoke.yaml --output runs/comparison --methods independent static adaptive scheduler-reference qos-reference pandora

# 개인화 / FL / 위험 보정 / 결합 제약 ablation
pandora ablate --config configs/smoke.yaml --output runs/ablations

# xApp 공격성 변화
pandora sweep --config configs/smoke.yaml --output runs/aggressiveness --values 0.5 1.0 1.5 --seeds 0 1 2

# 논문 규모의 설정, 백엔드는 여전히 샌드박스
pandora run --config configs/paper.yaml --output runs/paper-scale --seeds 0 1 2
```

`paper.yaml`에는 4개 사이트, 사이트별 20 UE와 2개 셀, 3600 슬롯, 30 슬롯 계약 갱신, 600 슬롯 FL 갱신, 기본 20개 seed가 들어 있습니다. 규모를 크게 설정해도 ns-3 실험으로 바뀌지는 않습니다.

`scheduler-reference`와 `qos-reference`는 기능 비교용 휴리스틱이며 Sched-xApp·QACM 원 논문의 구현이 아닙니다. FRL-Slicing·SafeSlice는 원본 구현 정보 부족으로 포함하지 않았습니다. 이 차이는 영문 README와 대응표에도 명시했습니다.

## 결과 확인

- `performance.png`: 실제 실행으로 생성한 throughput CDF, delay CCDF, 개입률.
- `summary.csv`: 평균 및 seed 단위 95% 신뢰구간.
- `metrics.json`: paired 차이, 모델 예측 오차, 위험 보정 및 지원 영역 진단.
- `telemetry.csv`: 슬롯별 KPI, 개입 여부, fallback 사용 이유.
- `manifest.json`: 실제 백엔드, 소스 해시, 의존성 버전, 완료 상태.
- `seed-*/site-*`: 로컬 에피소드, 개인화 모델 체크포인트, 계약 생성 진단.

단위는 Mbps, ms, Mbit이고 손실·위반·개입률은 0–1 비율입니다. 지연의 top-10%는 지연이 낮은 좋은 구간, bottom-10%는 지연이 높은 나쁜 구간입니다. seed가 하나면 신뢰구간은 `null`로 기록합니다.

작은 데이터에서는 보정된 위험 임계값을 만족하는 후보가 없어 대부분 또는 전부 fallback을 사용할 수 있습니다. 구현은 이런 결과를 숨기거나 논문의 성능 숫자로 대체하지 않습니다.

## 외부 시뮬레이터 연결

```bash
python examples/external_loop.py --run runs/first --output runs/external-audit.jsonl
```

샌드박스를 외부 프로세스로 취급해 실제 JSONL 제어 프로토콜을 시험합니다. 실제 ns-O-RAN 연결에는 플랫폼의 telemetry 추출 및 scheduler/E2 제어 hook을 연결해야 합니다. 코드가 제안한 행동과 실제 적용한 행동을 별도로 기록하므로 실행 단계에서 달라진 행동을 잘못 학습하지 않습니다.

위험 보정과 유한 개수의 검증은 경험적 검사입니다. fallback이 비어 있지 않다는 사실도 모든 네트워크 상태에서 SLA를 보장한다는 뜻은 아닙니다. 논문에서 빠진 설정과 구현상 선택은 [상세 문서](docs/paper-to-code.md)에 정리했습니다.

## 개발 및 라이선스

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
```

저장소의 새 코드는 MIT 라이선스로 제공합니다. ns-O-RAN과 QuaDRiGa는 각 프로젝트의 라이선스를 따릅니다. 익명 원고의 DOI나 출판 상태는 임의로 기재하지 않았습니다.
