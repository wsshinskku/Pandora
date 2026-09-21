# Pandora — 한국어 안내

**독립적인 Open RAN xApp들을 개인화된 연합 계약으로 조정하는 실행 가능한 연구용 구현입니다.**

[English README](README.md) · [논문–코드 대응표](docs/paper-to-code.md) · [실험 재현 절차](docs/reproducibility.md) · [외부 RAN 연동](docs/integration.md)

이 저장소는 논문 *Pandora: Personalized Adaptive Network-Driven Open RAN Orchestration with Federated Contracts*의 **저자 공식 구현**입니다. 개인화 연합학습, 계약 생성, 실행 시점 투영, Python RAN 시뮬레이션, 실험 도구와 외부 RAN 연동 인터페이스를 제공합니다.

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
# 비교 방법 전체
pandora run --config configs/smoke.yaml --output runs/comparison --methods independent static adaptive scheduler-reference qos-reference pandora

# 개인화 / FL / 위험 보정 / 결합 제약 ablation
pandora ablate --config configs/smoke.yaml --output runs/ablations

# xApp 공격성 변화
pandora sweep --config configs/smoke.yaml --output runs/aggressiveness --values 0.5 1.0 1.5 --seeds 0 1 2

# 논문 규모의 실험 설정
pandora run --config configs/paper.yaml --output runs/paper-scale --seeds 0 1 2
```

`paper.yaml`에는 4개 사이트, 사이트별 20 UE와 2개 셀, 3600 슬롯, 30 슬롯 계약 갱신, 600 슬롯 FL 갱신, 기본 20개 seed가 들어 있습니다.

`scheduler-reference`는 xApp을 순서대로 선택하고, `qos-reference`는 큐 상태에 따라 자원과 우선순위를 조정합니다. 사용 가능한 비교 방법은 `independent`, `static`, `adaptive`, `scheduler-reference`, `qos-reference`, `pandora`입니다.

## 결과 확인

- `performance.png`: 실제 실행으로 생성한 throughput CDF, delay CCDF, 개입률.
- `summary.csv`: 평균 및 seed 단위 95% 신뢰구간.
- `metrics.json`: paired 차이, 모델 예측 오차, 위험 보정 및 지원 영역 진단.
- `telemetry.csv`: 슬롯별 KPI, 개입 여부, fallback 사용 이유.
- `manifest.json`: 실제 백엔드, 소스 해시, 의존성 버전, 완료 상태.
- `seed-*/site-*`: 로컬 에피소드, 개인화 모델 체크포인트, 계약 생성 진단.

단위는 Mbps, ms, Mbit이고 손실·위반·개입률은 0–1 비율입니다. 지연의 top-10%는 지연이 낮은 좋은 구간, bottom-10%는 지연이 높은 나쁜 구간입니다. seed가 하나면 신뢰구간은 `null`로 기록합니다.

보정된 위험 임계값을 통과한 후보가 없으면 fallback을 사용합니다. 계약 수락률·fallback 사용률·지원 영역 진단을 성능 지표와 함께 확인할 수 있습니다.

## 외부 시뮬레이터 연결

```bash
python examples/external_loop.py --run runs/first --output runs/external-audit.jsonl
```

별도 제어 프로세스와 Python RAN 환경을 JSONL로 연결합니다. ns-O-RAN의 telemetry 추출 및 scheduler/E2 제어 hook도 같은 경계에 연결할 수 있습니다. 제안한 행동과 실제 적용한 행동은 별도로 기록합니다.

위험 보정은 분리된 데이터의 잔차를 사용하고, 계약 생성은 유한한 후보 검증을 수행합니다. fallback은 실행 가능한 중립 행동을 유지합니다. 수식과 설정은 [상세 문서](docs/paper-to-code.md)에 정리했습니다.

## 개발 및 라이선스

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest -q
python -m build
```

코드는 [MIT License](LICENSE)를 따릅니다. 소프트웨어 인용 정보는 [CITATION.cff](CITATION.cff)에 있으며, 실험에 사용한 Git revision을 함께 기록합니다. ns-O-RAN과 QuaDRiGa는 각 프로젝트의 라이선스를 따릅니다.