# Chess Engine (ML Focus)

이 프로젝트는 체스 기보(PGN)를 토큰화하여 Transformer로 학습/평가하는 ML 파이프라인에 초점을 맞춥니다.

## 목표
2) Transformer 구현 (PyTorch)
- 가중치 저장/불러오기, 하이퍼파라미터 유연성, 파인 튜닝 지원

3) 실험 모델 ID 생성/식별
- 규칙: `모델명-레이어개수_헤드개수_d_ff크기_초기화타입-learningRate-lossType-epoch`
- 예: `transformer-6_8_2048_xavier-0.001-crossentropy-100`

---

## 디렉터리 구조(ML)

```
ml/
  ├── __init__.py
  ├── config.py         # ModelConfig/TrainConfig
  ├── tokenizer.py      # CharTokenizer / SimpleMoveTokenizer
  ├── dataset.py        # PGNDataset(.pgn → 시퀀스)
  ├── model.py          # TransformerDecoderLM (decoder-only)
  ├── train.py          # 학습 CLI (저장/불러오기 지원)
  └── finetune.py       # 체크포인트 재개 기반 파인튜닝
chessdata/              # .pgn 데이터 (사용자 제공)
requirements.txt        # Python 의존성
```

---

## 설치
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 다른 컴퓨터에서 빠르게 실행하기 (Getting Started)
- Python 3.10+ 권장, CUDA 환경이면 자동으로 `--device cuda`가 사용됩니다.

```bash
# 1) 리포지토리 클론
git clone <repo-url>
cd chess_engine

# 2) 가상환경 및 의존성 설치
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3) 데이터 준비 (작은 PGN 1~2개로 테스트 가능)
mkdir -p chessdata
cp /path/to/sample.pgn chessdata/

# 4) 학습 실행 (검증 포함)
python -m ml.train --files chessdata/sample.pgn --epochs 3 --batch-size 8

# 5) 모니터링 결과 생성
python unified_model_monitor.py --latest-only

# 6) 결과 위치
echo "See ./model_result_monitoring/<model_id>/plots/*.png"
```

GPU가 있는 머신에서는 다음 옵션을 권장합니다:
```bash
python -m ml.train --files chessdata/sample.pgn --epochs 10 --batch-size 32 --device cuda
```

학습 중 불법 수 패널티/패배 측 마스킹을 적용하려면:
```bash
python -m ml.train --files chessdata/sample.pgn --epochs 3 --penalize-illegal --mask-loser
```

## 데이터
- `chessdata/` 폴더에 `.pgn` 파일을 넣어주세요. (헤더 포함 PGN 형식)
- 초기에 토크나이저는 최대 50개 파일 샘플로 vocab을 구성합니다.

## 학습
```bash
python -m ml.train \
  --data-dir chessdata \
  --epochs 1 \
  --batch-size 8 \
  --d-model 256 --n-heads 4 --n-layers 4 --d-ff 1024 \
  --max-seq-len 256
```

또는 특정 PGN 파일만 골라 학습하려면:
```bash
python -m ml.train \
  --files chessdata/Adams.pgn chessdata/Abdusattorov.pgn \
  --epochs 1 --batch-size 8 \
  --d-model 256 --n-heads 4 --n-layers 4 --d-ff 1024 \
  --max-seq-len 256
```

검증셋 없이 빠르게 학습:
```bash
python -m ml.train --files chessdata/Adams.pgn --epochs 100 --no-validation --device gpu
```

**영구 기록 보관:**
```bash
# --record 플래그로 타임스탬프 포함 영구 백업
python -m ml.train --files chessdata/Adams.pgn --epochs 200 --model-id important_exp --record --device gpu
```
→ `experiments_record/{model_id}_{timestamp}/` 폴더에 저장:
  - 체크포인트 (best.pt, last.pt)
  - 손실 데이터 (graph/)
  - 모니터링 결과 (monitoring/)
  - 메타데이터 (하이퍼파라미터, 데이터 정보)

- 체크포인트 저장 경로: `checkpoints/{model_id}/{best.pt,last.pt}`
- 모델 ID 자동 생성: `transformer-{layers}_{heads}_{d_ff}_default-{lr}-ce-{epochs}`
- 각 학습마다 고유한 모델 ID 폴더에 저장되어 이전 실험 결과 유지
- 재개 학습:
```bash
python -m ml.train --data-dir chessdata --resume checkpoints/best.pt --epochs 1
```

## 모델 불러오기

모델 ID로 불러오기:
```bash
# 사용 가능한 모든 모델 조회
python -m ml.load_model --list

# 모델 ID로 로드
python -m ml.load_model --load-id transformer-4_4_1024_default-0.0003-ce-100

# 체크포인트 경로로 직접 로드
python -m ml.load_model --load-path checkpoints/best.pt
```

Python 코드에서:
```python
from ml.load_model import load_model_by_id, load_model_from_checkpoint, list_available_models

# 1. 모델 ID로 검색 및 로드
model, ckpt = load_model_by_id('transformer-4_4_1024_default-0.0003-ce-100')
vocab = ckpt['vocab']

# 2. 경로로 직접 로드
model = load_model_from_checkpoint('checkpoints/best.pt')

# 3. 사용 가능한 모델 목록
models = list_available_models()
for m in models:
    print(m['model_id'], m['path'])
```

## 학습 모니터링

학습 중 loss 데이터가 `graph/{model_id}/` 폴더에 자동으로 기록됩니다:
- `epoch-loss.txt`: `timestamp,epoch,avg_loss,val_loss` CSV 형식
  - `timestamp`: `YYYY-MM-DD HH:MM:SS`
  - `avg_loss`: 학습 에폭 평균 손실
  - `val_loss`: 검증 손실(검증이 없으면 빈 값)
- `batch-loss.txt`: `timestamp,epoch,batch_num,loss` CSV 형식

**통합 모니터링 시스템**으로 분석 및 시각화:
```bash
# 모든 실험 분석
python unified_model_monitor.py

# 최신 실험만 분석
python unified_model_monitor.py --latest-only

# 특정 실험만 분석
python unified_model_monitor.py --experiments first transformer-4_4_1024_default-0.0003-ce-200
```

트러블슈팅(다른 머신):
- `ImportError`: `pip install -r requirements.txt`를 재확인하고, 시스템 Python과 venv가 섞이지 않았는지 확인.
- `ModuleNotFoundError: ml.xxx`: 반드시 리포지토리 루트에서 실행하세요. (`cd chess_engine`)
- `CUDA out of memory`: `--batch-size`를 줄이고 `--device cpu`로 전환.
- 그래프가 비어있음: 학습이 완료되었는지, `graph/{id}/epoch-loss.txt` 파일이 생성되었는지 확인.

결과는 `model_result_monitoring/{model_id}/` 폴더에 저장:
- `csv_data/`: 에폭/배치 통계, 기울기 분석 CSV
- `plots/`: 다음 그래프들이 자동 생성
  - `epoch_loss_{id}.png`: Training + Validation 손실 곡선
  - `epoch_val_loss_{id}.png`: Validation 손실만 별도 표시
  - `batch_loss_{id}.png`: 선택 에폭들의 배치 손실 비교
  - `batch_loss_val_{id}.png`: Validation 에폭들만 필터링해 배치 손실 비교
  - `gradient_analysis_{id}.png`: Training 손실 기울기/2차 미분/절댓값 분석
  - `gradient_analysis_val_{id}.png`: Validation 손실 기울기 분석
  - `statistics_trends_{id}.png`: 배치 손실의 표준편차/분산/CV/범위 트렌드
  - `statistics_distributions_{id}.png`: 배치 손실 통계 분포
  - `statistics_distributions_val_{id}.png`: 에폭 단위 Validation 손실 분포/비교
- `experiment_report_{model_id}.md`: 종합 리포트

여러 실험 비교:
- `model_result_monitoring/consolidated_csv/`: 전체 실험 통합 CSV
- `model_result_monitoring/comparison_report.md`: 실험 성능 순위 및 비교

## 파인 튜닝
```bash
python -m ml.finetune \
  --resume checkpoints/best.pt \
  --epochs 1 --lr 1e-4 \
  --data-dir chessdata \
  --save-dir checkpoints_finetune
```

### 파인 튜닝/학습 옵션 추가
- `--mask-loser`: 승패가 난 게임에서 패배한 쪽의 수를 마스킹하여 학습에 반영하지 않음
- `--penalize-illegal`: 데이터셋에서 불법 수를 패널티 마스크로 표시하여 손실을 가중(페널티) 처리
- `--record`: 결과를 `experiments_record/{model_id}_{timestamp}/`로 타임스탬프 포함 영구 저장

두 옵션은 `train.py`와 `finetune.py` 모두 지원하며, 로깅 포맷은 상단의 CSV 포맷으로 통일되었습니다.

---

## 구성 요소

- `tokenizer.py`
  - `CharTokenizer`: 문자 단위 토크나이저
  - `SimpleMoveTokenizer`: PGN에서 SAN 유사 토큰을 휴리스틱으로 추출
- `dataset.py`
  - `PGNDataset`: `.pgn` → 토큰 시퀀스, pad/attn_mask 지원
  - 옵션: `mask_loser=True`, `penalize_illegal=True`로 학습 시 손실 마스킹/패널티 처리
- `model.py`
  - `TransformerDecoderLM`: decoder-only LM, `d_model/n_heads/n_layers/d_ff` 구성 가능, weight tying 옵션
- `train.py`
  - 학습/검증 루프, `--resume`로 로드, best/last 체크포인트 저장
  - 그래프 로그: 에폭/배치 CSV에 `timestamp` 컬럼 포함, 검증 손실 `val_loss` 기록
- `config.py`
  - `ModelConfig`, `TrainConfig` 데이터클래스
- `unified_model_monitor.py`
  - 디렉터리형 그래프 포맷(`graph/{id}/epoch-loss.txt`, `batch-loss.txt`) 자동 파싱
  - Train/Validation 손실 그래프, 배치 비교, 기울기 분석, 통계 트렌드/분포 생성
  - 최신/특정 실험 선택 실행, 통합 CSV 및 비교 리포트 생성

---

## TODO (축소 버전)
- [x] 2) Transformer 구현 (PyTorch): 토크나이저/데이터셋/모델/학습 스크립트
- [x] 3) 실험 모델 ID 생성/식별: 학습 시 자동 생성 및 체크포인트 저장, ID/경로로 불러오기 지원
- [x] 학습 로깅 타임스탬프/검증 손실 추가, 통합 모니터링 개선(Validation 그래프 포함)


