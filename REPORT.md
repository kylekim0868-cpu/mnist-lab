# REPORT - NumPy 신경망으로 손글씨 숫자 인식

코드를 길게 복사하지 말고, 구현할 때 판단한 내용과 검증 근거를 작성하세요.

## 0. 팀 정보

| 반 | 팀명 | 팀원 | 구현·리뷰 방식 |
| --- | --- | --- | --- |
|  |  | 김영민(ym), 백찬(bc), 김민희(mh) | 각자 `이니셜/dev`에서 Day별 작업 브랜치를 따서 구현하고, `pytest`가 통과하면 PR로 셀프 merge한다. `이니셜/dev` → `main`은 팀원 1명 이상이 승인해야 merge한다 |

## 1. 전체 학습 흐름

실제 실행한 batch의 shape를 채우세요.

MNIST baseline 기준이다. B = 128(마지막 배치만 60000 % 128 = 96), H = 128.

```text
x_batch                       (128, 784)
-> Affine 1                   (128, 128)
-> ReLU                       (128, 128)
-> Affine 2 / logits          (128, 10)
-> SoftmaxWithLoss            scalar

loss
-> dlogits                    (128, 10)
-> Affine 2 backward          dW2 (128, 10) / db2 (10,) / dx (128, 128)
-> ReLU backward              (128, 128)
-> Affine 1 backward          dW1 (784, 128) / db1 (128,) / dx (128, 784)
-> optimizer update
```

`Forward -> Loss -> Backward -> Update`가 이 순서여야 하는 이유를 설명하세요.

- Loss는 forward가 만든 logits가 있어야 계산할 수 있다.
- Backward는 forward 때 저장해 둔 값으로 국소 미분을 계산한다(Affine의 `x`, ReLU의 `mask`, Softmax의 `y_prob`). 출발점인 첫 dout(∂L/∂logits)은 loss 층의 backward가 만든다. 그래서 forward와 loss가 먼저 돌아야 한다.
- Update는 backward가 채운 `model.grads`를 쓴다. backward보다 먼저 update하면 이전 배치의 gradient(맨 처음에는 0)로 갱신하게 된다.

## 2. 핵심 구현

### 2.1 Affine

- `x @ W + b`에서 각 배열의 shape: `x (B, D) @ W (D, H) → (B, H)`. `b (H,)`는 `(1, H)`로 broadcast되어 모든 행에 같은 값이 더해진다. 1층은 D = 784, H = 128이고 2층은 D = 128, H = 10이다.
- `dW = x.T @ dout`이 되는 이유: 스칼라로 보면 y = x·w + b에서 ∂L/∂w = dout·x다. W는 배치의 모든 샘플이 같이 쓰므로 샘플마다 dout·x를 구해서 더한다. dW[i, j] = Σₙ x[n, i]·dout[n, j]는 x의 i번 열과 dout의 j번 열의 내적이다. 행렬곱은 "앞 행렬의 행 · 뒤 행렬의 열"이라서 x를 뒤집어 `x.T @ dout`이 된다. shape는 (D, B) @ (B, H) = (D, H)로 W와 같다.
- `db`를 batch 방향으로 합산하는 이유: b도 B개 샘플 모두에 똑같이 더해졌으니 L로 가는 경로가 B개다. 국소 미분이 1이라 dout을 샘플 방향으로 더하면 된다. 평균이 아닌 이유는 1/B를 Softmax backward에서 이미 나눴기 때문이다.
- forward 입력을 저장해야 하는 이유: dW를 구하려면 x가 필요한데, backward는 dout만 받는다. 그래서 forward 때 `self.x`에 저장해 둔다.

### 2.2 ReLU

- forward에서 저장한 mask: 입력 x에서 x ≤ 0인 위치를 담은 bool 배열이다. x와 shape가 같다.
- backward에서 gradient가 0이 되는 위치와 이유: forward 때 x ≤ 0이었던 위치다. ReLU의 국소 미분이 x > 0이면 1, x ≤ 0이면 0이라서 dout에 0이나 1을 곱하는 것과 같다. 그래서 곱셈 대신 mask 위치를 0으로 덮어쓴다. 기준은 dout의 부호가 아니라 forward 때 x의 부호다. x = 2였던 자리의 dout −7은 그대로 통과한다.

### 2.3 SoftmaxWithLoss

- logits와 확률의 차이: logits는 마지막 Affine의 출력이라 범위 제한이 없고 합도 1이 아니다. softmax(exp를 취하고 행 합으로 나눔)를 거치면 각 값이 0~1이고 행 합이 1인 확률이 된다.
- max-shift를 적용한 이유: e^(z−c) / Σ e^(z_j−c) = e^z / Σ e^(z_j)라서 c를 빼도 확률은 같다. 행 최댓값을 빼면 exp의 입력이 0 이하가 돼서 exp 값이 (0, 1] 안에 들어오고, 분모는 1 이상이 된다. 그래서 e^709.78을 넘는 overflow가 생기지 않는다. 행마다 따로 빼야 하므로 `keepdims=True`로 (N, 1)을 유지한다.
- 모든 logits가 같을 때 loss가 `ln(10)`인 이유: logits가 모두 같으면 10개 클래스의 확률이 모두 1/10이다. 정답 확률도 1/10이니 loss = −ln(1/10) = ln 10 = 2.3026이다.
- gradient가 `(확률 - 정답 one-hot) / batch_size`가 되는 의미: L = −z_y + log Σ e^(z_j)를 z_k로 미분하면 p_k − t_k다. 정답 칸은 p − 1 < 0이라 그 logit을 올리라는 뜻이고, 나머지 칸은 p > 0이라 내리라는 뜻이다. 틀린 정도만큼 미는 셈이다. forward에서 batch 평균을 냈으니 B로 나눈다. 각 행의 합은 1 − 1 = 0이다.

### 2.4 SGD와 학습 루프

- gradient 반대 방향으로 이동하는 이유: gradient는 "파라미터가 조금 커지면 L이 얼마나 커지는가"다. dW[0, 0] = 7이면 W[0, 0]을 키울수록 L이 커지므로 줄여야 한다. 그래서 param − lr · grad이고, lr은 한 번에 얼마나 움직일지를 정한다.
- 파라미터를 제자리에서 갱신해야 하는 이유: 레이어를 `Affine(params["W1"], ...)`로 만들어서 `params["W1"]`과 `affine1.W`가 같은 배열을 가리킨다. `params[key] = params[key] - ...`는 새 배열을 만들어 dict의 이름표만 옮긴다. 그러면 `affine1.W`는 예전 배열을 계속 보고, forward 결과가 바뀌지 않아 학습이 안 된다. `-=`는 배열 내용 자체를 바꾸므로 둘 다 바뀐다.
- 한 미니배치에서 실행한 네 단계:
  1. model forward(학습 모드)로 logits를 만든다.
  2. criterion forward로 logits와 정답을 비교해 loss를 구한다.
  3. criterion backward가 돌려준 ∂L/∂logits를 model backward에 dout으로 넘긴다. 이 과정에서 `model.grads`가 채워진다.
  4. optimizer가 `model.params`를 `model.grads`로 제자리 갱신한다.

## 3. 검증 결과

| 검증 | 실행 조건 | 결과 | 무엇을 확인했는가 |
| --- | --- | --- | --- |
| 작은 배열 단위 테스트 | 각 함수 | Affine 8, ReLU 5, Softmax 11, SGD 4 통과 | 값과 shape |
| 초기 loss | `ln(10)`과 비교 | logits가 모두 0이면 2.3026. 초기 모델(seed 0)로 MNIST 128개를 넣으면 2.4359, smoke lr = 0의 첫 epoch는 2.3819 | 출력층 forward |
| gradient check | 최대 상대 오차 | hidden 16, MNIST 8개, 파라미터당 6곳: 0.00e+00(차이가 atol 1e-8 이하라 0으로 기록됨). atol = 0으로 다시 재면 4.82e-08(Day 2에 직접 비교했을 때는 1e-11 수준). 허용 1e-6 | backward 수학적 일치 |
| 학습 Smoke | 인공 데이터 | Adam lr 0.001, 5 epoch: 1.650 -> 0.007. 파라미터 4개 모두 바뀜. SGD lr 0.1은 1.638 -> 0.076 | 네 단계 연결과 parameter 변화 |
| 전체 테스트 | `pytest tests/ -v` | 36 passed | 전체 불변조건 |

인공 데이터 정확도 검사는 학습 연결을 확인하는 테스트일 뿐 MNIST 성능 목표가 아닙니다.

## 4. MNIST 실행 관찰

| 항목 | 값 |
| --- | --- |
| hidden size | 128 |
| optimizer / learning rate | SGD / 0.1 |
| epochs / batch size | 10 / 128 |
| 파라미터 수 | 101,770 (784·128 + 128 + 128·10 + 10) |
| 첫 epoch loss | 0.4561 |
| 마지막 epoch loss | 0.0903 |
| 관찰한 정확도 | test 97.05% (학습 전 9.87%, train 97.72%) |
| 실행 시간 | 약 2.0초 (10 epoch) |

정확도는 실행 상태를 관찰하기 위한 값이며 제출 통과 기준이 아닙니다. 학습 전후 예측이 어떻게 달라졌는지 한두 문장으로 적으세요.

학습 전 모델은 test 10000개 중 7587개를 2로 예측했다. 거의 한 클래스만 찍은 셈이라 정확도가 9.87%다. 학습 후에는 클래스별 예측 수가 908~1136개로 고르게 퍼졌고, test 앞 10개 중 9개를 맞혔다(5를 6으로 틀림).

- 첫 epoch loss가 ln 10 ≈ 2.30이 아니라 0.4561인 이유: epoch loss는 그 epoch의 469개 배치 loss의 평균이고, 첫 epoch 동안에도 update가 계속 돈다.
- loss 감소가 일반화를 보장하지 않는 이유: 학습 loss는 학습에 쓴 데이터로만 잰 값이다. 처음 보는 데이터에서도 잘 맞히는지는 test로 따로 확인해야 한다. 이번에는 train 97.72%와 test 97.05%의 차이가 작았다.
- seed(초기 W, 배치 섞는 순서)를 고정해서 같은 설정으로 다시 돌리면 같은 값이 나온다.

## 5. 오류와 해결 과정

| 증상 | 처음 세운 가설 | 확인한 최소 입력 | 실제 원인 | 수정 후 검증 |
| --- | --- | --- | --- | --- |
| pytest에서 `ModuleNotFoundError: numpy` | 코드가 틀렸다 | 테스트 파일 하나 | conda 환경이 아니라 시스템 Python 3.14로 실행됐다 | `conda activate mnist-nn` 후 `python -m pytest` |
| Softmax forward에서 `(4, 10)`과 `(4,)`의 broadcast 에러. `(3, 3)`에서는 행 합이 1이 아님 | `axis=1`로 행별 max와 sum을 구하면 바로 빼고 나눌 수 있다 | `(3, 3)` logits | 결과가 `(N,)`이 돼서 뒤쪽 축(C)에 맞춰지는 바람에 열 방향으로 빠졌다 | `keepdims=True`로 `(N, 1)` 유지. 행 합이 1 |
| loss가 0.1(정답 확률의 평균)로 나옴 | 정답 확률을 평균 내면 loss다 | 정답 확률 `[0.7, 0.8, 0.3]` | clip과 평균 사이의 `-log`가 빠졌다 | logits가 모두 0일 때 2.3026 |
| Softmax backward에서 `(4, 10)`과 `(4,)`의 broadcast 에러(N == C면 조용히 틀림), 이어서 `y_prob[y_true]`의 IndexError | 라벨 배열을 빼면 one-hot을 빼는 것과 같다 | logits `zeros((2, 3))`, 정답 `[0, 1]` | `y_true`는 one-hot `(N, C)`가 아니라 라벨 번호 `(N,)`이다. 인덱스를 하나만 주면 행을 고른다 | `[np.arange(N), y_true]` 칸에서 1을 뺌. 손계산 `[[1/3 − 1, 1/3, 1/3], [1/3, 1/3 − 1, 1/3]] / 2`와 일치 |
| params 키 개수를 모르는데 전부 갱신해야 함 | 키를 하나하나 적어야 한다 | 키가 `W1`, `b1`, `W2`인 dict (`test_update_all_keys`) | dict를 for문으로 돌면 키가 다 나온다. params와 grads는 키가 같다 | key 하나로 두 dict를 같이 꺼냄. `test_sgd.py` 4개 통과 |
| `TypeError: TwoLayerNet.backward() missing ... 'dout'` | criterion backward와 model backward를 차례로 부르면 값이 알아서 넘어간다 | smoke 테스트 | `criterion.backward()`의 반환값(∂L/∂logits)을 받지 않고 버렸다. `optimizer.update()`에도 인자를 안 줬다 | 반환값을 model backward에 넘기고, update에는 `model.params`, `model.grads`를 넘김. smoke 6개 통과 |

## 6. 참고와 AI 사용 기록

| 출처 | 질문·참고 내용 | 내가 먼저 시도한 것 | 직접 검증한 방법 |
| --- | --- | --- | --- |
| AI (Claude Code) | Day 1: Affine 식의 원소 단위 유도, `y_prob[np.arange(N), y_true]` 인덱싱, `-log`를 쓰는 이유 | 한 행만 계산, `-log` 없이 정답 확률 평균 | 작은 배열로 직접 출력 비교, Day 1 테스트 13개, `selfcheck --day 1 --trace` |
| AI (Claude Code) | Day 2: Softmax + Cross Entropy 미분 유도, 스칼라 chain rule, 연쇄 대입, `.T` 사용법, gradient check의 `atol` 의미 | 라벨 배열을 그대로 빼기, dW를 `.T` 없이 작성 | 손계산 dW[0, 0] = 7, dx[0, 0] = −2, Softmax 11개·ReLU 5개·Affine 8개 통과, `selfcheck --day 2` 26 passed |
| AI (Claude Code) | Day 3: chain rule(경로 곱과 경로 합) | 국소 미분을 다 구해서 곱하기만 하면 된다고 생각 | `test_sgd.py` 4개, `test_training_smoke.py` 6개 통과, `selfcheck --all` 36 passed, 같은 seed로 baseline을 두 번 돌려 같은 값 |
| 과제 docstring·테스트 | Softmax forward 순서 1~4와 clip을 쓰는 이유, backward 순서 1~3, `test_affine.py`의 손계산 숫자, gradient check 실패 메시지의 범위 힌트, SGD 제자리 갱신 경고, train TODO 1~4 힌트, smoke 테스트의 lr = 0 초기 loss 검사 | - | 같은 숫자로 손계산한 값과 테스트 기대값 비교 |

## 7. 회고

- 이제 코드와 shape으로 설명할 수 있는 개념: 각 층의 forward와 backward shape, chain rule(경로마다 곱하고 경로끼리 더하기), Softmax + CE의 gradient가 p − t가 되는 이유, 제자리 갱신이 필요한 이유
- 아직 설명하기 어려운 개념: 행렬 chain rule을 처음부터 행렬로 떠올리기. 아직은 스칼라 하나로 쪼갠 다음 shape에 맞춰 `.T`를 붙여야 한다
- 다시 구현한다면 먼저 확인할 것: 중간 배열의 shape를 찍어 `(N,)`인지 `(N, 1)`인지 보기, 대입이 새 배열을 만드는지 제자리를 바꾸는지 보기, 함수 반환값을 받았는지 보기
