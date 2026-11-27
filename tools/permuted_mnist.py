# -*- coding: utf-8 -*-
"""
추가 실험: 픽셀 순서를 섞어도 2층 MLP가 똑같이 배우는지 확인한다.

baseline 설정(hidden 128, SGD lr 0.1, batch 128, 10 epoch, seed 0)을 네 가지 입력으로 돌리고
docs/에 baseline loss 곡선과 원본/섞은 이미지 그림을 저장한다.

실행:
    python tools/permuted_mnist.py
"""

import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from data import load_mnist
from losses import SoftmaxWithLoss
from network import TwoLayerNet
from optimizers import SGD
from training import evaluate, plot_loss_history, train

warnings.filterwarnings("ignore", message="FigureCanvasAgg is non-interactive")
DOCS = ROOT / "docs"

(x_train, y_train), (x_test, y_test) = load_mnist(str(ROOT / "data"))
perm = np.random.default_rng(42).permutation(784)


def run(name, xtr, xte, permute_w1=False):
    model = TwoLayerNet(hidden_size=128, seed=0)
    if permute_w1:
        # 레이어와 같은 배열이라 [:]로 내용만 바꾼다
        model.params["W1"][:] = model.params["W1"][perm]
    history = train(model, SoftmaxWithLoss(), SGD(lr=0.1), xtr, y_train,
                    epochs=10, batch_size=128, verbose=False)
    accuracy, _ = evaluate(model, xte, y_test)
    print(f"| {name} | {history[0]:.6f} -> {history[-1]:.6f} | {accuracy:.2f}% |")
    return history


print("| 입력 | loss | test 정확도 |")
print("| --- | --- | --- |")
history = run("원본", x_train, x_test)
run("픽셀 섞음", x_train[:, perm], x_test[:, perm])
run("픽셀 섞음 + W1 행도 같은 순서로 섞음", x_train[:, perm], x_test[:, perm], permute_w1=True)

# 위치마다 정해진 값 p를 모든 샘플에 똑같이 더한다 (positional embedding 흉내)
p = (0.1 * np.sin(np.arange(784) / 784 * 2 * np.pi * 8)).astype(x_train.dtype)
run("위치마다 정해진 값 p를 더함 (x + p)", x_train + p, x_test + p)

DOCS.mkdir(exist_ok=True)
plt.figure(figsize=(5, 3.2))
plot_loss_history(history, label="baseline")
plt.savefig(DOCS / "baseline_loss.png", dpi=120, bbox_inches="tight")

fig, axes = plt.subplots(1, 2, figsize=(5, 2.6))
for ax, image, title in [(axes[0], x_test[0], f"original (label {y_test[0]})"),
                         (axes[1], x_test[0][perm], "pixels permuted")]:
    ax.imshow(image.reshape(28, 28), cmap="gray")
    ax.set_title(title, fontsize=9)
    ax.axis("off")
fig.tight_layout()
fig.savefig(DOCS / "permuted_digit.png", dpi=120)
print(f"\n그림 저장: {DOCS / 'baseline_loss.png'}, {DOCS / 'permuted_digit.png'}")
