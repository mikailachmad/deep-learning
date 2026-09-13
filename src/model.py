from __future__ import annotations
import math
from dataclasses import dataclass, field

@dataclass
class SLPWeights:
    bias: float = 0.5
    teta1: float = 0.5
    teta2: float = 0.5
    teta3: float = 0.5
    teta4: float = 0.5

    def copy(self) -> "SLPWeights":
        return SLPWeights(self.bias, self.teta1, self.teta2, self.teta3, self.teta4)

@dataclass
class StepResult:
    z: float
    g: float
    pred: int
    error: float
    sse: float
    correct: bool
    dbias: float
    dteta1: float
    dteta2: float
    dteta3: float
    dteta4: float

def sigmoid(z: float) -> float:
    return 1.0 / (1.0 + math.exp(-z))

def forward(w: SLPWeights, x1: float, x2: float, x3: float, x4: float, target: float) -> StepResult:
    z = w.bias + w.teta1 * x1 + w.teta2 * x2 + w.teta3 * x3 + w.teta4 * x4
    g = sigmoid(z)
    pred = 1 if g > 0.5 else 0
    error = g - target
    sse = error ** 2
    correct = pred == int(target)

    dbias = 2 * error * (1 - g) * g
    dteta1 = dbias * x1
    dteta2 = dbias * x2
    dteta3 = dbias * x3
    dteta4 = dbias * x4

    return StepResult(z, g, pred, error, sse, correct, dbias, dteta1, dteta2, dteta3, dteta4)

def update_weights(w: SLPWeights, step: StepResult, lr: float) -> SLPWeights:
    return SLPWeights(
        bias=w.bias - lr * step.dbias,
        teta1=w.teta1 - lr * step.dteta1,
        teta2=w.teta2 - lr * step.dteta2,
        teta3=w.teta3 - lr * step.dteta3,
        teta4=w.teta4 - lr * step.dteta4,
    )