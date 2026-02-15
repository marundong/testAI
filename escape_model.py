"""逃生行为融合与校准模型。

根据给定公式：
1) 乾坤融合计算: M = aD + BK, 且 a + B = 1
2) 偏差计算: E = |M - Z|
3) 中和校准计算: G = (1 - γ)M + γZ, 其中 γ ∈ [0, 1]

变量范围：D, K, Z, M, E, G ∈ [0, 1]
"""

from __future__ import annotations

from dataclasses import dataclass


class ValueRangeError(ValueError):
    """输入值不在 [0, 1] 区间时抛出的异常。"""


@dataclass(frozen=True)
class ModelResult:
    """模型输出结果。"""

    D: float
    K: float
    Z: float
    a: float
    B: float
    M: float
    E: float
    gamma: float
    G: float


class EscapeBehaviorModel:
    """AR 逃生行为融合与校准模型。"""

    def __init__(self, a: float = 0.7, B: float = 0.3) -> None:
        self._validate_unit_interval(a, "a")
        self._validate_unit_interval(B, "B")
        if abs((a + B) - 1.0) > 1e-9:
            raise ValueError(f"约束不满足: a + B 必须等于 1，当前为 {a + B:.6f}")

        self.a = a
        self.B = B

    @staticmethod
    def _validate_unit_interval(value: float, name: str) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueRangeError(f"{name} 必须在 [0, 1] 区间，当前值: {value}")

    def fuse(self, D: float, K: float) -> float:
        """第一步：乾坤融合计算 M = aD + BK。"""
        self._validate_unit_interval(D, "D")
        self._validate_unit_interval(K, "K")
        M = self.a * D + self.B * K
        # 理论上 M 已在 [0,1]，这里做数值稳定性保护
        return max(0.0, min(1.0, M))

    @staticmethod
    def deviation(M: float, Z: float) -> float:
        """偏差计算 E = |M - Z|。"""
        EscapeBehaviorModel._validate_unit_interval(M, "M")
        EscapeBehaviorModel._validate_unit_interval(Z, "Z")
        return abs(M - Z)

    @staticmethod
    def dynamic_gamma(E: float, input_gamma: float | None = None) -> float:
        """根据 E 趋近于 0 的程度动态计算 gamma，并输出 gamma。

        设计策略：
        - 趋近因子 close_to_zero = 1 - E，E 越接近 0，close_to_zero 越接近 1。
        - 若未输入 gamma: 直接使用 close_to_zero，即 gamma = 1 - E。
        - 若输入 gamma: 在输入值基础上向 close_to_zero 融合，避免完全忽略人工输入。
          gamma = 0.5 * input_gamma + 0.5 * close_to_zero
        """
        EscapeBehaviorModel._validate_unit_interval(E, "E")
        close_to_zero = 1.0 - E

        if input_gamma is None:
            gamma = close_to_zero
        else:
            EscapeBehaviorModel._validate_unit_interval(input_gamma, "input_gamma")
            gamma = 0.5 * input_gamma + 0.5 * close_to_zero

        return max(0.0, min(1.0, gamma))

    @staticmethod
    def calibrate(M: float, Z: float, gamma: float) -> float:
        """第二步：中和校准计算 G = (1 - gamma)M + gamma Z。"""
        EscapeBehaviorModel._validate_unit_interval(M, "M")
        EscapeBehaviorModel._validate_unit_interval(Z, "Z")
        EscapeBehaviorModel._validate_unit_interval(gamma, "gamma")
        G = (1.0 - gamma) * M + gamma * Z
        return max(0.0, min(1.0, G))

    def run(
        self, D: float, K: float, Z: float, gamma: float | None = None
    ) -> ModelResult:
        """执行完整流程。

        参数:
            D: 乾特征（行为过激指数）
            K: 坤特征（行为迟缓指数）
            Z: 中特征（标准逃生指数）
            gamma: 输入 gamma 值（可选）。
                - 若不输入，则由 E 自动动态计算。
                - 若输入，则与动态 gamma 融合后用于校准。

        返回:
            ModelResult: 包含 M、E、gamma、G 等全部中间与输出值。
        """
        self._validate_unit_interval(Z, "Z")

        M = self.fuse(D, K)
        E = self.deviation(M, Z)

        used_gamma = self.dynamic_gamma(E, gamma)
        G = self.calibrate(M, Z, used_gamma)

        return ModelResult(
            D=D,
            K=K,
            Z=Z,
            a=self.a,
            B=self.B,
            M=M,
            E=E,
            gamma=used_gamma,
            G=G,
        )


if __name__ == "__main__":
    # 示例：推荐系数 a=0.7, B=0.3
    model = EscapeBehaviorModel(a=0.7, B=0.3)

    # 不输入 gamma：完全动态计算 gamma
    result_auto = model.run(D=0.9, K=0.2, Z=0.8)
    print("自动 gamma:", result_auto)

    # 输入 gamma：在输入值基础上动态调整后输出 gamma
    result_with_input = model.run(D=0.9, K=0.2, Z=0.8, gamma=0.4)
    print("输入并动态调整 gamma:", result_with_input)
