"""손글씨 숫자 인식용 합성곱 신경망(CNN) 정의."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class 숫자인식망(nn.Module):
    """28x28 흑백 이미지를 받아 숫자 0~9에 대한 점수 10개를 내는 신경망.

    구조: 합성곱 2개(각각 뒤에 최대 풀링) → 전결합 2개.
    학습 때만 드롭아웃을 쓰고, 예측 때는 드롭아웃이 아무 일도 하지 않는다.
    """

    def __init__(self):
        super().__init__()
        # 합성곱층 1: 흑백 1채널 → 특징 32개, 3x3 필터, 가장자리 1칸 덧대기로 크기 유지
        self.합성곱1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        # 합성곱층 2: 특징 32개 → 64개
        self.합성곱2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 두 번의 2x2 풀링으로 28 → 14 → 7 이 되므로 64 x 7 x 7 개의 값이 나온다
        self.전결합1 = nn.Linear(64 * 7 * 7, 128)
        self.전결합2 = nn.Linear(128, 10)
        self.드롭아웃 = nn.Dropout(0.3)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.합성곱1(x)), 2)   # 32 x 14 x 14
        x = F.max_pool2d(F.relu(self.합성곱2(x)), 2)   # 64 x 7 x 7
        x = torch.flatten(x, 1)                         # 3136
        x = self.드롭아웃(F.relu(self.전결합1(x)))       # 128
        return self.전결합2(x)                           # 10 (소프트맥스 전 점수)


if __name__ == "__main__":
    모델 = 숫자인식망()
    print(모델)
    print("파라미터 수:", sum(p.numel() for p in 모델.parameters()))
