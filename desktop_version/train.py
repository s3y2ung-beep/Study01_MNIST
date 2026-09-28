"""MNIST 손글씨 숫자 데이터로 숫자인식망을 학습하고 가중치를 mnist_cnn.pt로 저장한다.

실행: 이 파일이 있는 폴더에서  py train.py  (맥: python3 train.py)
데이터는 상대 경로 data/ 에 내려받으므로 반드시 이 폴더 안에서 실행해야 한다.
"""

import ssl
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import 숫자인식망
from preprocess import 정규화_평균, 정규화_표준편차

데이터_폴더 = "data"
가중치_파일 = "mnist_cnn.pt"
배치_크기 = 64
에폭_수 = 6
학습률 = 0.001
학습률_감소 = 0.7


class 획굵기_바꾸기:
    """손그림은 사람과 펜에 따라 획 굵기가 제각각이므로, 학습 그림의 획을 무작위로 굵게 또는 가늘게 만든다.

    굵게: 3x3 최대 필터(팽창), 가늘게: 3x3 최소 필터(침식). 각각 1/3 확률, 나머지 1/3은 그대로 둔다.
    """

    def __call__(self, 그림):  # 그림: 1x28x28 텐서 (0~1)
        선택 = torch.rand(1).item()
        if 선택 < 1 / 3:
            return F.max_pool2d(그림.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        if 선택 < 2 / 3:
            return -F.max_pool2d(-그림.unsqueeze(0), 3, stride=1, padding=1).squeeze(0)
        return 그림


def 데이터_불러오기(학습용: bool, 변환):
    """MNIST를 내려받아 불러온다. 학교망 등에서 SSL 인증서 오류가 나면 검증을 끄고 한 번 더 시도한다."""
    try:
        return datasets.MNIST(데이터_폴더, train=학습용, download=True, transform=변환)
    except Exception as 오류:
        if "CERTIFICATE_VERIFY_FAILED" not in str(오류):
            raise
        print("SSL 인증서 오류가 나서 인증서 검증 없이 다시 내려받습니다.")
        ssl._create_default_https_context = ssl._create_unverified_context
        return datasets.MNIST(데이터_폴더, train=학습용, download=True, transform=변환)


def 평가(모델, 불러오개):
    """시험 데이터에 대한 평균 손실과 정확도를 계산한다."""
    모델.eval()
    손실합, 정답수 = 0.0, 0
    with torch.no_grad():
        for 이미지, 라벨 in 불러오개:
            점수 = 모델(이미지)
            손실합 += F.cross_entropy(점수, 라벨, reduction="sum").item()
            정답수 += (점수.argmax(1) == 라벨).sum().item()
    개수 = len(불러오개.dataset)
    return 손실합 / 개수, 정답수 / 개수


def main():
    torch.manual_seed(0)

    # 학습 때만 회전, 이동, 확대/축소, 기울임, 획 굵기 증강을 넣어 손그림의 다양한 모양에 강하게 만든다
    학습_변환 = transforms.Compose([
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.85, 1.1), shear=15),
        transforms.ToTensor(),
        획굵기_바꾸기(),
        transforms.Normalize((정규화_평균,), (정규화_표준편차,)),
    ])
    시험_변환 = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((정규화_평균,), (정규화_표준편차,)),
    ])

    학습_불러오개 = DataLoader(데이터_불러오기(True, 학습_변환), batch_size=배치_크기, shuffle=True)
    시험_불러오개 = DataLoader(데이터_불러오기(False, 시험_변환), batch_size=1000)

    모델 = 숫자인식망()
    최적화기 = torch.optim.Adam(모델.parameters(), lr=학습률)
    스케줄러 = torch.optim.lr_scheduler.StepLR(최적화기, step_size=1, gamma=학습률_감소)

    for 에폭 in range(1, 에폭_수 + 1):
        모델.train()
        시작 = time.time()
        손실합 = 0.0
        for 번호, (이미지, 라벨) in enumerate(학습_불러오개, 1):
            최적화기.zero_grad()
            손실 = F.cross_entropy(모델(이미지), 라벨)
            손실.backward()
            최적화기.step()
            손실합 += 손실.item()
            if 번호 % 100 == 0:
                print(f"  에폭 {에폭} 배치 {번호}/{len(학습_불러오개)} 손실 {손실.item():.4f}", flush=True)
        현재_학습률 = 스케줄러.get_last_lr()[0]
        스케줄러.step()
        시험_손실, 정확도 = 평가(모델, 시험_불러오개)
        print(f"에폭 {에폭}: 학습률 {현재_학습률:.5f}, 학습 평균 손실 {손실합 / len(학습_불러오개):.4f}, "
              f"시험 손실 {시험_손실:.4f}, 시험 정확도 {정확도 * 100:.2f}%, "
              f"소요 {time.time() - 시작:.0f}초", flush=True)

    torch.save(모델.state_dict(), 가중치_파일)
    print(f"가중치를 {가중치_파일} 에 저장했습니다.")


if __name__ == "__main__":
    main()
