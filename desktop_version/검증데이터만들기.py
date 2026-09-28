"""웹 버전 검증용 정답 데이터(../web_version/검증데이터.json)를 만든다.

MNIST 시험 이미지 200장을 앱 그림판과 같은 280x280 그림으로 키우고,
그 그림에 대한 파이썬 전처리 결과, 예측 확률 10개, 정답 라벨을 함께 담는다.
검증.html 이 같은 그림을 자바스크립트 경로로 통과시켜 이 값들과 대조한다.

그림 200장 중
- 170장: 28x28을 최근접 이웃으로 정확히 10배 확대 (MNIST와 가장 가까운 경우)
- 30장 : 임의 배율(어중간한 크기), 쌍선형 확대(부드러운 가장자리), 임의 위치(가장자리 접촉 포함)
  → 10배 확대만 쓰면 숫자 크기가 늘 10의 배수로 떨어져 손그림에서 생기는 경우를 시험하지 못한다.

실행: desktop_version 폴더 안에서  py 검증데이터만들기.py  (결과 파일은 깃에 넣지 않음)
"""

import base64
import io
import json
import os
import random

import numpy as np
import torch
from PIL import Image
from torchvision import datasets

from model import 숫자인식망
from preprocess import 전처리, 정규화

폴더 = os.path.dirname(os.path.abspath(__file__))
결과_경로 = os.path.join(폴더, "..", "web_version", "검증데이터.json")
표본_수 = 200
임의배율_수 = 30
그림판_크기 = 280


def 그림_만들기(작은그림: Image.Image, 번호: int, 난수: random.Random):
    """28x28 MNIST 그림을 280x280 그림판 크기로 키운다."""
    if 번호 < 표본_수 - 임의배율_수:
        return 작은그림.resize((그림판_크기, 그림판_크기), Image.NEAREST), "10배"
    크기 = 난수.randint(90, 그림판_크기)
    큰 = 작은그림.resize((크기, 크기), Image.BILINEAR)
    바탕 = Image.new("L", (그림판_크기, 그림판_크기), 0)
    남는칸 = 그림판_크기 - 크기
    위치 = (난수.choice([0, 남는칸, 난수.randint(0, 남는칸)]), 난수.randint(0, 남는칸))
    바탕.paste(큰, 위치)
    return 바탕, f"임의배율 {크기}"


def png_주소(그림: Image.Image):
    버퍼 = io.BytesIO()
    그림.save(버퍼, format="PNG")
    return "data:image/png;base64," + base64.b64encode(버퍼.getvalue()).decode("ascii")


def main():
    모델 = 숫자인식망()
    모델.load_state_dict(torch.load(os.path.join(폴더, "mnist_cnn.pt"), map_location="cpu"))
    모델.eval()
    시험셋 = datasets.MNIST(os.path.join(폴더, "data"), train=False, download=True)
    난수 = random.Random(0)

    표본들, 정답수 = [], 0
    for 번호 in range(표본_수):
        작은그림, 라벨 = 시험셋[번호]
        그림, 종류 = 그림_만들기(작은그림, 번호, 난수)
        배열28 = 전처리(그림)
        입력 = torch.from_numpy(정규화(배열28)).float().view(1, 1, 28, 28)
        with torch.no_grad():
            확률 = torch.softmax(모델(입력), dim=1)[0].numpy()
        정답수 += int(확률.argmax() == 라벨)
        표본들.append({
            "번호": 번호,
            "종류": 종류,
            "라벨": int(라벨),
            "그림": png_주소(그림),
            "전처리": np.round(배열28, 6).ravel().tolist(),
            "확률": [float(p) for p in 확률],
        })

    with open(결과_경로, "w", encoding="utf-8") as 파일:
        json.dump({"표본": 표본들}, 파일, ensure_ascii=False)
    print(f"표본 {표본_수}장, 파이썬 정확도 {정답수 / 표본_수 * 100:.1f}% ({정답수}/{표본_수})")


if __name__ == "__main__":
    main()
