"""mnist_cnn.pt 의 가중치를 웹 버전이 읽을 수 있는 파일 2개로 내보낸다.

- ../web_version/가중치.bin      : 텐서 8개(층 4개의 weight, bias)를 float32 리틀 엔디언으로 이어 붙인 것
- ../web_version/가중치정보.json : 텐서별 이름, 형상, 시작 위치(원소 단위)와 정규화 상수

실행: desktop_version 폴더 안에서  py 가중치내보내기.py
"""

import json
import os

import numpy as np
import torch

from model import 숫자인식망
from preprocess import 정규화_평균, 정규화_표준편차

폴더 = os.path.dirname(os.path.abspath(__file__))
웹_폴더 = os.path.join(폴더, "..", "web_version")


def main():
    모델 = 숫자인식망()
    모델.load_state_dict(torch.load(os.path.join(폴더, "mnist_cnn.pt"), map_location="cpu"))
    상태 = 모델.state_dict()

    조각들, 텐서정보 = [], []
    위치 = 0
    for 이름, 텐서 in 상태.items():
        값 = 텐서.detach().cpu().numpy().astype("<f4").ravel()
        텐서정보.append({"이름": 이름, "형상": list(텐서.shape), "시작": 위치, "개수": int(값.size)})
        조각들.append(값)
        위치 += 값.size

    os.makedirs(웹_폴더, exist_ok=True)
    전체 = np.concatenate(조각들)
    with open(os.path.join(웹_폴더, "가중치.bin"), "wb") as 파일:
        파일.write(전체.tobytes())

    정보 = {
        "형식": "float32 리틀 엔디언, 텐서를 아래 순서대로 이어 붙임",
        "파라미터수": int(위치),
        "정규화": {"평균": 정규화_평균, "표준편차": 정규화_표준편차},
        "텐서": 텐서정보,
    }
    with open(os.path.join(웹_폴더, "가중치정보.json"), "w", encoding="utf-8") as 파일:
        json.dump(정보, 파일, ensure_ascii=False, indent=2)

    # 검산: 파일을 다시 읽어 원본과 비교
    다시 = np.fromfile(os.path.join(웹_폴더, "가중치.bin"), dtype="<f4")
    assert 다시.size == 위치 and np.array_equal(다시, 전체)
    print(f"텐서 {len(텐서정보)}개, 파라미터 {위치:,}개, {다시.nbytes:,}바이트를 내보냈습니다.")


if __name__ == "__main__":
    main()
