"""손글씨 숫자 인식 데스크톱 앱 (tkinter).

마우스로 숫자를 그리고 손을 떼면 바로 인식하여 상위 3개 후보를 확률과 함께 보여 준다.
실행: 이 파일이 있는 폴더에서  py app.py  (맥: python3 app.py)
"""

import os
import tkinter as tk

import torch
from PIL import Image, ImageDraw

from model import 숫자인식망
from preprocess import 전처리, 정규화

그림판_크기 = 280
펜_굵기 = 18
폴더 = os.path.dirname(os.path.abspath(__file__))
가중치_경로 = os.path.join(폴더, "mnist_cnn.pt")


class 손글씨앱:
    def __init__(self, 창):
        self.창 = 창
        창.title("손글씨 숫자 인식")
        창.resizable(False, False)

        # 모델 불러오기 (예측 모드: 드롭아웃 끔)
        self.모델 = 숫자인식망()
        self.모델.load_state_dict(torch.load(가중치_경로, map_location="cpu"))
        self.모델.eval()

        # 화면에 보이는 그림판 (흰 바탕에 검은 글씨)
        self.그림판 = tk.Canvas(창, width=그림판_크기, height=그림판_크기, bg="white",
                              highlightthickness=1, highlightbackground="#888")
        self.그림판.grid(row=0, column=0, columnspan=2, padx=12, pady=12)

        # 모델에 넘길 그림 (MNIST처럼 검은 바탕에 흰 글씨)
        self.그림 = Image.new("L", (그림판_크기, 그림판_크기), 0)
        self.붓 = ImageDraw.Draw(self.그림)
        self.이전점 = None

        self.결과_글 = tk.StringVar(value="숫자를 그려 보세요.")
        tk.Label(창, textvariable=self.결과_글, font=("맑은 고딕", 14), justify="left",
                 anchor="w", width=24, height=4).grid(row=1, column=0, padx=12, sticky="w")
        tk.Button(창, text="지우기", width=8, command=self.지우기).grid(row=1, column=1, padx=12)

        self.그림판.bind("<Button-1>", self.그리기_시작)
        self.그림판.bind("<B1-Motion>", self.그리기)
        self.그림판.bind("<ButtonRelease-1>", self.그리기_끝)

    def 점찍기(self, x, y):
        반 = 펜_굵기 / 2
        self.그림판.create_oval(x - 반, y - 반, x + 반, y + 반, fill="black", outline="black")
        self.붓.ellipse((x - 반, y - 반, x + 반, y + 반), fill=255)

    def 그리기_시작(self, 사건):
        self.이전점 = (사건.x, 사건.y)
        self.점찍기(사건.x, 사건.y)

    def 그리기(self, 사건):
        if self.이전점 is not None:
            x0, y0 = self.이전점
            self.그림판.create_line(x0, y0, 사건.x, 사건.y, width=펜_굵기, fill="black",
                                  capstyle=tk.ROUND, smooth=True)
            self.붓.line((x0, y0, 사건.x, 사건.y), fill=255, width=펜_굵기)
            self.점찍기(사건.x, 사건.y)
        self.이전점 = (사건.x, 사건.y)

    def 그리기_끝(self, 사건):
        self.이전점 = None
        self.인식()

    def 지우기(self):
        self.그림판.delete("all")
        self.붓.rectangle((0, 0, 그림판_크기, 그림판_크기), fill=0)
        self.결과_글.set("숫자를 그려 보세요.")

    def 인식(self):
        배열28 = 전처리(self.그림)
        if 배열28 is None:
            return
        입력 = torch.from_numpy(정규화(배열28)).float().view(1, 1, 28, 28)
        with torch.no_grad():
            확률 = torch.softmax(self.모델(입력), dim=1)[0]
        값들, 숫자들 = 확률.topk(3)
        줄들 = [f"인식 결과: {숫자들[0].item()}"]
        for 순위, (숫자, 값) in enumerate(zip(숫자들.tolist(), 값들.tolist()), 1):
            줄들.append(f"{순위}위  {숫자}  ({값 * 100:.1f}%)")
        self.결과_글.set("\n".join(줄들))


if __name__ == "__main__":
    창 = tk.Tk()
    손글씨앱(창)
    창.mainloop()
