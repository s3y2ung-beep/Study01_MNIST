# CLAUDE.md

MNIST 손글씨 숫자 인식 프로젝트. 같은 모델을 두 버전으로 제공한다.

## 두 버전의 관계

| 폴더 | 내용 | 언어 |
|---|---|---|
| `desktop_version/` | 학습, tkinter 데스크톱 앱, 웹용 가중치 내보내기 | 파이썬 (PyTorch) |
| `web_version/` | 브라우저 앱, GitHub Pages 배포 대상 | 순수 자바스크립트 |

각 폴더의 CLAUDE.md에 세부 지침이 있다.

## 가중치 흐름

```
desktop_version/train.py        → desktop_version/mnist_cnn.pt
desktop_version/가중치내보내기.py → web_version/가중치.bin + 가중치정보.json
desktop_version/검증데이터만들기.py → web_version/검증데이터.json (깃 제외)
```

모델을 다시 학습하면 가중치를 다시 내보내고 `web_version/검증.html`로 확인한다.

## 공통 규칙

- 코드, 주석, 식별자를 한글로 쓴다.
- 전처리 3단계(여백 자르기 → 비율 유지 20x20 축소 → 무게중심을 28x28 중앙으로 이동)는 두 버전이 같아야 한다.
- 정규화 상수(0.1307, 0.3081)의 출처는 `desktop_version/preprocess.py` 하나다.

## 실행 명령

- 데스크톱: `cd desktop_version` 후 `py app.py` (맥: `python3 app.py`)
- 웹(로컬): `cd web_version` 후 `py -m http.server 8000`, 브라우저에서 `http://localhost:8000/`
- 배포: main에 푸시하면 `.github/workflows/pages.yml`이 `web_version/`을 GitHub Pages로 올린다.
