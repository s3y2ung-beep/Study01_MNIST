# Study01_MNIST

손글씨 숫자 인식 (MNIST). 같은 CNN 모델을 데스크톱 앱과 웹 앱 두 가지로 만들었습니다.

- `desktop_version/`: PyTorch 학습과 tkinter 데스크톱 앱 ([설명](desktop_version/README.md))
- `web_version/`: 외부 라이브러리 없이 순수 자바스크립트로 추론하는 웹 앱 (GitHub Pages 배포)

## 웹 앱 검증 결과

| 항목 | 기준 | 실측 |
|---|---|---|
| 순전파 일치 | 확률 최대 절대차 1e-4 이하 | 5.41e-7 |
| 전체 정확도 | 200장 중 97% 이상 | 98.0% (196장) |
| 손그림 흉내 | 숫자 0~9 × 5가지 형태, 크기·기울기·획 굵기를 바꾼 500장 | 99.2% |

설계와 구현 계획은 `docs/superpowers/`에 있습니다.
