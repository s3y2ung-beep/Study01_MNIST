// preprocess.py 의 3단계 전처리를 자바스크립트로 옮긴 것.
// 1단계 여백 자르기 → 2단계 비율을 유지하며 긴 변을 20으로 축소 → 3단계 밝기 무게중심을 28x28 중앙으로 이동
// 다른 점은 1곳: PIL의 LANCZOS 축소는 브라우저에서 같게 재현할 수 없어 면적 평균 축소를 직접 구현했다.

const 상자_크기 = 20;
const 최종_크기 = 28;

/** 0.5를 항상 위로 올리는 반올림. 파이썬 쪽 반올림()과 같은 규칙. */
export function 반올림(x) {
  return Math.floor(x + 0.5);
}

/**
 * 면적 평균 축소. 출력 한 칸이 덮는 원본 영역의 밝기를 겹친 넓이만큼 가중 평균한다.
 * 원본 너비 w 가 새너비의 배수가 아니면 경계가 칸 중간에 걸리므로 겹친 넓이를 실수로 계산한다.
 */
function 면적평균축소(원본, w, h, 새너비, 새높이) {
  const 결과 = new Float32Array(새너비 * 새높이);
  const 가로비 = w / 새너비;
  const 세로비 = h / 새높이;
  for (let oy = 0; oy < 새높이; oy++) {
    const y0 = oy * 세로비;
    // 부동소수점 오차로 끝이 h를 살짝 넘을 수 있어(예: 19 * (21 / 19) = 21.000000000000004) 잘라 둔다
    const y1 = Math.min((oy + 1) * 세로비, h);
    for (let ox = 0; ox < 새너비; ox++) {
      const x0 = ox * 가로비;
      const x1 = Math.min((ox + 1) * 가로비, w);
      let 합 = 0;
      let 넓이합 = 0;
      for (let sy = Math.floor(y0); sy < Math.ceil(y1); sy++) {
        const 세로겹침 = Math.min(sy + 1, y1) - Math.max(sy, y0);
        if (세로겹침 <= 0) continue;
        for (let sx = Math.floor(x0); sx < Math.ceil(x1); sx++) {
          const 가로겹침 = Math.min(sx + 1, x1) - Math.max(sx, x0);
          if (가로겹침 <= 0) continue;
          const 넓이 = 세로겹침 * 가로겹침;
          합 += 원본[sy * w + sx] * 넓이;
          넓이합 += 넓이;
        }
      }
      결과[oy * 새너비 + ox] = 합 / 넓이합;
    }
  }
  return 결과;
}

/**
 * 흰 글씨/검은 바탕 흑백 그림(0~255, 길이 너비*높이)을 받아 0~1 범위의 28x28(길이 784) 배열을 돌려준다.
 * 아무것도 그려지지 않았으면 null.
 */
export function 전처리(밝기, 너비, 높이) {
  // 1단계: 여백 자르기
  let 위 = 높이, 아래 = -1, 왼 = 너비, 오른 = -1;
  for (let y = 0; y < 높이; y++) {
    for (let x = 0; x < 너비; x++) {
      if (밝기[y * 너비 + x] > 0) {
        if (y < 위) 위 = y;
        if (y > 아래) 아래 = y;
        if (x < 왼) 왼 = x;
        if (x > 오른) 오른 = x;
      }
    }
  }
  if (아래 < 0) return null;
  const w = 오른 - 왼 + 1;
  const h = 아래 - 위 + 1;
  const 잘린 = new Float32Array(w * h);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) 잘린[y * w + x] = 밝기[(위 + y) * 너비 + (왼 + x)];
  }

  // 2단계: 비율 유지, 긴 변을 20으로
  const 배율 = 상자_크기 / Math.max(w, h);
  const 새너비 = Math.max(1, 반올림(w * 배율));
  const 새높이 = Math.max(1, 반올림(h * 배율));
  const 작은 = 면적평균축소(잘린, w, h, 새너비, 새높이).map((v) => v / 255);

  // 3단계: 밝기 무게중심을 (14, 14)로
  let 합 = 0, 합y = 0, 합x = 0;
  for (let y = 0; y < 새높이; y++) {
    for (let x = 0; x < 새너비; x++) {
      const v = 작은[y * 새너비 + x];
      합 += v;
      합y += y * v;
      합x += x * v;
    }
  }
  const 제한 = (v, 최소, 최대) => Math.min(Math.max(v, 최소), 최대);
  const 시작y = 제한(반올림(최종_크기 / 2 - 합y / 합), 0, 최종_크기 - 새높이);
  const 시작x = 제한(반올림(최종_크기 / 2 - 합x / 합), 0, 최종_크기 - 새너비);

  const 결과 = new Float32Array(최종_크기 * 최종_크기);
  for (let y = 0; y < 새높이; y++) {
    for (let x = 0; x < 새너비; x++) {
      결과[(시작y + y) * 최종_크기 + (시작x + x)] = 작은[y * 새너비 + x];
    }
  }
  return 결과;
}

/** 0~1 범위 28x28 배열을 모델 입력용으로 정규화. 상수는 가중치정보.json 에서 읽은 값을 쓴다. */
export function 정규화(배열784, 상수) {
  return 배열784.map((v) => (v - 상수.평균) / 상수.표준편차);
}
