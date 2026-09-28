// 숫자인식망(model.py)의 순전파를 외부 라이브러리 없이 자바스크립트로 다시 쓴 것.
// 구조: 합성곱(3x3, 덧대기 1) → ReLU → 2x2 최대 풀링을 2번, 전결합 → ReLU → 전결합 → 소프트맥스
// 드롭아웃은 예측 때 아무 일도 하지 않으므로 뺐다.

/** 가중치정보.json 과 가중치.bin 을 읽어 모델을 만든다. */
export async function 모델_불러오기(기준경로 = "./") {
  const [정보응답, 가중치응답] = await Promise.all([
    fetch(기준경로 + "가중치정보.json"),
    fetch(기준경로 + "가중치.bin"),
  ]);
  if (!정보응답.ok || !가중치응답.ok) {
    throw new Error("가중치 파일을 불러오지 못했습니다.");
  }
  const 정보 = await 정보응답.json();
  const 버퍼 = await 가중치응답.arrayBuffer();
  if (버퍼.byteLength % 4 !== 0 || 버퍼.byteLength / 4 !== 정보.파라미터수) {
    throw new Error(`가중치.bin 크기가 맞지 않습니다: ${버퍼.byteLength}바이트`);
  }
  const 전체 = new Float32Array(버퍼);
  const 텐서 = {};
  for (const t of 정보.텐서) {
    텐서[t.이름] = { 값: 전체.subarray(t.시작, t.시작 + t.개수), 형상: t.형상 };
  }
  return { 텐서, 정규화: 정보.정규화 };
}

/** 합성곱(3x3, 덧대기 1, 보폭 1) 뒤에 ReLU. 입력 [입력채널][크기*크기] 평탄 배열. */
function 합성곱_렐루(입력, 입력채널, 크기, 가중치, 편향, 출력채널) {
  const 출력 = new Float32Array(출력채널 * 크기 * 크기);
  for (let o = 0; o < 출력채널; o++) {
    for (let y = 0; y < 크기; y++) {
      for (let x = 0; x < 크기; x++) {
        let 합 = 편향[o];
        for (let c = 0; c < 입력채널; c++) {
          const 입력시작 = c * 크기 * 크기;
          const 가중치시작 = (o * 입력채널 + c) * 9;
          for (let ky = 0; ky < 3; ky++) {
            const yy = y + ky - 1;
            if (yy < 0 || yy >= 크기) continue;
            for (let kx = 0; kx < 3; kx++) {
              const xx = x + kx - 1;
              if (xx < 0 || xx >= 크기) continue;
              합 += 입력[입력시작 + yy * 크기 + xx] * 가중치[가중치시작 + ky * 3 + kx];
            }
          }
        }
        출력[o * 크기 * 크기 + y * 크기 + x] = 합 > 0 ? 합 : 0;
      }
    }
  }
  return 출력;
}

/** 2x2 최대 풀링 (보폭 2). */
function 최대풀링(입력, 채널, 크기) {
  const 반 = 크기 / 2;
  const 출력 = new Float32Array(채널 * 반 * 반);
  for (let c = 0; c < 채널; c++) {
    for (let y = 0; y < 반; y++) {
      for (let x = 0; x < 반; x++) {
        const i = c * 크기 * 크기 + 2 * y * 크기 + 2 * x;
        출력[c * 반 * 반 + y * 반 + x] = Math.max(입력[i], 입력[i + 1], 입력[i + 크기], 입력[i + 크기 + 1]);
      }
    }
  }
  return 출력;
}

/** 전결합층: 출력 = 가중치 · 입력 + 편향. 가중치 형상은 [출력수][입력수]. */
function 전결합(입력, 가중치, 편향, 출력수, 렐루) {
  const 입력수 = 입력.length;
  const 출력 = new Float32Array(출력수);
  for (let o = 0; o < 출력수; o++) {
    let 합 = 편향[o];
    const 시작 = o * 입력수;
    for (let i = 0; i < 입력수; i++) 합 += 가중치[시작 + i] * 입력[i];
    출력[o] = 렐루 && 합 < 0 ? 0 : 합;
  }
  return 출력;
}

/** 소프트맥스. 스프레드 연산 대신 반복문으로 최댓값을 구해 큰 배열에서도 안전하다. */
function 소프트맥스(점수) {
  let 최대 = -Infinity;
  for (const v of 점수) if (v > 최대) 최대 = v;
  const 지수 = Array.from(점수, (v) => Math.exp(v - 최대));
  const 합 = 지수.reduce((a, b) => a + b, 0);
  return 지수.map((v) => v / 합);
}

/** 정규화된 28x28 입력(길이 784)을 받아 숫자 0~9의 확률 10개를 돌려준다. */
export function 예측(모델, 입력784) {
  const t = 모델.텐서;
  let x = 합성곱_렐루(입력784, 1, 28, t["합성곱1.weight"].값, t["합성곱1.bias"].값, 32);
  x = 최대풀링(x, 32, 28); // 32 x 14 x 14
  x = 합성곱_렐루(x, 32, 14, t["합성곱2.weight"].값, t["합성곱2.bias"].값, 64);
  x = 최대풀링(x, 64, 14); // 64 x 7 x 7 (파이토치 flatten 순서와 같음: 채널, 행, 열)
  x = 전결합(x, t["전결합1.weight"].값, t["전결합1.bias"].값, 128, true);
  x = 전결합(x, t["전결합2.weight"].값, t["전결합2.bias"].값, 10, false);
  return 소프트맥스(x);
}

/** 확률 10개에서 상위 k개 후보를 [{숫자, 확률}] 로 돌려준다. */
export function 상위후보(확률, k = 3) {
  return 확률
    .map((p, 숫자) => ({ 숫자, 확률: p }))
    .sort((a, b) => b.확률 - a.확률)
    .slice(0, k);
}
