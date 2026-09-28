// 웹 앱의 시작점: 모델을 불러오고, 그림판에서 손을 떼면 인식하여 상위 3개 후보를 보여 준다.

import { 모델_불러오기, 예측, 상위후보 } from "./모델.js";
import { 전처리, 정규화 } from "./전처리.js";
import { 그림판 } from "./그림판.js";

const 상태 = document.getElementById("상태");
const 결과 = document.getElementById("결과");
const 후보목록 = document.getElementById("후보");
let 모델 = null;

function 인식() {
  if (!모델) return; // 적재가 끝나면 아래에서 한 번 더 부른다
  const 배열 = 전처리(판.밝기(), 280, 280);
  if (!배열) return;
  const 확률 = 예측(모델, 정규화(배열, 모델.정규화));
  const 후보 = 상위후보(확률, 3);
  결과.textContent = 후보[0].숫자;
  후보목록.replaceChildren(
    ...후보.map((h, i) => {
      const 줄 = document.createElement("li");
      줄.innerHTML = `<span class="순위">${i + 1}위</span><span class="숫자">${h.숫자}</span>` +
        `<span class="막대"><span style="width:${(h.확률 * 100).toFixed(1)}%"></span></span>` +
        `<span class="값">${(h.확률 * 100).toFixed(1)}%</span>`;
      return 줄;
    })
  );
}

const 판 = new 그림판(document.getElementById("그림판"), 인식);

document.getElementById("지우기").addEventListener("click", () => {
  판.지우기();
  결과.textContent = "?";
  후보목록.replaceChildren();
});

try {
  모델 = await 모델_불러오기();
  상태.textContent = "숫자를 그려 보세요.";
  인식(); // 적재 중에 그린 그림이 있으면 바로 인식
} catch (오류) {
  상태.textContent = "모델을 불러오지 못했습니다. 웹 서버로 열었는지 확인하세요.";
  상태.classList.add("오류");
  console.error(오류);
}
