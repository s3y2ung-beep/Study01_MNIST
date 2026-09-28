// 마우스와 터치로 그리는 280x280 그림판.
// 화면에는 흰 바탕에 검은 글씨로 보이고, 모델에 넘길 때는 MNIST처럼 흰 글씨/검은 바탕 밝기로 바꾼다.

const 펜_굵기 = 18;

export class 그림판 {
  /** @param {HTMLCanvasElement} 캔버스 @param {() => void} 그리기끝 손을 뗄 때 부르는 함수 */
  constructor(캔버스, 그리기끝) {
    this.캔버스 = 캔버스;
    this.붓 = 캔버스.getContext("2d", { willReadFrequently: true });
    this.그리기끝 = 그리기끝;
    this.이전점 = null;
    this.지우기();

    캔버스.addEventListener("pointerdown", (e) => this.#시작(e));
    캔버스.addEventListener("pointermove", (e) => this.#이동(e));
    캔버스.addEventListener("pointerup", (e) => this.#끝(e));
    캔버스.addEventListener("pointercancel", (e) => this.#끝(e));
  }

  /** 화면 좌표를 캔버스 내부 좌표(280x280)로 바꾼다. 화면에서 캔버스가 줄어 보여도 맞게 그려진다. */
  #좌표(e) {
    const 틀 = this.캔버스.getBoundingClientRect();
    return {
      x: ((e.clientX - 틀.left) / 틀.width) * this.캔버스.width,
      y: ((e.clientY - 틀.top) / 틀.height) * this.캔버스.height,
    };
  }

  #점(p) {
    this.붓.beginPath();
    this.붓.arc(p.x, p.y, 펜_굵기 / 2, 0, Math.PI * 2);
    this.붓.fill();
  }

  #시작(e) {
    e.preventDefault();
    this.캔버스.setPointerCapture(e.pointerId);
    this.이전점 = this.#좌표(e);
    this.#점(this.이전점);
  }

  #이동(e) {
    if (!this.이전점) return;
    e.preventDefault();
    const p = this.#좌표(e);
    this.붓.beginPath();
    this.붓.moveTo(this.이전점.x, this.이전점.y);
    this.붓.lineTo(p.x, p.y);
    this.붓.stroke();
    this.#점(p);
    this.이전점 = p;
  }

  #끝(e) {
    if (!this.이전점) return;
    this.이전점 = null;
    this.그리기끝();
  }

  지우기() {
    const 붓 = this.붓;
    붓.fillStyle = "#fff";
    붓.fillRect(0, 0, this.캔버스.width, this.캔버스.height);
    붓.fillStyle = "#000";
    붓.strokeStyle = "#000";
    붓.lineWidth = 펜_굵기;
    붓.lineCap = "round";
    붓.lineJoin = "round";
  }

  /** 흰 글씨/검은 바탕 밝기 배열(0~255)을 돌려준다. */
  밝기() {
    const { width: w, height: h } = this.캔버스;
    const 화소 = this.붓.getImageData(0, 0, w, h).data;
    const 결과 = new Uint8ClampedArray(w * h);
    for (let i = 0; i < w * h; i++) 결과[i] = 255 - 화소[i * 4];
    return 결과;
  }
}
