// Captura del micrófono → 16 kHz mono Int16 en bloques de 512 muestras (32 ms), con nivel RMS para la UI.
class MicProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.acc = 0;
    this.sum = 0;
    this.count = 0;
    this.out = new Int16Array(512);
    this.n = 0;
  }
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      // Diezmado con promedio (filtro paso-bajo sencillo) de la frecuencia nativa a 16 kHz.
      this.sum += ch[i];
      this.count++;
      this.acc += 1;
      if (this.acc >= this.ratio) {
        this.acc -= this.ratio;
        const v = Math.max(-1, Math.min(1, this.sum / this.count));
        this.sum = 0;
        this.count = 0;
        this.out[this.n++] = v < 0 ? v * 0x8000 : v * 0x7fff;
        if (this.n === 512) {
          let e = 0;
          for (let k = 0; k < 512; k++) e += (this.out[k] / 32768) ** 2;
          this.port.postMessage({ pcm: this.out.buffer, rms: Math.sqrt(e / 512) }, [this.out.buffer]);
          this.out = new Int16Array(512);
          this.n = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor("mic-processor", MicProcessor);
