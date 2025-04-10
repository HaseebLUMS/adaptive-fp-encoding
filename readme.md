# Float32 Bitstream Encoder

A configurable encoder/decoder for `float32` tensors that enables compact storage and flexible lossy or lossless reconstruction. 

This library splits IEEE 754 float32 numbers into two bitstreams:
- **Primary Stream**: Contains sign, exponent, and configurable upper mantissa bits
- **Residual Stream**: Contains the remaining mantissa bits

## 🔧 Features

- Encode and decode `float32` tensors with full or partial precision
- Adjustable **aggression** parameter to trade off accuracy vs. compression
- Supports lossy decoding using just the primary stream
- Designed with PyTorch tensors for GPU/CPU compatibility

---

## 🧠 How It Works

Each IEEE 754 `float32` number has:
- 1 sign bit
- 8 exponent bits
- 23 mantissa bits

The encoder splits the 23-bit mantissa into:
- `ceil((1 - aggression) * 23)` bits in the **primary stream**
- Remaining bits in the **residual stream**

**Aggression = 0.0** → all mantissa bits go to primary (lossless)  
**Aggression = 1.0** → all mantissa bits go to residual (minimal precision in primary)  
**Aggression = 0.5** → even split of mantissa bits

---

## 📦 Installation

No pip package yet — just clone the repo:

```bash
git clone <this repo>
cd <this repo>
python3 main.py