import torch
from encoding import Encoding

# Example usage
tensor = torch.tensor([1.5, -2.75, 0.0, 100.125], dtype=torch.float32)

enc = Encoding(aggression=0.5)  # 60% mantissa bits go to primary
primary, residual = enc.encode(tensor)

full = enc.decode(primary, residual)
low = enc.decode_low(primary)

print("Original:     ", tensor)
print("Decoded full: ", full)
print("Decoded low:  ", low)
