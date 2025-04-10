import torch

class Encoding:
    """
    A class for custom float32 encoding with configurable precision split 
    between a primary and residual stream based on an 'aggression' parameter.

    This encoder approximates IEEE 754 float32 values (1 sign bit, 8 exponent bits, 23 mantissa bits) 
    by splitting the mantissa bits between a compact 'primary_stream' and a secondary 'residual_stream'.

    Attributes:
        aggression (float): Determines how aggressively mantissa bits are pushed into the residual stream.
                            - aggression = 0.0 → All mantissa bits (23) in the primary stream (lossless).
                            - aggression = 1.0 → All mantissa bits in the residual stream (minimal primary precision).
                            - aggression = 0.5 → Mantissa split evenly (11 bits primary, 12 bits residual).
        primary_m_bits (int): Number of mantissa bits encoded in the primary stream.
        residual_m_bits (int): Number of mantissa bits stored in the residual stream.

    Methods:
        encode(tensor_fp32):
            Encodes a float32 tensor into two integer streams: primary and residual.
        
        decode(primary_stream, residual_stream):
            Fully reconstructs the original float32 values using both primary and residual streams.

        decode_low(primary_stream):
            Approximates the original float32 values using only the primary stream,
            assuming all residual bits are zero (useful for lossy decoding).

    Use Case:
        This class enables dynamic tradeoffs between compression and precision
        for distributed systems, low-bandwidth scenarios, or lossy inference.

    Example:
        >>> enc = Encoding(aggression=0.7)
        >>> primary, residual = enc.encode(tensor)
        >>> recovered = enc.decode(primary, residual)
        >>> lossy = enc.decode_low(primary)
    """

    def __init__(self, aggression: float=0.5):
        assert 0.0 <= aggression <= 1.0, "Aggression must be between 0 and 1"
        self.aggression = aggression
        self.m_bits = 23
        self.primary_m_bits = round((1 - aggression) * self.m_bits)
        self.residual_m_bits = self.m_bits - self.primary_m_bits

    def encode(self, tensor_fp32: torch.Tensor):
        tensor = tensor_fp32.reshape(-1)
        raw = tensor.view(torch.int32)

        sign = (raw >> 31) & 0x1
        exponent = (raw >> 23) & 0xFF
        mantissa = raw & 0x7FFFFF

        mantissa_primary = mantissa >> self.residual_m_bits
        mantissa_residual = mantissa & ((1 << self.residual_m_bits) - 1)

        primary = (sign << (8 + self.primary_m_bits)) | (exponent << self.primary_m_bits) | mantissa_primary

        return primary, mantissa_residual

    def decode(self, primary_stream: torch.Tensor, residual_stream: torch.Tensor):
        primary = primary_stream.to(torch.int32)
        residual = residual_stream.to(torch.int32)

        mantissa_primary = primary & ((1 << self.primary_m_bits) - 1)
        exponent = (primary >> self.primary_m_bits) & 0xFF
        sign = (primary >> (8 + self.primary_m_bits)) & 0x1

        mantissa = (mantissa_primary << self.residual_m_bits) | residual

        raw = (sign << 31) | (exponent << 23) | mantissa
        return raw.view(torch.float32)

    def decode_low(self, primary_stream: torch.Tensor):
        primary = primary_stream.to(torch.int32)

        mantissa_primary = primary & ((1 << self.primary_m_bits) - 1)
        exponent = (primary >> self.primary_m_bits) & 0xFF
        sign = (primary >> (8 + self.primary_m_bits)) & 0x1

        mantissa = mantissa_primary << self.residual_m_bits

        raw = (sign << 31) | (exponent << 23) | mantissa
        return raw.view(torch.float32)
