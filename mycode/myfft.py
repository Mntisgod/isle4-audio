import numpy as np
import time


def FFT(f: np.ndarray) -> np.ndarray:
    n = len(f)
    w = np.exp(-2j * np.pi / n)
    w_N = w ** np.arange(n//2)
    if n == 1:
        return f[0]
    F_even = FFT(f[::2])
    F_odd = FFT(f[1::2])
    F = np.zeros(n, dtype=np.complex128)
    F[0:n//2] = F_even + w_N * F_odd
    F[n//2:] = F_even - w_N * F_odd

    return F


# f = np.array(input().split(), dtype=np.complex128)
input_array = np.arange(2**14, dtype=int)
start = time.perf_counter()
FFT(input_array)
end = time.perf_counter()
print(end - start)
start = time.perf_counter()
np.fft.rfft(input_array)
end = time.perf_counter()
print(end - start)

print(np.allclose(FFT(input_array), np.fft.fft(input_array), atol=1e-10))