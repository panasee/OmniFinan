import numpy as np
import pandas as pd
import talib as ta


def XMA(src: np.ndarray, N: int, array=False) -> np.ndarray:
    """XMA function implementation"""
    if not isinstance(src, pd.Series):
        src = pd.Series(src)
    data_len = len(src)
    half_len: int = (N // 2) + (1 if N % 2 else 0)
    if data_len < half_len:
        out = np.array([np.nan for i in range(data_len)], dtype=float)
        if array:
            return out
        return out[-half_len:] if len(out) else np.array([], dtype=float)

    head = np.array([ta.MA(src[0:ilen], ilen)[-1] for ilen in range(half_len, N)])
    out = head
    if data_len >= N:
        body = ta.MA(src, N)[N - 1 :]
        out = np.append(out, body)
        tail = np.array([ta.MA(src[-ilen:], ilen)[-1] for ilen in range(N - 1, half_len - 1, -1)])
        out = np.append(out, tail)

    if array:
        return out

    return out[-half_len:]

def cross_over(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    a from below to above b (UPCROSS)
    """
    return (a.shift(1) < b.shift(1)) & (a > b)

def cross_under(a: pd.Series, b: pd.Series) -> pd.Series:
    """
    a from above to below b (DOWNCROSS)
    """
    return (a.shift(1) > b.shift(1)) & (a < b)


if __name__ == "__main__":
    import pandas as pd

    # Create a DataFrame with a sample column
    df = pd.DataFrame({'src': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, dtype=float)
    N = 5

    # Apply XMA to the DataFrame column
    df['XMA'] = XMA(df['src'], N)

    print(df)