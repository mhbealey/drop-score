"""Smoke test for to_scalar and ensure_series helpers."""
import numpy as np
import pandas as pd
from utils import to_scalar, ensure_series

# to_scalar tests
assert isinstance(to_scalar(3.14), float)
assert to_scalar(3.14) == 3.14

assert isinstance(to_scalar(np.float64(2.71)), float)
assert to_scalar(np.float64(2.71)) == 2.71

assert isinstance(to_scalar(pd.Series([42.0])), float)
assert to_scalar(pd.Series([42.0])) == 42.0

assert isinstance(to_scalar(pd.DataFrame({'a': [99.0]})), float)
assert to_scalar(pd.DataFrame({'a': [99.0]})) == 99.0

# Multi-row DataFrame, .iloc[-1] returns a Series — to_scalar handles it
df = pd.DataFrame({'Close': [10.0, 20.0, 30.0]})
val = df['Close'].iloc[-1]
assert isinstance(to_scalar(val), float)
assert to_scalar(val) == 30.0

# DataFrame column that is itself a DataFrame (MultiIndex case)
df2 = pd.DataFrame({('Close', 'SPY'): [100.0, 200.0, 300.0]})
col = df2['Close']  # This is a DataFrame with column 'SPY'
assert isinstance(col, pd.DataFrame)
row_val = col.iloc[-1]  # This is a Series
assert isinstance(to_scalar(row_val), float)
assert to_scalar(row_val) == 300.0

# ensure_series tests
s = pd.Series([1, 2, 3])
assert isinstance(ensure_series(s), pd.Series)

df_single = pd.DataFrame({'x': [1, 2, 3]})
result = ensure_series(df_single)
assert isinstance(result, pd.Series)

print("All tests passed.")
