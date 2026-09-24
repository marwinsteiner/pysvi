# Market context and conventions

A surface stores a rate and per-slice forwards, and maturities are bare year-fraction floats — nothing says which day count produced $T = 0.5$, or guarantees discounting, forwards and time share one convention. `MarketContext` is the single source of numeraire truth:

```python
from pysvi import MarketContext, OptionChain

ctx = MarketContext(
    valuation_time="2026-09-24",
    spot=764.3,
    rate=curve,                 # any rate view: float, irm curve/model, callable
    dividend_yield=0.012,
    day_count="ACT/365F",       # or ACT/360, ACT/365.25, BUS/252 (+ holidays)
)
chain = OptionChain.from_dataframe(df, expiry="expiry_date", context=ctx)
```

With a context, the expiry column holds **real dates**, resolved to year fractions through the context's day count from its valuation time; spot, rates and dividends come from the context. Mixing a context with separately supplied `rate`/`spot`/`dividend_yield` raises — coherence is the point. The context also exposes `year_fraction(date)`, `rate_at(T)`, `discount(T_or_date)` and the fallback `forward(T)`, and put-call parity of the fitted surface's prices holds against the context's discount factors by construction.
