# Contributing

Contributions, bug reports, and feature requests are welcome. Open an issue or submit a PR on [GitHub](https://github.com/marwinsteiner/pysvi).

## Development setup

```bash
git clone https://github.com/marwinsteiner/pysvi.git
cd pysvi
uv sync --dev
uv run pytest
```

## Adding a parametrization

New models subclass `Parametrization` in `src/pysvi/models.py` and implement `calibrate(k, w_target, **kwargs)` and `total_variance(k, params)`. Register the model in the `get_model` factory in `src/pysvi/calibration.py`, export it from `__init__.py`, add tests mirroring the existing per-model suites, and add a docs page under `docs/models/` following the shared structure (overview, model, parameters, usage, arbitrage behaviour, references).

## Wanted: the original Gamma-Vanna-Volga paper

The Gamma-Vanna-Volga parametrization is something of a holy grail in the quant vol surface literature and would be a great addition to this library. If you have a copy of the original paper, please send it to [marwin.steiner@gmail.com](mailto:marwin.steiner@gmail.com).

## License

MIT
