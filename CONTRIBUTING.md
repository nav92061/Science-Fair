# Contributing

Thanks for helping. Good places to start:

- **Source an assumption.** Several generator settings are marked
  "Assumption" in [docs/METHODS.md](docs/METHODS.md#key-generator-settings).
  If you find a published value for one, open a pull request that cites the
  paper and says where the number appears (table or page).
- **Test on real data.** If you have approved access to paired echo, strain and
  clinical data, a script that scores it with a saved model would be the most
  useful addition. Never commit patient data or EchoNet files.
- **Fix bugs or improve tests.**

## Setup

```bash
pip install -e ".[dev]"
pytest -q
ruff check src tests
```

The tests use demo data and the quick sampler, so they run in about 30 seconds
and need neither EchoNet nor SDV.

## Ground rules

- Report only numbers from an actual run, and say which data source and sampler
  produced them (`summary.json` records both).
- Keep the "not for clinical use" framing. Do not add treatment or
  sports-participation advice to the report.
- Keep changes small and focused, with a test where it makes sense.
