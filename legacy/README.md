# Legacy SCYiBL-NLP (2022)

The original RoBERTa multi-label classifier, kept as a reference and future baseline (see issue #10).
It is **not** part of the new `scyibl` package and is not installed by default.

Known limitations (tracked in #10):
- Retrains the model at every server start, with random unseeded splits.
- Expects `data/data_huggingface.csv`, which is no longer in the repository. The dataset now lives in private storage (#16).
- `max_token_len=30` truncates long sentences; computed class weights are ignored in favour of hard-coded `pos_weight`.

To run it you need the private dataset at `legacy/data/data_huggingface.csv`, then:

```bash
cd legacy
docker compose up --build   # serves POST http://localhost:5010/
```
