# Data Sources

- True/false factual statements (probe training): Azaria & Mitchell (2023),
  via [Representation Engineering](https://github.com/andyzoujm/representation-engineering), Zou et al. (2023).
- DeceptionBench scenarios (evaluation): Huang et al. (2025),
  [repository](https://github.com/Aries-iai/DeceptionBench). CC BY-SA 4.0,
  academic use.
- Sentiment statements (auxiliary probes): UMSAB via
  [XLM-T](https://github.com/cardiffnlp/xlm-t), Barbieri et al. (2022), for
  EN/PT/ES/DE. ChnSentiCorp, Tan & Zhang (2008), for ZH.
- Control corpus (calibration):
  [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca),
  Taori et al. (2023). CC BY-NC 4.0.

The Portuguese, Spanish, German, and Chinese versions are the authors'
machine translations and keep their sources' terms. Rollout generation and
judge-transport code adapts the [MASK](https://github.com/centerforaisafety/mask)
codebase (MIT). All redistributed data remains subject to its upstream terms.
