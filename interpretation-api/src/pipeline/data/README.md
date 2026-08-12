# Optional reference data

Nothing in this directory is shipped with the repository. The pipeline runs without it;
fetching it improves one analysis, as described below.

## `collectri_human.tsv` — CollecTRI TF→target regulon (human)

**Not included — fetch it yourself** (see *How to obtain it* below). It is third-party data
redistributed under its own terms rather than this repository's licence, so we point at the
source instead of vendoring a copy.

Signed transcription-factor → target-gene regulatory network, used by
`src/pipeline/services/regulon_service.py` to infer candidate **upstream regulators** of a
gene set (the "Upstream Regulator Candidates" report section). CollecTRI is the default
regulon of the [decoupleR](https://decoupler-py.readthedocs.io/) ecosystem and is built to
answer "which genes does TF *X* control", which KEGG gene-expression edges cannot for
enzyme-centric metabolic programs.

**Format** (tab-separated, one header row):

| column   | meaning                                                        |
|----------|----------------------------------------------------------------|
| `source` | TF HGNC symbol (upper-case)                                     |
| `target` | target-gene HGNC symbol (upper-case)                           |
| `weight` | edge sign: `+1` activation, `-1` repression, `0` unknown sign  |

A 4th optional `confidence` column (e.g. DoRothEA A–E) is tolerated by the loader if a
future drop-in provides it. The loader reads this `source / target / weight` TSV schema; a
different source (DoRothEA, or an unsigned MSigDB C3:TFT `.gmt`) should be converted to it
first — for an unsigned source set every `weight` to `0`, and the service reports
`inferred_tf_activity = 'unknown'` for those edges.

### How to obtain it

Save the result as `collectri_human.tsv` **in this directory** — that is the path the
loader expects (`_DEFAULT_PATH` in `src/pipeline/services/regulon_service.py`).

The simplest route is decoupleR:

```python
import decoupler as dc

net = dc.get_collectri(organism='human', split_complexes=False)
net[['source', 'target', 'weight']].to_csv(
    'collectri_human.tsv', sep='\t', index=False
)
```

Equivalently, straight from the OmniPath web service (which serves the CollecTRI network),
reduced to `source / target / weight`:

```
url = "https://omnipathdb.org/interactions?datasets=collectri&genesymbols=1&fields=sources"
# weight = +1 if is_stimulation and not is_inhibition
#          -1 if is_inhibition and not is_stimulation
#          0  otherwise (unknown / conflicting sign)
# drop self-loops and duplicate (source,target) pairs; upper-case symbols.
```

Please cite the underlying work and observe its terms: Müller-Dott et al., *Nucleic Acids
Research* 2023 (CollecTRI); Türei et al., *Mol. Syst. Biol.* 2021 (OmniPath). OmniPath
aggregates resources with differing licences, some academic-use-only — check them for your
use case.

A correct fetch yields roughly 62k edges, ~1200 TFs and ~6900 genes.

### What happens if you skip it

Nothing breaks. The service reports `available == False`, and Step 3's upstream-regulator
analysis falls back from hypergeometric enrichment over a curated regulon to **LLM-proposed
candidate TFs, clearly labelled as hypotheses** (`evidence_source: 'llm_hypothesis'`,
`fallback_reason: 'db_unavailable'`). Every other part of the pipeline is unaffected. This
is the default configuration and is covered by the test suite.

The file is read from disk at start-up and never fetched at request time, so once installed
the runtime stays offline and reproducible.
