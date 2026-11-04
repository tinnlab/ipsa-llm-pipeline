# Gene-set collections (GMT)

**Nothing in this directory is shipped with the repository.** The pipeline runs without
it; adding a collection enables one lookup, described below.

## What this is for

For a KEGG pathway the engine reads membership from the KGML it fetches. For every other
pathway — GO, Reactome, MitoPathways, or a custom set — the caller sends only the
differentially expressed members of that pathway, and on the meta-analysis path it sends
no genes at all. `src/pipeline/services/pathway_membership_service.py` recovers the full
membership by matching those pathways against the `.gmt` files placed here, so Step 3 can
request interactions for all of a pathway's genes rather than just the DE ones.

With this directory empty, that lookup finds nothing: each pathway keeps the genes the
caller supplied, and the report's Methods section cannot attribute a pathway database.
The service prints a one-line warning at start-up when it loads no files, so an empty
directory is never silent.

## Not included — add your own

The collections are third-party data under their own redistribution terms, so we point at
the sources instead of vendoring copies.

Drop one or more `.gmt` files directly in this directory. Any file matching `*.gmt` is
loaded at start-up; the file name is used to report which database a match came from, so
keep the publisher's name (for example `Mouse.MitoPathways3.0.gmt`).

Common sources:

- **MitoPathways / MitoCarta** — https://www.broadinstitute.org/mitocarta
- **GO, Reactome, Hallmark and others** — MSigDB, https://www.gsea-msigdb.org/gsea/msigdb

## Format

One pathway per line, tab-separated: set name, a description, then the gene symbols.

```
TCA cycle	Citric acid cycle	ACO2	CS	FH	IDH2	MDH2	OGDH	SDHA
```

Matching is deliberately conservative, because set names collide with common pathway
names ("Translation", "Apoptosis", "TCA cycle"). See the module docstring in
`pathway_membership_service.py` for the association and matching rules.
