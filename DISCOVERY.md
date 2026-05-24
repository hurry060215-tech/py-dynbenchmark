# Discovery — py-dynbenchmark

## 1. Special status

py-dynbenchmark is **not a rebuildr port** in the conventional sense. It is a *benchmarking framework* that consumes the other trajectory ports and scores them against synthetic gold standards.

The omicverse-rebuildr protocol applies to "an R package with a clear numerical output that should match R bit-for-bit". py-dynbenchmark instead implements four well-known benchmarking metrics (HIM, F1_branches, cor_dist, wcor_features) and provides synthetic dataset generators. None of these have a single canonical R reference — they were originally implemented across `dynverse/dyneval` and `dynverse/dyntoy`, neither of which we attempt to mirror bit-for-bit.

## 2. R dependencies + py-mirror reuse

R dyneval-family DESCRIPTION lists: `igraph`, `dynwrap`, `lpSolve`, `philentropy`, `dynfeature`, `purrr`, `furrr`.

| R component | Python equivalent | Status in py-dynbenchmark |
|---|---|---|
| `dyneval::calculate_metrics` | inline metric implementations | ✅ ported |
| `dyneval::metric_him` | `pydynbenchmark.him_distance` | ✅ ported |
| `dyneval::calculate_mapping_*` (F1) | `pydynbenchmark.f1_branches` | ✅ ported |
| `dyneval::metric_correlation` | `pydynbenchmark.cor_dist` | ✅ ported |
| `dyneval::metric_featureimp` | `pydynbenchmark.wcor_features` | ✅ (simplified) |
| `dyntoy::generate_dataset_*` | `pydynbenchmark.generate_{linear,bifurcation,tree}` | ✅ |
| `dynwrap` (TI method wrapping) | `pydynbenchmark.run_benchmark` harness | ✅ (different design) |

## 3. Decision

**Build, don't port.** The R framework's value is in the metric formulas and synthetic generators, not in any specific numerical output. The Python implementation provides equivalent functionality with the same metric definitions; users who need bit-for-bit matching with R should use the R framework directly.

## 4. Why this DISCOVERY.md is short

Per the rebuildr README:
> "Don't use it when... You want a Python algorithm that's *better* than R, not identical."

py-dynbenchmark is closer to "an independent Python implementation of well-known benchmarking concepts" than to "a 1:1 port of a specific R package". It is appropriately listed under `omicverse/` as a benchmarking tool, not under the `py-<RPkg>` ports.
