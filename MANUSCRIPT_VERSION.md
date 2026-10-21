# NEST manuscript version and provenance

Verified on 2026-10-21. The default `main` branch contains the manuscript code; cloning the repository does not require switching to an experimental branch.

## Paper and result anchors

Title: **NEST: Normality-guided learning and evolving-neighborhood scoring for cross-domain dynamic graph anomaly detection**.

- DGraphFin-to-FiGraph zero-shot, seeds 42/43/44: ROC-AUC `0.7704 +/- 0.0132`; AP `0.0926 +/- 0.0114`, evaluated on 5,132 binary-labeled final-snapshot nodes.
- Fixed-query few-shot reference: ROC-AUC `0.766450367563441`, evaluated after excluding the reserved pool of 20 anomalous and 100 normal nodes.
- Ten-shot correction: 10 anomalous and 50 normal support nodes; ROC-AUC `0.7700228297407641` on the same 5,012 query nodes (103 anomalies).
- Within each seed, the query set is fixed and support sets are nested for 5/10/20 shots. The zero-shot main table and few-shot query reference have different evaluation populations.

## Source history

The manuscript implementation is preserved at commit `9776c21c09dd67379ecd43775286a26e2162955e` (formerly reached through `exp/spad-component-ablations`). Its history includes `c1d7b4b` for the final source objective and `e3234f9` for the frozen-backbone few-shot code. The later reverse-transfer and Elliptic Actors additions are separate experiments and are not needed for this manuscript.

`main` incorporates that implementation while preserving the previous default-branch history ending at `87d3678cc6622de66ed904aa50c9756299d4233f`. Documentation and plot display labels now use NEST; executable historical identifiers and result keys remain unchanged.

The following raw-file SHA-256 values identify the original experiment workspace and match the source hashes recorded in the 2026-10-07 final-component run manifest. Git may convert LF/CRLF line endings on checkout, which changes raw-file SHA-256 without changing source content. The Git blob IDs identify the repository content independently of that checkout conversion; the release and original workspace have identical blob IDs.

| File | Experiment workspace SHA-256 | Git blob ID |
| --- | --- | --- |
| `train_anomaly_gfm.py` | `380a05ef04d79cddd9258a8cce0961a80570747376bdb91b411a684371e92b16` | `caf04fc0dcd1f9dcddca1e6cb1f30fd73bc9d023` |
| `anomaly_gfm_complete_model.py` | `bfd36238265b75eab4607c84eb3f60fd59cb9573efbfa1c6e129ac90fd8ed188` | `18c18d065f5a0fafb936a81075a672a445a111f3` |
| `data_processor.py` | `ece7ccc0e2c09c1f9569cd9c51c925c7c04eeab9902580079ca3e0c32ba992ac` | `8d1e7ec145657350dcf09070a0b92a212ee82ee2` |

## Archived result records

- `results/component_ablation_runs.csv`: includes the retained full-model runs `spad_ablation_no_reg_seed42/43/44` as well as historical configurations. Historical variants with an extra regularizer are not the final matched ablation table.
- `results/component_ablation_final_20261007_runs.csv`: 18 final-objective ablation reruns (six variants, three seeds).
- `results/component_ablation_final_20261007_table3_runs.csv`: 21 rows, including the three verified full-model references used in Table 3.
- `results/few_shot_curve_runs.csv`: nine fixed-query runs, seeds 42/43/44 and shots 5/10/20, including support and query hashes.
- `results/few_shot_curve_mean_std.csv`: means and population standard deviations for those nine runs.

Result rows retain their original run names, original machine paths, metrics, and historical labels. Paths inside archived records identify the original local artifacts; they are not download URLs or paths guaranteed to exist in a fresh clone. Raw datasets and the complete collection of three-seed model/few-shot checkpoints are not included. In particular, the inherited seed-42 checkpoint is a historical snapshot, not the complete three-seed manuscript release. Reproduce the current runs with `RUNNING.md`; validating saved few-shot artifacts also requires those artifacts to be present locally.

## Verification performed during recovery

1. Compared the three original workspace source files to the recorded final-run hashes: all matched. Checked release Git blob IDs and content after line-ending normalization against that workspace: all matched.
2. Compared core files, `few_shot_finetune.py`, and `few_shot_curve.py` between the manuscript commit and the later local experiment head: no differences.
3. Ran the existing fixed-query validator against all nine locally saved few-shot artifacts: passed.
4. Ran the existing code--manuscript alignment checks: checkpoint state loaded strictly; source logits, temporal loss, residual interaction, neighbor scoring, recency weighting, stable ranks, TCN causality, and reported table statistics passed.

These checks establish the archived code/result linkage. They do not constitute a new full training run. The manuscript's disclosed evaluation limitations remain applicable.
