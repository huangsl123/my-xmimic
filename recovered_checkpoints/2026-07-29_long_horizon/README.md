# Long-horizon checkpoint recovery

These checkpoints were recovered byte-for-byte from the local Git LFS object
store after the ignored `logs/` directory was cleaned during repository/LFS
maintenance.

| File | Internal iteration | Actor / critic input | SHA-256 |
|---|---:|---:|---|
| `state_feedback_init_model_50008.pt` | 50008 | 130 / 346 | `f4d085682d17b0292c24790900400aff2888eb239e0924591844c684f5af7e3f` |
| `preliminary_a_5s_model_50507.pt` | 50507 | 130 / 346 | `5f852b96d59bfaabba510d30856435e662b804118e523c3a0d6071049ebf2e1c` |
| `ab_b_10s_model_50507.pt` | 50507 | 130 / 346 | `760b856233e58cd3352cda8e9aaf8ab1d447dffc40d7575d3c64b00e414b2f0d` |
| `stage1_10s_model_51000.pt` | 51000 | 130 / 346 | `157a5001ef47c7d0d61c22629af986b4e104c7d1240afda82cb139b63034130b` |
| `stage1_10s_model_51500.pt` | 51500 | 130 / 346 | `f40a35aaf253981eed22de1f32e6ccb63541d594fdad858777acde41062c39f1` |
| `stage1_10s_model_52000.pt` | 52000 | 130 / 346 | `8b4a552a94567938172def991cbe65456ca301bd691b2cb8c2f53c658e0a6e29` |
| `stage1_10s_model_52006.pt` | 52006 | 130 / 346 | `325cb839bce9491d5a339441e396423fa81e8a15d75dd1fa7d55b6340bd0e372` |
| `stage2_20s_model_51000_after_1_update.pt` | 51000 | 130 / 346 | `5315f3f84ce87b8b8cf92413e57cfd4bca316c66a6e67c87e61a0a5b5fd193a4` |
| `stage2_20s_model_51500.pt` | 51500 | 130 / 346 | `71a0056cd50f05cf76ff9faaa8720fe6ac1ddf4dacc0461d0025d1c1a8872e5f` |
| `stage2_20s_model_52000.pt` | 52000 | 130 / 346 | `cb67990abb75c4e0386937915ac40cc6c23622dab4e8fc4d77a970b7d51fae6e` |
| `stage2_20s_model_52499.pt` | 52499 | 130 / 346 | `47ae4b9b12cf122104a3548779cd8d9a21ccdb1b7d5a06af0d863e75dfd8ace5` |
| `stage3_full39p72_model_52500_after_2_updates.pt` | 52500 | 130 / 346 | `b16623b8b4813bf8e4e05568e76b71bceb7052c45f2ca751e5fdc2f213bcd684` |
| `stage3_full39p72_model_52998.pt` | 52998 | 130 / 346 | `194ed7aa9cb83bfe432fdf48367550a6204896cf0cd92d4371eb6598f3cbf88f` |
| `FINAL_dance1_long_horizon_policy.pt` | 52499* | 130 / 346 | `09bc93df418905b8e5105eea6d8653b05d7963183adb785ab3cd43f33b493928` |

The B checkpoint was identified unambiguously as the second 130-dimensional
iteration-50507 model. The A hash was already recorded by the completed
training analysis, leaving the `760b…` object as the 10-second B branch.
`validate_rsl_checkpoint.py` passed every tensor, shape, optimizer, normalizer,
and iteration check after recovery.

Hard links under `logs/rsl_rl/dex_evt_flat/` restore the run layout expected by
the RSL-RL playback and resume helpers. The files in this directory are the
durable copies.

The `stage1_10s_model_*.pt` entries are periodic snapshots from the
joint-fidelity fine-tuning stage. Each was copied into this durable directory
as soon as the trainer emitted it and then passed the checkpoint validator.
`model_52006.pt` is the final snapshot after all 1,500 additional updates.

The Stage 2 `model_51000` file is intentionally named
`stage2_20s_model_51000_after_1_update.pt`: RSL-RL saved it after the first
20-second-horizon update, so it is not byte-identical to the Stage 1
`model_51000` warm start. The longer-horizon candidate snapshots use the
`stage2_20s_` prefix to keep those lineages distinct.

Stage 2 contains 1,500 contiguous updates (`51000` through `52499`,
inclusive). Its final snapshot is `stage2_20s_model_52499.pt`; `52500` is the
exclusive end of the iteration range and is not a checkpoint name.

Stage 3 adds 500 full-motion (`39.72 s`) updates from Stage 2 iteration 52499.
Its final snapshot is `stage3_full39p72_model_52998.pt`.

`FINAL_dance1_long_horizon_policy.pt` is the selected inference policy. It is a
hard link to `interpolated_candidates/model_interp_52000_52499_a025.pt` and was
created with:

```text
final = 0.75 * stage2 model_52000 + 0.25 * stage2 model_52499
```

The two source checkpoints have bit-identical observation normalizers and
compatible model tensors. The final policy passed all strict full-motion
checks in the nominal evaluation and three randomized robustness evaluations
(`4/4`). The `52499*` iteration is metadata inherited from the B endpoint, not
an additional training update. The checkpoint retains the B optimizer state
for file-format completeness, but that state is not consistent with the
interpolated weights; use the final file for inference and use one of the two
Stage 2 endpoints for any future resumed training.

The accompanying Stage 1 records are:

| File | SHA-256 |
|---|---|
| `stage1_10s_additional1500_training.txt` | `845f9c3ddb1248375ce78c5a1ddb144686d71356ecfca6b76d1e2d431edbbd2f` |
| `events.out.tfevents.1785314055.eai-pro.158355.0` | `e698944bfc18fc24dbc675d686a0cde773bfe38b40318df5f3a5e8b971f9b31e` |
| `stage2_20s_additional1500_training.txt` | `c06f300ca6fc1c81ce34026d43966e0daa0c9fb2021ed3cd8dd1c7b3e3ba1c9c` |
| `events.out.tfevents.1785316164.eai-pro.165721.0` | `73644c559fbeadcbe5a52cb6d00e12fbbe59185889a1552e032070a684f22b2c` |
| `stage3_full39p72_additional500_training.txt` | `f6f7c004debd6f78727437a873b12d5d594265bf9ae99205153a78a17af0d582` |
| `events.out.tfevents.1785318386.eai-pro.174402.0` | `ebfce337875a9ab1756b5531fd05f9a615b29df6a0c670aea6acd908da31406b` |
