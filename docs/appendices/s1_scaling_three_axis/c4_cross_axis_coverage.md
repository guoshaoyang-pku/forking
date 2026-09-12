# Appendix C.4 · Three-axis cross-table

> Generated from tracked CSVs. This is a coverage and endpoint table; it does not invent a three-way fit.

## Measured axis families

| axis | rows | branch/family | step | source |
|---|---:|---|---:|---|
| table-size | 62 | bigram, trigram | ['1000'] | s1_table_size_points.csv |
| dose (128×) | 12 | input | ['2000'] | s1_dose_points_128x.csv |
| epoch length | 12 | multiple | ['1011', '126', '1263', '1518', '168', '1770', '2022', '252', '336', '504', '672', '759'] | s1_epoch_length_points.csv |

## Table-size endpoint range

| branch | R range | n | gap min | gap max |
|---|---:|---:|---:|---:|
| bigram | 1–2,347,000 | 31 | -0.0062 | 1.1915 |
| trigram | 1–2,347,000 | 31 | -0.0317 | 3.6166 |

## Dose endpoints (128× input)

| run_id | dose | final gap | source |
|---|---:|---:|---|
| nglab0_25x_input_v5_128x_freq10 | 0.25 | 10.894863903522491 | data/runs_fixed/nglab0_25x_input_v5_128x_freq10_fixed |
| nglab0_5x_input_v5_128x_freq10 | 0.5 | 9.160114586353302 | data/runs_fixed/nglab0_5x_input_v5_128x_freq10_fixed |
| nglab0_75x_input_v5_128x_freq10 | 0.75 | 7.207066774368286 | data/runs_fixed/nglab0_75x_input_v5_128x_freq10_fixed |
| nglab1x_input_v5_128x_freq10 | 1.0 | 5.671686172485352 | data/runs_fixed/nglab1x_input_v5_128x_freq10_fixed |
| nglab1_5x_input_v5_128x_freq10 | 1.5 | 3.651792883872986 | data/runs_fixed/nglab1_5x_input_v5_128x_freq10_fixed |
| nglab2x_input_v5_128x_freq10 | 2.0 | 2.305497169494629 | data/runs_fixed/nglab2x_input_v5_128x_freq10_fixed |
| nglab2_5x_input_v5_128x_freq10 | 2.5 | 1.8373233079910278 | data/runs_fixed/nglab2_5x_input_v5_128x_freq10_fixed |
| nglab3x_input_v5_128x_freq10 | 3.0 | 0.8344939351081848 | data/runs_fixed/nglab3x_input_v5_128x_freq10_fixed |
| nglab4x_input_v5_128x_freq10 | 4.0 | 0.6398343443870544 | data/runs_fixed/nglab4x_input_v5_128x_freq10_fixed |
| nglab5x_input_v5_128x_freq10 | 5.0 | 0.3552113175392151 | data/runs_fixed/nglab5x_input_v5_128x_freq10_fixed |
| nglab6x_input_v5_128x_freq10 | 6.0 | -0.0870022177696228 | data/runs_fixed/nglab6x_input_v5_128x_freq10_fixed |
| nglab8x_input_v5_128x_freq10 | 8.0 | -0.054738402366638184 | data/runs_fixed/nglab8x_input_v5_128x_freq10_fixed |

## Epoch-length endpoints

| run_id | multiplier | target steps | final gap | source |
|---|---:|---:|---:|---|
| s1v5_128_ep_tri_0p125xL4_3ep | 0.125 | 126 | 3.5524487495422363 | data/runs_scaling/s1v5_128_ep_tri_0p125xL4_3ep_fixed |
| s1v5_128_ep_tri_0p1667xL4_3ep | 0.1667 | 168 | 3.575384259223938 | data/runs_scaling/s1v5_128_ep_tri_0p1667xL4_3ep_fixed |
| s1v5_128_ep_tri_0p25xL4_3ep | 0.25 | 252 | 3.4537723064422607 | data/runs_scaling/s1v5_128_ep_tri_0p25xL4_3ep_fixed |
| s1v5_128_ep_tri_0p3333xL4_3ep | 0.3333 | 336 | 3.101970672607422 | data/runs_scaling/s1v5_128_ep_tri_0p3333xL4_3ep_fixed |
| s1v5_128_ep_tri_0p5xL4_3ep | 0.5 | 504 | 3.0099236965179443 | data/runs_scaling/s1v5_128_ep_tri_0p5xL4_3ep_fixed |
| s1v5_128_ep_tri_0p6667xL4_3ep | 0.6667 | 672 | 2.809251308441162 | data/runs_scaling/s1v5_128_ep_tri_0p6667xL4_3ep_fixed |
| s1v5_128_ep_tri_0p75xL4_3ep | 0.75 | 759 | 2.725416421890259 | data/runs_scaling/s1v5_128_ep_tri_0p75xL4_3ep_fixed |
| s1v5_128_ep_tri_1p0xL4_3ep | 1.0 | 1011 | 2.4688557386398315 | data/runs_scaling/s1v5_128_ep_tri_1p0xL4_3ep_fixed |
| s1v5_128_ep_tri_1p25xL4_3ep | 1.25 | 1263 | 3.5897722244262695 | data/runs_scaling/s1v5_128_ep_tri_1p25xL4_3ep_fixed |
| s1v5_128_ep_tri_1p5xL4_3ep | 1.5 | 1518 | 4.434708476066589 | data/runs_scaling/s1v5_128_ep_tri_1p5xL4_3ep_fixed |
| s1v5_128_ep_tri_1p75xL4_3ep | 1.75 | 1770 | 5.0426143407821655 | data/runs_scaling/s1v5_128_ep_tri_1p75xL4_3ep_fixed |
| s1v5_128_ep_tri_2p0xL4_3ep | 2.0 | 2022 | 5.5823822021484375 | data/runs_scaling/s1v5_128_ep_tri_2p0xL4_3ep_fixed |

## Identification boundary

- These files provide separate table-size, dose, and epoch-length families; they do not provide a fully crossed table-size × dose × epoch design.
- Fits below are one-axis fits. Empty cells mean unmeasured, not zero; a crossed run needs a new registration.

| family | branch | model | n | slope | R² |
|---|---|---|---:|---:|---:|
| table_size | bigram |  | 18 | 0.42900854694068896 | 0.976185314489623 |
| table_size | trigram |  | 18 | 0.657470203345452 | 0.9951085946177981 |
| epoch_length | trigram-only | quadratic gap vs ln(multiplier) | 12 |  | 0.7741549068017504 |
| dose | input | positive gap; dose <= 5x only | 10 | -1.7269830361182248 | 0.8872251449478308 |
| frequency_exact | bigram | token-mass-weighted geometric-bin fit; positive gap only | 7 | -0.25274555683860056 | 0.9971648881056272 |
| frequency_exact | trigram | token-mass-weighted geometric-bin fit; positive gap only | 7 | -0.3181214825388326 | 0.9955480943221907 |
