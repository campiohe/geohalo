# geohalo benchmarks

Environment: Python 3.14.4 on Linux x86_64, geohalo @ 8c87ee6, scipy 1.17.1, numpy 2.4.6, shapely 2.1.2, xarray 2026.4.0, exactextract 0.3.0, geopandas 1.1.3, cfgrib 0.9.15.1.
Hardware: AMD Ryzen 5 7600X 6-Core Processor (6C/12T), 15.5 GB RAM.
ECMWF: IFS ENS 0.25 degree, cycle 20260529 00z, global 721x1440 grid.
Polygons: Brazil (country)=1, Americas (countries)=35, Americas & Europe (countries)=79, Brazil (municipalities)=5572, United States (counties)=3148.
Cache: LocalCache @ /tmp/claude-1000/geohalo-bench-olhfoi5p (fresh temp dir per run; RedisCache not measured).
Masked path: NaN mask = 0.3 of cells (seed 0), synthetic path-coverage.
Timing: hot/hit rows 2 warmup + 7 iters; miss rows 1 warmup + 3 iters (global 0.05 degree resample builds: 1 timed build). Median (p10 - p90). miss = build + serialize + store; hit = load + deserialize; speedup = miss/hit. Hot rows operate on ascending-latitude data. dRSS = peak current-RSS over baseline.

## ECMWF inputs (whole world)

| batch               | shape         | cells       | in-mem   |
| ------------------- | ------------- | ----------- | -------- |
| (member=1, step=1)  | 721x1440      | 1,038,240   | 4.0 MB   |
| (member=50, step=1) | 50x721x1440   | 51,912,000  | 198.0 MB |
| (member=50, step=4) | 50x4x721x1440 | 207,648,000 | 792.1 MB |

## Reduce - precompute (cache miss vs hit)

| region                        | artifact       | n_polys | target   | iters | miss (build+store)        | hit (load+deser)          | speedup | CSR    | blob   | dRSS      |
| ----------------------------- | -------------- | ------- | -------- | ----- | ------------------------- | ------------------------- | ------- | ------ | ------ | --------- |
| Brazil (country)              | Stencil        | 1       | native   | n/a   | 40 ms  (40 ms - 43 ms)    | 0.9 ms  (0.9 ms - 1.2 ms) | 43.7x   | 138 KB | 156 KB | 7.7 MB    |
| Brazil (country)              | ReduceOperator | 1       | native   | n/a   | 0.2 ms  (0.2 ms - 0.4 ms) | 0.1 ms  (0.1 ms - 0.2 ms) | 2.1x    | 138 KB | 156 KB | 4 KB      |
| Americas (countries)          | Stencil        | 35      | native   | n/a   | 1.88 s  (1.84 s - 1.88 s) | 41 ms  (40 ms - 42 ms)    | 46.0x   | 959 KB | 977 KB | 93.1 MB   |
| Americas (countries)          | ReduceOperator | 35      | native   | n/a   | 0.7 ms  (0.5 ms - 0.8 ms) | 0.2 ms  (0.1 ms - 0.3 ms) | 3.8x    | 959 KB | 977 KB | 4 KB      |
| Americas & Europe (countries) | Stencil        | 79      | native   | n/a   | 2.89 s  (2.88 s - 2.92 s) | 64 ms  (62 ms - 65 ms)    | 45.3x   | 1.7 MB | 1.7 MB | 130.5 MB  |
| Americas & Europe (countries) | ReduceOperator | 79      | native   | n/a   | 1.0 ms  (0.9 ms - 1.1 ms) | 0.2 ms  (0.2 ms - 0.4 ms) | 4.6x    | 1.7 MB | 1.7 MB | 4 KB      |
| Brazil (municipalities)       | Stencil        | 5572    | 0.05 deg | n/a   | 1.89 s  (1.89 s - 1.89 s) | 63 ms  (60 ms - 88 ms)    | 30.2x   | 4.5 MB | 4.7 MB | 198.2 MB  |
| Brazil (municipalities)       | ReduceOperator | 5572    | 0.05 deg | 1     | 4.40 s  (4.40 s - 4.40 s) | 0.6 ms  (0.6 ms - 0.7 ms) | 7100x   | 1.3 MB | 1.5 MB | 3262.1 MB |
| Brazil (municipalities)       | ReduceOperator | 5572    | 0.05 deg | 3     | 4.86 s  (4.86 s - 4.86 s) | 1.1 ms  (0.9 ms - 1.5 ms) | 4603x   | 4.7 MB | 4.9 MB | 3262.0 MB |
| United States (counties)      | Stencil        | 3148    | 0.05 deg | n/a   | 1.11 s  (1.11 s - 1.11 s) | 38 ms  (34 ms - 44 ms)    | 29.3x   | 6.2 MB | 6.3 MB | 197.7 MB  |
| United States (counties)      | ReduceOperator | 3148    | 0.05 deg | 1     | 3.86 s  (3.86 s - 3.86 s) | 0.5 ms  (0.4 ms - 0.6 ms) | 8123x   | 1.1 MB | 1.2 MB | 3262.0 MB |
| United States (counties)      | ReduceOperator | 3148    | 0.05 deg | 3     | 4.92 s  (4.92 s - 4.92 s) | 0.6 ms  (0.5 ms - 0.8 ms) | 8691x   | 3.2 MB | 3.4 MB | 3262.0 MB |

## Reduce - hot (apply)

| region                        | n_polys | iters | batch               | path   | median  (p10 - p90)       | dRSS      |
| ----------------------------- | ------- | ----- | ------------------- | ------ | ------------------------- | --------- |
| Brazil (country)              | 1       | n/a   | (member=1, step=1)  | clean  | 0.8 ms  (0.6 ms - 1.8 ms) | 4 KB      |
| Brazil (country)              | 1       | n/a   | (member=50, step=1) | clean  | 367 ms  (352 ms - 515 ms) | 791.0 MB  |
| Brazil (country)              | 1       | n/a   | (member=50, step=4) | clean  | 1.64 s  (1.48 s - 1.67 s) | 3168.4 MB |
| Brazil (country)              | 1       | n/a   | (member=1, step=1)  | masked | 4.2 ms  (3.9 ms - 4.4 ms) | 4 KB      |
| Brazil (country)              | 1       | n/a   | (member=50, step=1) | masked | 970 ms  (958 ms - 1.07 s) | 1038.3 MB |
| Brazil (country)              | 1       | n/a   | (member=50, step=4) | masked | 4.02 s  (3.97 s - 4.05 s) | 4158.6 MB |
| Americas (countries)          | 35      | n/a   | (member=1, step=1)  | clean  | 0.9 ms  (0.7 ms - 1.3 ms) | 4 KB      |
| Americas (countries)          | 35      | n/a   | (member=50, step=1) | clean  | 409 ms  (367 ms - 451 ms) | 792.1 MB  |
| Americas (countries)          | 35      | n/a   | (member=50, step=4) | clean  | 1.66 s  (1.52 s - 1.86 s) | 3168.5 MB |
| Americas (countries)          | 35      | n/a   | (member=1, step=1)  | masked | 3.5 ms  (3.3 ms - 3.8 ms) | 4 KB      |
| Americas (countries)          | 35      | n/a   | (member=50, step=1) | masked | 1.03 s  (977 ms - 1.23 s) | 1039.7 MB |
| Americas (countries)          | 35      | n/a   | (member=50, step=4) | masked | 4.07 s  (3.78 s - 4.94 s) | 4158.6 MB |
| Americas & Europe (countries) | 79      | n/a   | (member=1, step=1)  | clean  | 1.0 ms  (0.6 ms - 1.2 ms) | 4 KB      |
| Americas & Europe (countries) | 79      | n/a   | (member=50, step=1) | clean  | 391 ms  (372 ms - 436 ms) | 792.1 MB  |
| Americas & Europe (countries) | 79      | n/a   | (member=50, step=4) | clean  | 1.77 s  (1.50 s - 2.03 s) | 3168.5 MB |
| Americas & Europe (countries) | 79      | n/a   | (member=1, step=1)  | masked | 2.9 ms  (2.9 ms - 3.1 ms) | 4 KB      |
| Americas & Europe (countries) | 79      | n/a   | (member=50, step=1) | masked | 988 ms  (961 ms - 1.03 s) | 1039.7 MB |
| Americas & Europe (countries) | 79      | n/a   | (member=50, step=4) | masked | 4.08 s  (4.01 s - 4.19 s) | 4158.6 MB |
| Brazil (municipalities)       | 5572    | 1     | (member=1, step=1)  | clean  | 1.1 ms  (0.9 ms - 1.6 ms) | 4 KB      |
| Brazil (municipalities)       | 5572    | 1     | (member=50, step=1) | clean  | 393 ms  (381 ms - 401 ms) | 792.1 MB  |
| Brazil (municipalities)       | 5572    | 1     | (member=50, step=4) | clean  | 1.71 s  (1.46 s - 1.75 s) | 3168.5 MB |
| Brazil (municipalities)       | 5572    | 3     | (member=1, step=1)  | clean  | 1.0 ms  (0.9 ms - 1.7 ms) | 4 KB      |
| Brazil (municipalities)       | 5572    | 3     | (member=50, step=1) | clean  | 388 ms  (383 ms - 423 ms) | 792.1 MB  |
| Brazil (municipalities)       | 5572    | 3     | (member=50, step=4) | clean  | 1.76 s  (1.47 s - 1.84 s) | 3168.5 MB |
| United States (counties)      | 3148    | 1     | (member=1, step=1)  | clean  | 0.8 ms  (0.7 ms - 0.9 ms) | 4 KB      |
| United States (counties)      | 3148    | 1     | (member=50, step=1) | clean  | 386 ms  (368 ms - 418 ms) | 792.1 MB  |
| United States (counties)      | 3148    | 1     | (member=50, step=4) | clean  | 1.68 s  (1.44 s - 1.72 s) | 3168.5 MB |
| United States (counties)      | 3148    | 3     | (member=1, step=1)  | clean  | 0.9 ms  (0.8 ms - 1.7 ms) | 4 KB      |
| United States (counties)      | 3148    | 3     | (member=50, step=1) | clean  | 369 ms  (362 ms - 427 ms) | 792.1 MB  |
| United States (counties)      | 3148    | 3     | (member=50, step=4) | clean  | 1.69 s  (1.46 s - 1.76 s) | 3168.5 MB |

## Bias tree (cache miss/hit + hot)

| hierarchy                | n_leaves | depth | miss                      | hit                       | speedup | rollup CSR | blob   | apply  (p10 - p90)        | dRSS |
| ------------------------ | -------- | ----- | ------------------------- | ------------------------- | ------- | ---------- | ------ | ------------------------- | ---- |
| muni -> state            | 5572     | 2     | 1.56 s  (1.50 s - 1.61 s) | 5.2 ms  (5.0 ms - 6.1 ms) | 298x    | 152 KB     | 301 KB | 3.2 ms  (2.9 ms - 3.9 ms) | 4 KB |
| muni -> state -> country | 5572     | 3     | 1.61 s  (1.60 s - 1.63 s) | 5.3 ms  (5.2 ms - 5.6 ms) | 306x    | 218 KB     | 366 KB | 3.1 ms  (2.9 ms - 3.6 ms) | 4 KB |

