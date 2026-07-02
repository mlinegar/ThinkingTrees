# Pearson r by manifesto length bucket

Buckets: <20K / 20-50K / 50-100K / >100K chars. Tests whether the tree + summarization earn their keep on long manifestos (where flat-baseline truncation loses the tail).

## Per-dimension runs

|Source|<20K|20-50K|50-100K|>100K|all|
|---|---:|---:|---:|---:|---:|
|tree-economic|+0.89 (n=18)|+0.90 (n=49)|+0.89 (n=33)|+0.89 (n=118)|+0.89 (n=218)|
|tree-social|+0.83 (n=17)|+0.83 (n=49)|+0.81 (n=33)|+0.85 (n=118)|+0.84 (n=217)|
|tree-immigration|+0.94 (n=14)|+0.86 (n=35)|+0.82 (n=23)|+0.87 (n=91)|+0.87 (n=163)|
|tree-eu|+0.86 (n=13)|+0.92 (n=43)|+0.90 (n=29)|+0.88 (n=95)|+0.90 (n=180)|
|tree-environment|+0.93 (n=14)|+0.80 (n=44)|+0.76 (n=23)|+0.83 (n=103)|+0.81 (n=184)|
|tree-decentralization|+0.69 (n=17)|+0.49 (n=47)|+0.49 (n=33)|+0.47 (n=118)|+0.46 (n=215)|
|flat-economic|+0.88 (n=12)|+0.95 (n=19)|+0.96 (n=8)|+0.95 (n=11)|+0.93 (n=50)|
|flat-social|+0.79 (n=12)|+0.88 (n=18)|+0.91 (n=8)|+0.82 (n=10)|+0.85 (n=48)|
|flat-immigration|+0.98 (n=7)|+0.76 (n=17)|+0.85 (n=7)|+0.99 (n=4)|+0.89 (n=35)|
|flat-eu|+0.73 (n=9)|+0.89 (n=17)|+0.96 (n=5)|— (n=3)|+0.88 (n=34)|
|flat-environment|+0.71 (n=8)|+0.87 (n=19)|+0.94 (n=6)|+0.86 (n=13)|+0.83 (n=46)|
|flat-decentralization|+0.74 (n=6)|+0.66 (n=13)|-0.09 (n=4)|+0.47 (n=7)|+0.54 (n=30)|
|concat-economic|+0.90 (n=12)|+0.94 (n=19)|+0.94 (n=8)|+0.94 (n=11)|+0.93 (n=50)|
|concat-social|+0.80 (n=11)|+0.85 (n=19)|+0.93 (n=8)|+0.77 (n=11)|+0.82 (n=49)|
|concat-immigration|+0.94 (n=9)|+0.84 (n=22)|+0.94 (n=8)|+0.93 (n=6)|+0.91 (n=45)|
|concat-eu|+0.82 (n=10)|+0.93 (n=22)|+0.96 (n=7)|+0.90 (n=7)|+0.91 (n=46)|
|concat-environment|+0.79 (n=9)|+0.91 (n=21)|+0.97 (n=6)|+0.89 (n=13)|+0.87 (n=49)|
|concat-decentralization|+0.73 (n=10)|+0.56 (n=19)|+0.09 (n=8)|+0.74 (n=11)|+0.52 (n=48)|

## Combined pipeline (per-dim cells)

|Source|<20K|20-50K|50-100K|>100K|all|
|---|---:|---:|---:|---:|---:|
|combined-economic|+0.85 (n=23)|+0.88 (n=54)|+0.90 (n=34)|+0.86 (n=118)|+0.87 (n=229)|
|combined-social|+0.87 (n=20)|+0.84 (n=52)|+0.86 (n=34)|+0.85 (n=115)|+0.85 (n=221)|
|combined-immigration|+0.97 (n=15)|+0.87 (n=37)|+0.92 (n=23)|+0.86 (n=87)|+0.88 (n=162)|
|combined-eu|+0.87 (n=14)|+0.92 (n=50)|+0.82 (n=30)|+0.91 (n=92)|+0.90 (n=186)|
|combined-environment|+0.84 (n=18)|+0.77 (n=47)|+0.68 (n=24)|+0.86 (n=100)|+0.81 (n=189)|
|combined-decentralization|+0.40 (n=15)|+0.41 (n=46)|+0.46 (n=32)|+0.44 (n=115)|+0.40 (n=208)|

