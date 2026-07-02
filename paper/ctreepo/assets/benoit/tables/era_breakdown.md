# Pearson r by election era (per-dim full pipeline at chunk=24K)

Splits the 215-mfesto test set by election-year era. Benoit's data goes back to 1989; this checks whether scoring quality holds up as political rhetoric evolved.

|Dimension|1989-1999|2000-2009|2010-2019|all|
|---|---:|---:|---:|---:|
|economic|+0.899 (n=24)|+0.859 (n=60)|+0.905 (n=134)|+0.892 (n=218)|
|social|+0.847 (n=24)|+0.791 (n=59)|+0.865 (n=134)|+0.840 (n=217)|
|immigration|— (n=0)|+0.801 (n=45)|+0.889 (n=118)|+0.867 (n=163)|
|eu|— (n=0)|+0.862 (n=49)|+0.912 (n=131)|+0.896 (n=180)|
|environment|+0.802 (n=24)|+0.759 (n=27)|+0.830 (n=133)|+0.814 (n=184)|
|decentralization|+0.393 (n=24)|+0.441 (n=59)|+0.497 (n=132)|+0.464 (n=215)|
