# Phase 3 combined pipeline × chunk_chars (r per dimension)

One shared summarizer (JOINT_RUBRIC) + one scorer, all 6 dims from one summary. Benoit Fig 1 reference in the last row for context.

|chunk_chars|Economic|Social|Immigration|Eu|Environment|Decentralizat|Macro|
|---|---:|---:|---:|---:|---:|---:|---:|
|64,000|+0.910 (n=50)|+0.867 (n=50)|+0.897 (n=37)|+0.910 (n=37)|+0.795 (n=41)|+0.458 (n=41)|+0.806|
|32,000|+0.917 (n=50)|+0.849 (n=49)|+0.908 (n=37)|+0.889 (n=39)|+0.749 (n=40)|+0.438 (n=38)|+0.792|
|16,000|+0.919 (n=50)|+0.851 (n=49)|+0.916 (n=36)|+0.889 (n=39)|+0.808 (n=42)|+0.441 (n=43)|+0.804|
|8,000|+0.927 (n=49)|+0.874 (n=47)|+0.911 (n=37)|+0.917 (n=37)|+0.801 (n=42)|+0.413 (n=42)|+0.807|
|**Benoit Fig 1**|0.87|0.92|0.89|0.91|0.82|0.49|—|
