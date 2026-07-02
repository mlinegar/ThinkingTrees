# Outside Code Reference

Recorded on 2026-05-19 from `/home/mlinegar/ThinkingTrees`.

This document tracks outside method implementations that are available locally
for C-TreePO / `treepo` design work, parity tests, speed comparisons, and
optional adapters. The local source clones are in
`outside_data/method_reference_repos/`, which is intentionally ignored by git.
Do not treat these clones as vendored project source unless we make an explicit
license and maintenance decision.

## Readiness Summary

The external code is ready to use as reference implementations, black-box test
oracles, and optional local adapters. It is not yet ready to become hard runtime
dependencies of C-TreePO or to be copied into the repository.

Main reasons:

- Several key implementations are GPL-licensed (`grf`, `ranger`, `dsl-r`), so
  they should stay behind a subprocess, optional adapter, or benchmark boundary
  unless we deliberately accept GPL obligations for derived work.
- We ran focused upstream tests for the relevant surfaces, not full CI for every
  package.
- A few upstream caveats remain: `dsl-r` has no test suite and fails
  `R CMD check` on metadata, `ranger` has one flaky C++ statistical test, and
  the Python environment has unrelated existing `pip check` conflicts.

## Local Inventory

| Package | Local directory | Upstream | Commit | Branch | Installed version | License | Role for C-TreePO |
|---|---|---|---:|---|---|---|---|
| GRF | `outside_data/method_reference_repos/grf` | `https://github.com/grf-labs/grf.git` | `5bee99b` | `master` | R `grf 2.6.1` | GPL-3 | Primary causal forest / generalized random forest reference. |
| EconML | `outside_data/method_reference_repos/econml` | `https://github.com/py-why/EconML.git` | `06ae02a` | `main` | Python `econml 0.16.0` | MIT/BSD notices | Python causal forest DML and GRF-style APIs. |
| CausalML | `outside_data/method_reference_repos/causalml` | `https://github.com/uber/causalml.git` | `327200f` | `master` | Python `causalml 0.16.0` | Apache-2.0 | Python causal tree/forest and uplift-model baselines. |
| DSL-R | `outside_data/method_reference_repos/dsl-r` | `https://github.com/naoki-egami/dsl.git` | `537664a` | `master` | R `dsl 0.1.0` | GPL-2 | Canonical Design-based Supervised Learning implementation. |
| DSL Python | `outside_data/method_reference_repos/dsl-python` | `https://github.com/Enan456/dsl-python.git` | `ddf7406` | `main` | Python `dsl_kit 0.2.0` | MIT | Python DSL port with tests; useful for direct Python integration. |
| ranger | `outside_data/method_reference_repos/ranger` | `https://github.com/imbs-hl/ranger.git` | `dafa5db` | `master` | R `ranger 0.18.0` | GPL-3 | Fast random forest implementation and performance/design reference. |

## Environment State

Python packages installed in `./venv`:

- `econml 0.16.0`
- `causalml 0.16.0`
- `dsl_kit 0.2.0`
- `forestci 0.6`
- `lightgbm 4.6.0`
- `xgboost 3.2.0`
- `shap 0.51.0`
- `statsmodels 0.14.6`
- `duecredit 0.11.2`
- `cmake 4.3.2`

R packages installed in the user R library:

- `grf 2.6.1`
- `ranger 0.18.0`
- `dsl 0.1.0`
- `testthat 3.3.2`
- `RcppEigen 0.3.4.0.2`
- `estimatr 1.0.6`
- `SuperLearner 2.0.40`
- `arm 1.15.3`
- `nloptr 2.2.1`
- `lme4 2.0.1`

Known existing Python environment conflicts from `pip check`:

- `sbijax 0.3.6` requires `jax==0.8.1`, but the venv has `jax 0.10.0`.
- `sbijax 0.3.6` requires `jaxlib==0.8.1`, but the venv has
  `jaxlib 0.10.0`.
- `litellm 1.80.9` requires `grpcio<1.68.0,>=1.62.3`, but the venv has
  `grpcio 1.80.0`.

These conflicts predate the outside-code setup and were not changed.

## Installation Commands

Python editable installs:

```bash
source venv/bin/activate

python -m pip install -e outside_data/method_reference_repos/dsl-python
python -m pip install -e outside_data/method_reference_repos/econml
python -m pip install -e outside_data/method_reference_repos/causalml

# Optional/support packages used by the external test surfaces.
python -m pip install duecredit cmake
```

R source installs:

```bash
source venv/bin/activate
export PATH="/home/mlinegar/ThinkingTrees/venv/bin:$PATH"

Rscript -e 'install.packages(c(
  "RcppEigen", "DiceKriging", "lmtest", "estimatr", "SuperLearner",
  "arm", "matrixcalc", "nloptr", "lme4", "testthat", "pkgload"
), repos = "https://cloud.r-project.org")'

R CMD INSTALL outside_data/method_reference_repos/ranger
R CMD INSTALL outside_data/method_reference_repos/grf/r-package/grf
R CMD INSTALL outside_data/method_reference_repos/dsl-r
```

No project requirements or `pyproject.toml` files were updated. These packages
are currently local development dependencies.

## Tests Run

| Package | Command | Result | Notes |
|---|---|---|---|
| DSL Python | `python -m pytest -q tests` from `outside_data/method_reference_repos/dsl-python` | `60 passed, 3 skipped` | Full upstream Python test suite. |
| EconML | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m unittest -v econml.tests.test_grf_python econml.tests.test_grf_cython` from `outside_data/method_reference_repos/econml` | `19 tests OK` | Focused GRF/Causal Forest tests. Full EconML suite is much larger and environment-specific. |
| CausalML | `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m pytest -q tests/test_causal_trees.py` from `outside_data/method_reference_repos/causalml` | `20 passed, 2 warnings` | Focused causal tree/forest tests. Warnings came from `forestci` numerical overflow checks. |
| ranger R | `Rscript -e 'library(testthat); pkgload::load_all(".", quiet=TRUE); testthat::test_dir("tests/testthat", reporter="summary")'` from `outside_data/method_reference_repos/ranger` | Passed, `10 skipped` | Source test mode is needed so internal helpers are visible. Skips are missing optional `GenABEL` and stochastic preconditions. |
| GRF R | `Rscript -e 'library(testthat); pkgload::load_all(".", quiet=TRUE); options(warn=2); testthat::test_dir("tests/testthat", reporter="summary")'` from `outside_data/method_reference_repos/grf/r-package/grf` | Passed, one empty-test skip | R package tests for GRF passed with warnings promoted to errors. |
| DSL-R | Functional `Rscript` smoke using `data_lm` and `dsl(...)` | Passed | Upstream has no `tests/` directory. See command below. |
| GRF C++ core | `cmake -S . -B build && cmake --build build -j4 && ./build/grf` from `outside_data/method_reference_repos/grf/core` | `All tests passed (6541 assertions in 96 test cases)` | Build emitted compiler/CMake warnings but no failures. |
| ranger C++ | `cmake -S . -B build -DCMAKE_POLICY_VERSION_MINIMUM=3.5 && cmake --build build -j4 && ./build/runUnitTests` from `outside_data/method_reference_repos/ranger/cpp_version/test` | `75 passed, 1 failed` on full run; excluded run passed | The failing test is `drawWithoutReplacementSkip.small_large1`, a stochastic uniformity check seeded by `std::random_device`. It passed when run alone and failed on repeated full runs. |

DSL-R smoke command:

```bash
Rscript - <<'RS'
suppressPackageStartupMessages(library(dsl))
data(data_lm)
out <- dsl(
  model = "lm",
  formula = Y ~ X1 + X2 + X3 + X4 + X5,
  predicted_var = "Y",
  prediction = "pred_Y",
  data = data_lm,
  cross_fit = 2,
  sample_split = 2,
  seed = 1234
)
stopifnot(inherits(out, "dsl"))
stopifnot(all(is.finite(out$coefficients)))
cat("dsl-r smoke coefficients", length(out$coefficients), "\n")
RS
```

ranger C++ stable command excluding the flaky statistical test:

```bash
cd outside_data/method_reference_repos/ranger/cpp_version/test
./build/runUnitTests --gtest_filter=-drawWithoutReplacementSkip.small_large1
```

Result: `75` tests passed.

## Basic Usage

### EconML Causal Forest

```python
import numpy as np
from econml.dml import CausalForestDML
from sklearn.ensemble import RandomForestRegressor

rng = np.random.default_rng(0)
n = 200
X = rng.normal(size=(n, 4))
T = rng.binomial(1, 0.5, size=n)
Y = X[:, 0] + T * (1.0 + X[:, 1]) + rng.normal(scale=0.1, size=n)

est = CausalForestDML(
    model_y=RandomForestRegressor(n_estimators=20, random_state=0),
    model_t=RandomForestRegressor(n_estimators=20, random_state=1),
    n_estimators=40,
    min_samples_leaf=5,
    random_state=0,
)
est.fit(Y, T, X=X)
effects = est.effect(X[:5])
```

Relevant source:

- `outside_data/method_reference_repos/econml/econml/dml/causal_forest.py`
- `outside_data/method_reference_repos/econml/econml/grf/`
- `outside_data/method_reference_repos/econml/econml/tests/test_grf_python.py`
- `outside_data/method_reference_repos/econml/econml/tests/test_grf_cython.py`

### CausalML Causal Forest

```python
import numpy as np
from causalml.inference.tree import CausalRandomForestRegressor

rng = np.random.default_rng(1)
n = 200
X = rng.normal(size=(n, 4))
treatment = rng.integers(0, 2, size=n)
y = X[:, 0] + treatment * (0.5 + X[:, 1]) + rng.normal(scale=0.1, size=n)

forest = CausalRandomForestRegressor(
    n_estimators=20,
    min_samples_leaf=5,
    random_state=1,
)
forest.fit(X, treatment, y)
tau_hat = forest.predict(X[:5])
```

Relevant source:

- `outside_data/method_reference_repos/causalml/causalml/inference/tree/`
- `outside_data/method_reference_repos/causalml/tests/test_causal_trees.py`

### DSL Python

```python
from dsl.dsl import dsl

# See upstream tests for concrete expected inputs and comparison cases:
# outside_data/method_reference_repos/dsl-python/tests/
```

Relevant source:

- `outside_data/method_reference_repos/dsl-python/dsl/dsl.py`
- `outside_data/method_reference_repos/dsl-python/tests/`

### GRF in R

```r
library(grf)

set.seed(0)
n <- 200
X <- matrix(rnorm(n * 4), n, 4)
W <- rbinom(n, 1, 0.5)
Y <- X[, 1] + W * (1 + X[, 2]) + rnorm(n, sd = 0.1)

fit <- causal_forest(X, Y, W, num.trees = 100)
tau_hat <- predict(fit, X[1:5, ])$predictions
```

Relevant source:

- `outside_data/method_reference_repos/grf/r-package/grf/R/causal_forest.R`
- `outside_data/method_reference_repos/grf/core/src/forest/`
- `outside_data/method_reference_repos/grf/core/src/tree/`
- `outside_data/method_reference_repos/grf/core/src/splitting/`
- `outside_data/method_reference_repos/grf/core/src/prediction/`

### ranger in R

```r
library(ranger)

set.seed(0)
df <- data.frame(
  y = rnorm(200),
  x1 = rnorm(200),
  x2 = rnorm(200),
  x3 = rnorm(200)
)

fit <- ranger(y ~ ., data = df, num.trees = 100, write.forest = TRUE)
pred <- predict(fit, df[1:5, ])$predictions
```

Relevant source:

- `outside_data/method_reference_repos/ranger/R/ranger.R`
- `outside_data/method_reference_repos/ranger/src/`
- `outside_data/method_reference_repos/ranger/cpp_version/src/`

### DSL-R

```r
library(dsl)

data(data_lm)
fit <- dsl(
  model = "lm",
  formula = Y ~ X1 + X2 + X3 + X4 + X5,
  predicted_var = "Y",
  prediction = "pred_Y",
  data = data_lm,
  cross_fit = 5,
  sample_split = 5,
  seed = 1234
)

fit$coefficients
```

Relevant source:

- `outside_data/method_reference_repos/dsl-r/R/dsl.R`
- `outside_data/method_reference_repos/dsl-r/R/estimate_g.R`
- `outside_data/method_reference_repos/dsl-r/R/helper_dsl_general.R`
- `outside_data/method_reference_repos/dsl-r/R/helper_moment.R`

## Integration Guidance

Use these libraries in three lanes:

1. Reference design: read the source for splitting rules, honesty, nuisance
   fitting, sampling, weighting, and API conventions.
2. Black-box tests: call installed Python/R packages from focused parity tests
   and compare C-TreePO outputs on small deterministic fixtures.
3. Optional adapters: expose external methods behind optional imports or
   subprocess boundaries, with version and commit pins recorded in run
   manifests.

Practical rules:

- Keep GPL implementations (`grf`, `ranger`, `dsl-r`) outside the core Python
  package unless the project license strategy changes.
- Do not copy GPL source or transliterate substantial implementation details
  into C-TreePO. Use the public APIs for testing and benchmarking.
- Do not add unconditional top-level imports for `econml`, `causalml`, `dsl`,
  `ranger`, or `grf` in core modules. Put them behind optional adapters or
  test-only paths.
- Prefer small, deterministic parity tests over full upstream CI.
- Record upstream commit SHA, package version, random seed, thread limits, and
  command line in any benchmark artifact.
- Use thread caps for repeatable forest tests:

```bash
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
```

## Known Caveats

- `dsl-r` installed and ran a functional smoke test, but it has no upstream
  `tests/` directory.
- `R CMD check --no-manual --no-vignettes --ignore-vignettes dsl-r` failed
  before runtime checks because the DESCRIPTION metadata reports
  `Required field missing or empty: 'Author'`.
- `ranger` R tests should be run with `pkgload::load_all()` from source. Running
  the installed package directly against source tests hides internal helpers and
  causes a test failure unrelated to package functionality.
- `ranger` C++ requires old GoogleTest 1.7 for its upstream test harness. The
  local test tree has `gtest-1.7.0` cloned under
  `outside_data/method_reference_repos/ranger/cpp_version/test/`.
- `ranger` C++ full test runs currently fail one stochastic statistical check,
  `drawWithoutReplacementSkip.small_large1`. The same test can pass when run
  alone, and the rest of the suite passes when it is excluded.
- EconML and CausalML full upstream suites were not run; only their causal
  forest / causal tree relevant tests were run.
- The Python venv has unrelated `pip check` conflicts for `sbijax`/`jax` and
  `litellm`/`grpcio`.

## Refresh Commands

Update a clone and re-record its pin:

```bash
cd outside_data/method_reference_repos/<repo>
git fetch --all --tags
git status --short --branch
git rev-parse --short HEAD
```

Regenerate the inventory table manually after any update. For benchmarkable
work, prefer pinning to a commit SHA rather than a moving branch name.
