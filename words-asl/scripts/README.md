# scripts/

One-off utilities — data prep, sweeps, plotting, analysis. Each script is runnable as
`python scripts/<name>.py --help` and writes its outputs into `docs/` or
`checkpoints/` rather than the repo root.

Convention: scripts here may import from `data`, `models`, etc. Run
from the repository root so relative imports resolve.
