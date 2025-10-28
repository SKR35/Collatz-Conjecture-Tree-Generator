# Collatz-Conjecture-Tree-Generator

Interactive visualizer that draws thousands of Collatz trajectories as flowing paths and maps their traversal density to color/brightness. Designed for making nice-looking stills and interactively exploring mathematical concepts.

## Features

- Paths slider: number of start values (up to 10,000 + 2,000 default).

- Random starts: sample random seeds in range for organic variety.

- Turn° slider: per-step angle + even/odd steps turn opposite ways.

- Decay slider: decreases step length along the path (controls “curl”).

- Step slider: base step length (overall scale).

- Colormap menu: matplotlib maps (viridis, magma, cividis, turbo, plasma…) + custom sets (SoftSunset, Seashore, etc.).

- Gamma slider: non-linear brightness for glow/lightning effects.

- Animate toggle: draw paths progressively (off = fastest full render).

- Save PNG: exports the current canvas at window resolution.

- Resizable UI: maximize the window and the canvas grows with it.

- Status bar: render time and current settings.

## Create the environment

conda env create -f collatz-environment.yml
conda activate collatz-tree

## Run the viewer

python collatz_tk.py