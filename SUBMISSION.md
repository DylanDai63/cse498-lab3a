# CSE 398/498 Lab 3a submission (Hengde Dai)

This repository is the course workspace [YiyuchenHu/turtlebot3-gazebo-navigation-hw](https://github.com/YiyuchenHu/turtlebot3-gazebo-navigation-hw) (upstream commit `a048d1e`) plus my work.

| What | Where |
|---|---|
| Implemented detector (the graded task) | `src/tb3_detector/tb3_detector/detector_core.py`, `load()` and `infer()` |
| Demo video: two language commands | `videos/lab3a_demo.mp4` |
| Report (how the navigation works, results) | `report/main.pdf` (source `report/main.tex`) |
| Acceptance results used in the report | `report/results/*.json` (written by `scripts/acceptance_run.sh --json`) |

`detector_core.py` is the only source file I changed; `git diff --stat a048d1e HEAD -- src` lists just that file. Topic names, message types and the `infer()` output format are unchanged.

## Running it

Follow the course README (Ubuntu 22.04, ROS 2 Humble, Gazebo Classic 11). I ran it in WSL 2 on Windows as described in `docs/setup-windows.md`, with `MESA_D3D12_DEFAULT_ADAPTER_NAME=NVIDIA`.

One optional setting is added by my `detector_core.py`:

```bash
export TB3_DETECTOR_THREADS=4   # cap PyTorch CPU threads used by the detector; unset = default behaviour
```

On my laptop the default (8 threads) made the detector use about 8.5 CPU cores and starved SLAM Toolbox and Nav2; with 4 threads a frame took 32 ms instead of 45 ms. My acceptance runs and the demo video used `TB3_DETECTOR_THREADS=4`. Leave it unset to get the course's default behaviour.
