# Scout — CCTV Motion & Object Finder

**Stop scrubbing through hours of footage.** Scout scans a folder of CCTV
exports and tells you **when something moves** and **what it is**: people,
vehicles, weapons, bags and animals. Every clip is laid out on an activity
timeline, so you can jump straight to the moments worth watching.

Scout runs entirely on your own machine. Footage never leaves it, and once
the model weights are downloaded it works fully offline.

## Features

- **Batch analysis**: point it at a folder and it works through every clip,
  with results appearing as each file finishes.
- **Motion detection** that ignores what doesn't matter: shadows, IR
  switching, lights turning on and cameras being knocked are filtered out.
- **Object detection** with YOLO: people, vehicles, bags, animals and
  weapons, with an optional custom model for firearms.
- **Watch and ignore zones**: draw boxes over a doorway to focus on it, or
  over the burned-in clock, trees and busy roads to mute them.
- **Activity timeline** per clip, plus a motion heatmap showing where
  movement happened.
- **Event list with snapshots**: each event has its best frame with boxes
  drawn, its time in the clip, its length and a confidence score.
- **CSV export** for reports or further analysis.
- **Works with DVR formats** that other tools often skip: `.mp4`, `.avi`,
  `.mov`, `.mkv`, `.asf`, `.wmv`, `.ts`, `.dav`, `.264`, `.h264`, `.mpg`,
  `.mpeg`, `.m4v`, `.webm`.
- **Runs on CPU**, and a lot faster with an NVIDIA GPU.

## Installation

You need Python 3.10 or newer.

```bash
git clone https://github.com/harry083/scout.git
cd scout
python -m venv .venv
```

Activate the virtual environment and install the dependencies:

```bash
# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### GPU acceleration (optional)

`ultralytics` installs the CPU build of PyTorch by default. If you have an
NVIDIA GPU, install the CUDA build of PyTorch first by following
[pytorch.org/get-started](https://pytorch.org/get-started). Object detection
is several times faster on a GPU.

### Motion only

Object detection is optional. If `ultralytics` isn't installed, Scout still
runs and reports motion only. To get this lighter install, remove
`ultralytics` from `requirements.txt` before installing.

### Offline machines

The YOLO weights (`yolo26n.pt`, about 5 MB) download into `models/` the first
time object detection runs. On a machine without internet access, download
the `.pt` files somewhere else and copy them into `models/` by hand.

## Usage

Start the server:

```bash
python run.py
```

Then open **http://localhost:8758** in your browser. The server only listens
on your own machine (`127.0.0.1`).

1. **Folder**: paste a path, or use *Browse…* to pick the folder of exports.
2. **Zones** (optional): drag boxes on a preview frame.
   - *Watch* zones: only motion and objects inside them are reported, for
     example a doorway or a car park.
   - *Ignore* zones: nothing inside them is reported. Use them for the
     burned-in clock, trees, flags or a busy road in the background.

   Zones are kept when you load another folder from the same camera.
3. **Motion**: *Sensitivity* sets how small or subtle a movement has to be
   to count. *Frames checked* trades speed for catching very brief movement.
4. **Objects**: choose the model size, the minimum confidence, how often to
   look, and which kinds of object to report. *Only around motion* (on by
   default) runs the detector only while something is moving. That's much
   faster on long, quiet footage, but it won't report a parked car or
   someone standing completely still for the whole clip.
5. **Analyse Files**. Results fill in as each file finishes:
   - **Activity timeline**: one row per clip, with time measured from the
     start of the clip. The blue band shows the motion level, and the
     coloured lanes underneath show sightings of each object type. Hover for
     details, click a bar to open that event, or click a file name to see
     its motion heatmap.
   - **Events**: every motion event and sighting, with a snapshot, the time
     in the clip, how long it lasted and a confidence score. Click one to
     see the full snapshot with boxes drawn, then step through with ← →.
     *Open Clip* plays it in your default video player.
6. **Export CSV** to download one row per event.

### Tips

- Times are **seconds from the start of each clip**, not time of day. Add
  them to the clip's start time (usually in the burned-in timestamp or the
  file name) to get the real time.
- Always add an ignore zone over the on-screen clock. Otherwise its ticking
  digits can show up as motion.
- For fast scanning use the `n` (nano) model. Step up to `s` or `m` when
  accuracy matters more than speed.

## Detecting weapons

The stock YOLO models are trained on the COCO dataset, whose only weapon
classes are *knife* and *baseball bat*. **They will not detect firearms.**

To detect firearms, enter the path of a YOLO model trained for weapons in
*Extra model*. Any Ultralytics `.pt` or `.onnx` model works, for example one
fine-tuned on a public firearms dataset. It runs alongside the stock model
on the same frames. Its classes are sorted by name: anything containing
*gun*, *pistol*, *rifle*, *firearm*, *weapon*, *revolver*, *knife*,
*machete* or *sword* (or the words *blade*, *axe*, *bat*, *crowbar* or
*hammer*) counts as a **weapon**. Person and vehicle classes are recognised
the same way.

> **Treat weapon hits as leads to check, not findings.** Phones, tools and
> umbrellas get flagged, and small or partly hidden weapons get missed.
> Always review the footage yourself.

## How it works

For each clip, Scout:

1. **Samples frames** at a fixed rate instead of decoding every one.
2. **Detects motion** (`motion.py`). Frames are scaled down to 480 px wide
   and fed to OpenCV's MOG2 background model. It works in colour, not
   greyscale, because a red coat on grey paving is nearly the same
   brightness. Shadows are dropped, the mask is clipped to your zones and
   cleaned up, and blobs above a minimum size become boxes. A change
   covering more than 55% of the frame at once is reported as a *lighting
   change* rather than motion. It also builds the motion heatmap.
3. **Detects objects** (`detector.py`) on the samples worth looking at,
   using one or two YOLO models. Their class names are sorted into person,
   vehicle, weapon, bag and animal.
4. **Builds events** (`events.py`). Observations of the same kind less than
   a couple of seconds apart are joined into one event, so someone passing
   behind a pillar stays one sighting. The strongest frame is kept as the
   snapshot.

## Using it as a Python library

The web app is a thin layer over the analysis modules, which you can use
directly in your own scripts:

```python
from pathlib import Path
from analyse import AnalysisSettings, analyse_video, zones_from_dicts
from detector import get_detector

settings = AnalysisSettings(
    sensitivity="medium",           # low | medium | high
    zones=zones_from_dicts([{"x": 0, "y": 0, "w": 420, "h": 45, "mode": "ignore"}]),
    categories={"person", "vehicle", "weapon"},
    confidence=0.4,
)
clip = analyse_video(Path("cam1.mp4"), settings, get_detector("n"))

for ev in clip.events:
    print(ev.kind, ev.start, ev.end, ev.peak, sorted(ev.labels))
```

`get_detector` takes a model size (`"n"`, `"s"` or `"m"`) and an optional
`extra_model` path for a custom model. Detectors are cached, so repeated
calls reuse the loaded model.

## Project structure

```
├── backend/
│   ├── main.py            FastAPI app and API routes
│   ├── sessions.py        per-folder state and background analysis worker
│   └── file_browser.py    server-side directory listing for the folder picker
├── frontend/              web UI (plain HTML, CSS and JavaScript)
├── models/                YOLO weights (downloaded on first use, or copied in)
├── motion.py              background-subtraction motion detection and heatmap
├── detector.py            YOLO wrapper, maps class names to categories
├── events.py              turns per-frame observations into timed events
├── analyse.py             sample → motion → objects → events for one clip
├── run.py                 web app entry point (port 8758)
├── requirements.txt
└── LICENSE
```

## Limitations

- Event times are relative to the start of each clip, not wall-clock time.
- With *Only around motion* on, completely stationary objects aren't
  reported.
- Detection quality depends on the footage. Low resolution, poor light,
  heavy compression and steep camera angles all reduce accuracy.
- Scout is an aid for reviewing footage. It does not replace watching the
  relevant parts yourself.

## License

Scout is released under the [MIT License](LICENSE).

Object detection uses [Ultralytics YOLO](https://github.com/ultralytics/ultralytics),
which is licensed separately under AGPL-3.0. Check its terms if you
redistribute Scout with `ultralytics` bundled, or offer it as a service.
