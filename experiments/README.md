# Experiments

This directory contains a collection of classical planning problems implemented using the Unified Planning framework with our proposed extensions. 
Each folder corresponds to a specific domain and includes:
- A **handcrafted model** (in PDDL)
- One or more **extended models** (in Python) using different compilation strategies
- Problem instances
- Scripts to run models

---

## Folder Structure

Each domain folder (e.g. `15-Puzzle`, `Sokoban`) typically includes:

- `<Domain>.py`: our model
- (if applicable) `<Domain>Numeric.py`: same model using numeric values
- `instances.txt`: lists the problem instances used
- `handcrafted/`:  
  - `domain.pddl`: domain handcrafted PDDL model  
  - `<instance>.pddl`: problem instances
- `handcrafted_reader.py`: loads a specific handcrafted instance
- All models internally call the general compilation/solving script:  
  `experiments/compilation_solving.py`

Some domains also include:
- `read_instance.py`: a script used to read problem descriptions from text files
- `probs/`: a folder containing `<instance>.txt` files that describe the problem grid

The `read_instance.py` script parses these files and extracts all relevant elements of the instance (such as the initial state, undefined positions, and grid dimensions).  
It is invoked from within the corresponding domain model (e.g. `<Domain>.py`) by specifying which instance to load.

---

## How It Works

Our extended models are defined using both extended and existing constructs such as arrays, integer parameters in actions, and user-defined types.
To make these models compatible with standard planners, we apply **compilation strategies**, defined as a **sequence of *compilers***. 
Each compiler transforms specific constructs into equivalent low-level, planner-compatible representations.
We apply multiple compilation strategies to each domain model (when applicable), allowing us to generate several compiled versions.

Each domain defines the compilation mode it wants to use and delegates the work to `compilation_solving.py`, which:
1. Applies the selected compilation pipeline
2. Solves the resulting problem with the selected planner

### Compiler abbreviations:
- `IPAR`: INT_PARAMETER_ACTIONS_REMOVING
- `AR`: ARRAYS_REMOVING
- `ALR`: ARRAYS_LOGARITHMIC_REMOVING
- `IR`: INTEGERS_REMOVING
- `UFR`: USERTYPE_FLUENTS_REMOVING
- `CR`: COUNT_REMOVING
- `CIR`: COUNT_INT_REMOVING

| Compilation Strategies | Compilers Sequence       |
|-----------------|--------------------------|
| `up`            | `IPAR, AR, UFR`          |
| `logarithmic`   | `IPAR, ALR`              |
| `ut-integers`   | `IPAR, AR, IR, UFR`      |
| `count`         | `IPAR, AR, CR, UFR`      |
| `count-int`     | `IPAR, AR, CIR, IR, UFR` |
| `count-int-num` | `IPAR, AR, CIR, UFR`     |
| `integers`      | `IPAR, AR`               |

---

## Installation

Make sure you're inside the project root folder (e.g. `unified-planning/`).

To install the framework and make it available as a Python package, run:
```bash
pip install -e .
```

To install the required Python dependencies, run:
```bash
pip install -r requirements.txt
```

To run a domain-specific model:
```bash
python experiments/<Domain>/<Domain>.py --compilation <Compilation> --solving <Solving>
```

For example:
```bash
python experiments/15-puzzle/15Puzzle.py --compilation up --solving fast-downward
```
