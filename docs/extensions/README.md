# Unified Planning Extensions

This folder illustrates the main extensions to the Unified Planning framework:

- **Arrays**: define multi-dimensional fluents
- **Sets**: fluents representing sets of objects  
- **Integer parameters in actions**: parameters that take integer values in a given range  
- **Range variables**: variables that allow iterating over a range of integers in preconditions/effects  
- **Count**: using counting expressions

This folder contains:
- `simple_tutorial.py`: a minimal didactic example demonstrating the extensions (not meant to be solved).
- `15-puzzle.py`: a real problem illustrating compilation strategies and solving.

---

## Compilation and Solving

To make models compatible with standard planners, we use compilation pipelines, which transform high-level constructs into low-level, planner-compatible ones.
We apply multiple compilation strategies to each domain model (when applicable), allowing us to generate several compiled versions.

Each domain defines the compilation mode it wants to use, which:
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

## Running a real example: 15-puzzle

To run a domain-specific model:

```bash
python docs/extensions/15-puzzle.py --compilation ut-integers --solving fast-downward
```
`--compilation:` choose a compilation pipeline (ut-integers, integers, etc.)
`--solving:` select the planner (fast-downward, enhsp, etc.)

The script will:
- Compile the 15-puzzle problem according to the chosen pipeline
- Solve it with the selected planner
- Print the plan and statistics (problem, actions)

---

## Installation

Make sure you're inside the project root folder (e.g. `unified-planning/`).
Also, add the project folder to your `PYTHONPATH` so that Python can find the package:

```bash
export PYTHONPATH=/path/to/unified-planning:$PYTHONPATH
```

To install the framework and make it available as a Python package, run:
```bash
pip install -e .
```

To install the required Python dependencies, run:
```bash
pip install -r requirements.txt
```