from unified_planning.shortcuts import *
import argparse


# Parser
parser = argparse.ArgumentParser(description="Solve 15-Puzzle Numeric")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Example 15Puzzle
initial_blocks = [[6,0,14,12],[1,15,9,10],[11,4,7,2],[8,3,5,13]]
goal_blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]
n = 4
l = 15

# ------------------------------------------------------------
# Problem
# ------------------------------------------------------------

npuzzle_problem = unified_planning.model.Problem('npuzzle_problem')

# ------------------------------------------------------------
# Fluents
# ------------------------------------------------------------

puzzle = Fluent('puzzle', ArrayType(n, ArrayType(n, IntType(0,l))))
npuzzle_problem.add_fluent(puzzle, default_initial_value=0)
npuzzle_problem.set_initial_value(puzzle, initial_blocks)

# ------------------------------------------------------------
# Actions
# ------------------------------------------------------------

move_up = unified_planning.model.InstantaneousAction('move_up', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_up.parameter('c')
r = move_up.parameter('r')
move_up.add_precondition(Equals(puzzle[r-1][c], 0))
move_up.add_effect(puzzle[r-1][c], puzzle[r][c])
move_up.add_effect(puzzle[r][c], 0)

move_down = unified_planning.model.InstantaneousAction('move_down', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_down.parameter('c')
r = move_down.parameter('r')
move_down.add_precondition(Equals(puzzle[r+1][c], 0))
move_down.add_effect(puzzle[r+1][c], puzzle[r][c])
move_down.add_effect(puzzle[r][c], 0)

move_left = unified_planning.model.InstantaneousAction('move_left', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_left.parameter('c')
r = move_left.parameter('r')
move_left.add_precondition(Equals(puzzle[r][c-1], 0))
move_left.add_effect(puzzle[r][c-1], puzzle[r][c])
move_left.add_effect(puzzle[r][c], 0)

move_right = unified_planning.model.InstantaneousAction('move_right', r=IntType(0,n-1), c=IntType(0,n-1))
c = move_right.parameter('c')
r = move_right.parameter('r')
move_right.add_precondition(Equals(puzzle[r][c+1], 0))
move_right.add_effect(puzzle[r][c+1], puzzle[r][c])
move_right.add_effect(puzzle[r][c], 0)

npuzzle_problem.add_actions([move_up, move_down, move_left, move_right])
npuzzle_problem.add_goal(Equals(puzzle, goal_blocks))

# ------------------------------------------------------------
# Costs
# ------------------------------------------------------------

costs: Dict[Action, Expression] = {
    move_up: Int(1),
    move_down: Int(1),
    move_left: Int(1),
    move_right: Int(1)
}
npuzzle_problem.add_quality_metric(MinimizeActionCosts(costs))

# ------------------------------------------------------------
# Compilation
# ------------------------------------------------------------

COMPILATION_PIPELINES = {
    'integers': [ # requires a numeric planner
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
    ],
    'ut-integers': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'logarithmic': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_LOGARITHMIC_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
    ],
    'None': []
}

if compilation not in COMPILATION_PIPELINES:
    raise ValueError(
        f"Unknown compilation pipeline: '{compilation}'. "
        f"Available: {list(COMPILATION_PIPELINES.keys())}"
    )
results = []
problem = npuzzle_problem
compilation_kinds = COMPILATION_PIPELINES[compilation]

for i, ck in enumerate(compilation_kinds, 1):
    print(f"Compiling {ck}")
    # Compilation parameters
    params = {}
    if ck == CompilationKind.ARRAYS_REMOVING:
        params = {'mode': 'permissive'}
    # Compile
    with Compiler(problem_kind=problem.kind, compilation_kind=ck, params=params) as compiler:
        result = compiler.compile(problem, ck)
        results.append(result)
        problem = result.problem

print(f"Compiled problem: {problem}")

# ------------------------------------------------------------
# Solving
# ------------------------------------------------------------

with OneshotPlanner(name=solving) as planner:
    if not planner.supports(problem.kind):
        print("Warning: Problem has unsupported features:")
        unsupported = [
            f"  - {feature}" for feature in problem.kind.features
            if feature not in planner.supported_kind().features
        ]
        print("\n".join(unsupported))

    # Solve
    result = planner.solve(problem)
    if result.plan is not None:
        print("Solution found!\n")
        # Map back through compilation
        plan = result.plan
        for comp_result in reversed(results):
            plan = plan.replace_action_instances(
                comp_result.map_back_action_instance
            )
        print(plan)
        print(f"\nActions: {len(plan.actions)}")
    else:
        print("No solution found")
        print(f"Status: {result.status}")