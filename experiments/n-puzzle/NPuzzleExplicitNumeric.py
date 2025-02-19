from unified_planning.shortcuts import *
import time
start = time.time()

compilation = 'ut-integers'
solving = 'fast-downward'

# Example 8Puzzle
initial_blocks = [[8,0,6],[5,4,7],[2,3,1]]
goal_blocks = [[0,1,2],[3,4,5],[6,7,8]]

# Example 15Puzzle
#initial_blocks = [[14,13,15,7],[11,12,9,5],[6,0,2,1],[4,8,10,3]]
#goal_blocks = [[0,1,2,3],[4,5,6,7],[8,9,10,11],[12,13,14,15]]

n = len(initial_blocks)

# ---------------------------------------------------- Problem ---------------------------------------------------------

npuzzle_problem = unified_planning.model.Problem('npuzzle_problem')

puzzle = Fluent('puzzle', ArrayType(n, ArrayType(n, IntType(0,15))))
npuzzle_problem.add_fluent(puzzle)
npuzzle_problem.set_initial_value(puzzle, initial_blocks)

move_up = unified_planning.model.InstantaneousAction('move_up', t=IntType(1,15), r=IntType(1,n-1), c=IntType(0,n-1))
t = move_up.parameter('t')
c = move_up.parameter('c')
r = move_up.parameter('r')
move_up.add_precondition(Equals(puzzle[r-1][c], 0))
move_up.add_precondition(Equals(puzzle[r][c], t))
move_up.add_effect(puzzle[r-1][c], t)
move_up.add_effect(puzzle[r][c], 0)

move_down = unified_planning.model.InstantaneousAction('move_down', t=IntType(1,15), r=IntType(0,n-2), c=IntType(0,n-1))
t = move_down.parameter('t')
c = move_down.parameter('c')
r = move_down.parameter('r')
move_down.add_precondition(Equals(puzzle[r+1][c], 0))
move_down.add_precondition(Equals(puzzle[r][c], t))
move_down.add_effect(puzzle[r+1][c], t)
move_down.add_effect(puzzle[r][c], 0)

move_left = unified_planning.model.InstantaneousAction('move_left', t=IntType(1,15), r=IntType(0,n-1), c=IntType(1,n-1))
t = move_left.parameter('t')
c = move_left.parameter('c')
r = move_left.parameter('r')
move_left.add_precondition(Equals(puzzle[r][c-1], 0))
move_left.add_precondition(Equals(puzzle[r][c], t))
move_left.add_effect(puzzle[r][c-1], t)
move_left.add_effect(puzzle[r][c], 0)

move_right = unified_planning.model.InstantaneousAction('move_right', t=IntType(1,15), r=IntType(0,n-1), c=IntType(0,n-2))
t = move_right.parameter('t')
c = move_right.parameter('c')
r = move_right.parameter('r')
move_right.add_precondition(Equals(puzzle[r][c+1], 0))
move_right.add_precondition(Equals(puzzle[r][c], t))
move_right.add_effect(puzzle[r][c+1], t)
move_right.add_effect(puzzle[r][c], 0)

npuzzle_problem.add_actions([move_up, move_down, move_left, move_right])
npuzzle_problem.add_goal(Equals(puzzle, goal_blocks))

costs: Dict[Action, Expression] = {
    move_up: Int(1),
    move_down: Int(1),
    move_left: Int(1),
    move_right: Int(1)
}
npuzzle_problem.add_quality_metric(MinimizeActionCosts(costs))

print(npuzzle_problem.quality_metrics)

# -------------------------------------------------- Compilation -------------------------------------------------------

from unified_planning.engines import CompilationKind
if compilation == 'integers':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
    ]
elif compilation == 'ut-integers':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
elif compilation == 'logaritmic':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.INT_ARRAYS_BITS_REMOVING,
    ]
else:
    raise ValueError(f"Unsupported compilation type: {compilation}")

if solving == 'fast-downward-opt':
    compilation_kinds_to_apply.append(CompilationKind.CONDITIONAL_EFFECTS_REMOVING)


problem = npuzzle_problem
results = []
for ck in compilation_kinds_to_apply:
    params = {}
    if ck == CompilationKind.ARRAYS_REMOVING:
        # 'mode' should be 'strict' or 'permissive'
        params = {'mode': 'permissive'}
    # To get the Compiler from the factory we can use the Compiler operation mode.
    # It takes a problem_kind and a compilation_kind, and returns a compiler with the capabilities we need
    with Compiler(
            problem_kind = problem.kind,
            compilation_kind = ck,
            params=params
        ) as compiler:
        result = compiler.compile(
            problem,
            ck
        )
        results.append(result)
        problem = result.problem

mid = time.time()
print("Preprocessing", mid - start)

# ---------------------------------------------------- Solving ---------------------------------------------------------

with OneshotPlanner(name=solving) as planner:
    result = planner.solve(problem)

    plan = result.plan
    if plan is not None:
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )
        print(compiled_plan)
    else:
        print(result)

    if not planner.supports(problem.kind):
        unsupported_features = [
            f"{pk} is not supported by the planner"
            for pk in problem.kind.features if pk not in planner.supported_kind().features
        ]
        print("\n".join(unsupported_features))