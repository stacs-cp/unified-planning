from unified_planning.shortcuts import *
import time
start = time.time()

n_puzzle_problem = unified_planning.model.Problem('n_puzzle_problem')
x = 3
puzzle = Fluent('puzzle', ArrayType(x, ArrayType(x, IntType(0,8))))
n_puzzle_problem.add_fluent(puzzle)
n_puzzle_problem.set_initial_value(puzzle, [[8,0,6],[5,4,7],[2,3,1]])


slide_up = unified_planning.model.InstantaneousAction('slide_up', r=IntType(1,x-1), c=IntType(0,x-1))
c = slide_up.parameter('c')
r = slide_up.parameter('r')
slide_up.add_precondition(Equals(puzzle[r-1][c], 0))
slide_up.add_effect(puzzle[r-1][c], puzzle[r][c])
slide_up.add_effect(puzzle[r][c], 0)

slide_down = unified_planning.model.InstantaneousAction('slide_down', r=IntType(0,x-2), c=IntType(0,x-1))
c = slide_down.parameter('c')
r = slide_down.parameter('r')
slide_down.add_precondition(Equals(puzzle[r+1][c], 0))
slide_down.add_effect(puzzle[r+1][c], puzzle[r][c])
slide_down.add_effect(puzzle[r][c], 0)

slide_left = unified_planning.model.InstantaneousAction('slide_left', r=IntType(0,x-1), c=IntType(1,x-1))
c = slide_left.parameter('c')
r = slide_left.parameter('r')
slide_left.add_precondition(Equals(puzzle[r][c-1], 0))
slide_left.add_effect(puzzle[r][c-1], puzzle[r][c])
slide_left.add_effect(puzzle[r][c], 0)

slide_right = unified_planning.model.InstantaneousAction('slide_right', r=IntType(0,x-1), c=IntType(0,x-2))
c = slide_right.parameter('c')
r = slide_right.parameter('r')
slide_right.add_precondition(Equals(puzzle[r][c+1], 0))
slide_right.add_effect(puzzle[r][c+1], puzzle[r][c])
slide_right.add_effect(puzzle[r][c], 0)

n_puzzle_problem.add_actions([slide_up, slide_down, slide_left, slide_right])

n_puzzle_problem.add_goal(Equals(puzzle, [[0,1,2],[3,4,5],[6,7,8]]))


from unified_planning.engines import CompilationKind
# The CompilationKind class is defined in the unified_planning/engines/mixins/compiler.py file
compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    CompilationKind.ARRAYS_REMOVING
]

problem = n_puzzle_problem
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

with OneshotPlanner(name='enhsp-opt') as planner:
    result = planner.solve(problem)
    plan = result.plan
    end = time.time()
    print(f"Solving: {end-mid} seconds")
    if plan is None:
        print("No plan found.")
    else:
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )
        print("Moves:", len(compiled_plan.actions))
        print("Compiled plan: ", compiled_plan.actions)