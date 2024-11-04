import os
import subprocess

import selector

from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import *
import time
start = time.time()

n_puzzle_problem = Problem('n_puzzle_problem')
x = 3

puzzle = Fluent('puzzle', ArrayType(x, ArrayType(x, IntType(0, 8))))
n_puzzle_problem.add_fluent(puzzle, default_initial_value=0)
#n_puzzle_problem.set_initial_value(puzzle, [[0,1],[2,3]])
n_puzzle_problem.set_initial_value(puzzle, [[8,7,6],[0,4,1],[2,5,3]])
#n_puzzle_problem.set_initial_value(puzzle[0], [8,7,6])
#n_puzzle_problem.set_initial_value(puzzle[1], [0,4,1])
#n_puzzle_problem.set_initial_value(puzzle[2], [2,5,3])
#n_puzzle_problem.set_initial_value(puzzle, [[0,13,2,3],[4,1,6,7],[8,5,10,11],[12,9,14,15]])

slide_up = InstantaneousAction('slide_up', r=IntType(1, x - 1), c=IntType(0, x - 1))
c = slide_up.parameter('c')
r = slide_up.parameter('r')
slide_up.add_precondition(Equals(puzzle[r - 1][c], 0))
slide_up.add_effect(puzzle[r - 1][c], puzzle[r][c])
slide_up.add_effect(puzzle[r][c], 0)

slide_down = InstantaneousAction('slide_down', r=IntType(0, x - 2), c=IntType(0, x - 1))
c = slide_down.parameter('c')
r = slide_down.parameter('r')
slide_down.add_precondition(Equals(puzzle[r + 1][c], 0))
slide_down.add_effect(puzzle[r + 1][c], puzzle[r][c])
slide_down.add_effect(puzzle[r][c], 0)

slide_left = InstantaneousAction('slide_left', r=IntType(0, x - 1), c=IntType(1, x - 1))
c = slide_left.parameter('c')
r = slide_left.parameter('r')
slide_left.add_precondition(Equals(puzzle[r][c - 1], 0))
slide_left.add_effect(puzzle[r][c - 1], puzzle[r][c])
slide_left.add_effect(puzzle[r][c], 0)

slide_right = InstantaneousAction('slide_right', r=IntType(0, x - 1), c=IntType(0, x - 2))
c = slide_right.parameter('c')
r = slide_right.parameter('r')
slide_right.add_precondition(Equals(puzzle[r][c + 1], 0))
slide_right.add_effect(puzzle[r][c + 1], puzzle[r][c])
slide_right.add_effect(puzzle[r][c], 0)

n_puzzle_problem.add_actions([slide_up, slide_down, slide_left, slide_right])
#n_puzzle_problem.add_action(slide_up)

#n_puzzle_problem.add_goal(Equals(puzzle, [[2,1],[0,3]])
#n_puzzle_problem.add_goal(Equals(puzzle[0], [0,1,2]))
n_puzzle_problem.add_goal(Equals(puzzle, [[0,1,2],[3,4,5],[6,7,8]]))
#n_puzzle_problem.add_goal(Equals(puzzle[0][0], 0))
#n_puzzle_problem.add_goal(Equals(puzzle[0][1], 1))
#n_puzzle_problem.add_goal(Equals(puzzle[0][2], 2))
#n_puzzle_problem.add_goal(Equals(puzzle[1][0], 2))
#n_puzzle_problem.add_goal(Equals(puzzle[1][1], 4))
#n_puzzle_problem.add_goal(Equals(puzzle[1][2], 5))
#n_puzzle_problem.add_goal(Equals(puzzle[2][0], 1))
#n_puzzle_problem.add_goal(Equals(puzzle[2][1], 7))
#n_puzzle_problem.add_goal(Equals(puzzle[2][2], 8))
#n_puzzle_problem.add_goal(Equals(puzzle, [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]))

#print(n_puzzle_problem.kind.features)

from unified_planning.engines import CompilationKind, PlanGenerationResultStatus

# The CompilationKind class is defined in the unified_planning/engines/mixins/compiler.py file
compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    #CompilationKind.ARRAYS_AND_INTEGERS_REMOVING,
    #CompilationKind.ARRAYS_REMOVING,
    CompilationKind.INT_ARRAYS_BITS_REMOVING,
    #CompilationKind.INTEGERS_REMOVING,
    #CompilationKind.USERTYPE_FLUENTS_REMOVING,
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
print(problem)

# ----------------------------------------------------- Solver ---------------------------------------------------------

#mid = time.time()
#print("Preprocessing:", mid - start)
#
#with OneshotPlanner(name='fast-downward') as planner:
#    result = planner.solve(problem)
#    plan = result.plan
#    end = time.time()
#    print(f"Solving: {end-mid} seconds")
#    if plan is None:
#        print("No plan found.")
#        print(result.log_messages)
#        if not planner.supports(problem.kind):
#           for pk in problem.kind.features:
#               if pk not in planner.supported_kind().features:
#                   print(f"{pk} is not supported by the planner")
#    else:
#        compiled_plan = plan
#        for result in reversed(results):
#            compiled_plan = compiled_plan.replace_action_instances(
#                result.map_back_action_instance
#            )
#        print("Moves:", len(compiled_plan.actions))
#        print("Compiled plan: ", compiled_plan.actions)

# ------------------------------------------------------ PDDL ----------------------------------------------------------

domain_file = '8Puzzle_domain.pddl'
problem_file = '8Puzzle_problem.pddl'

if os.path.exists(domain_file):
    os.remove(domain_file)
if os.path.exists(problem_file):
    os.remove(problem_file)

w = PDDLWriter(problem, rewrite_bool_assignments=True)
w.write_domain(domain_file)
w.write_problem(problem_file)

mid = time.time()
print("Preprocessing:", mid - start)

command = [f"/Users/cds26/PycharmProjects/unified-planning/venv/lib/python3.9/site-packages/up_fast_downward/downward/fast-downward.py",
           "8Puzzle_domain.pddl", "8Puzzle_problem.pddl", "--search", "astar(hmax())"]
#command = [f"/Users/cds26/PycharmProjects/unified-planning/venv/lib/python3.9/site-packages/up_enhsp/enhsp_planner.py",
#           "8Puzzle_domain.pddl", "8Puzzle_problem.pddl"]
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

stdout_lines = result.stdout.splitlines()
plan_lines = [line for line in stdout_lines if "slide" in line]
end = time.time()
print(f"Solving: {end-mid} seconds")

if plan_lines:
    plan = []
    for line in plan_lines:
        plan.append(line.split()[0])
    print("Moves:", len(plan))
    print("Plan: ", plan)
else:
    print("Plan not found.")
if result.stderr:
    print("Errors:")
    print(result.stderr)
