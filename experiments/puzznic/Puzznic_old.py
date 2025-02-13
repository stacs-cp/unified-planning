import os
import subprocess
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import *
import time
start = time.time()

puzznic_problem = Problem('puzznic_problem')
Pattern = UserType('Pattern')

F = Object('F', Pattern)  # Free
B = Object('B', Pattern)  # Blue
Y = Object('Y', Pattern)  # Yellow
G = Object('G', Pattern)  # Green
R = Object('R', Pattern)  # Red
L = Object('L', Pattern)  # Lightblue
O = Object('O', Pattern)  # Orange
V = Object('V', Pattern)  # Violet
P = Object('P', Pattern)  # Pink
C = Object('C', Pattern)  # Coal
puzznic_problem.add_objects([F, B, Y, G, R, L, O, V, P, C])

initial_blocks = [[F, R, F, F], [R, B, R, F], [B, G, B, F], [G, B, R, F]]

rows = len(initial_blocks)
columns = len(initial_blocks[0])
walls = [(0,0)]
print(initial_blocks)
print(rows, columns)

patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns, Pattern)))
puzznic_problem.add_fluent(patterned, default_initial_value=F)
puzznic_problem.set_initial_value(patterned, initial_blocks)

score = Fluent('score', IntType(0, 20))
puzznic_problem.add_fluent(score, default_initial_value=0)

falling_flag = Fluent('falling_flag', DerivedBoolType())
puzznic_problem.add_fluent(falling_flag, default_initial_value=False)
matching_flag = Fluent('matching_flag', DerivedBoolType())
puzznic_problem.add_fluent(matching_flag, default_initial_value=False)

axiom_falling = Axiom('axiom_falling')
axiom_falling.set_head(falling_flag)
axiom_falling.add_body_condition(
    Or([
        And(Not(Equals(patterned[i-1][j], F)), Equals(patterned[i][j], F))
        for i in range(1, rows) for j in range(columns)
        if (i, j) not in walls and (i-1, j) not in walls
    ])
)
puzznic_problem.add_axiom(axiom_falling)

axiom_matching = Axiom('axiom_matching')
axiom_matching.set_head(matching_flag)
axiom_matching.add_body_condition(
    Or([
           And(Equals(patterned[i][j], patterned[i][j + 1]), Not(Equals(patterned[i][j], F)))
           for i in range(rows) for j in range(columns - 1)
           if (i, j) not in walls and (i, j + 1) not in walls
       ] + [
           And(Equals(patterned[i][j], patterned[i + 1][j]), Not(Equals(patterned[i][j], F)))
           for i in range(rows - 1) for j in range(columns)
           if (i, j) not in walls and (i + 1, j) not in walls
       ])
)
puzznic_problem.add_axiom(axiom_matching)

move_block = InstantaneousAction('move_block', r=IntType(0, rows - 1), c=IntType(0, columns - 2),
                                 m=IntType(-1, 1))
r = move_block.parameter('r')
c = move_block.parameter('c')
m = move_block.parameter('m')
move_block.add_precondition(Not(falling_flag))
move_block.add_precondition(Not(matching_flag))
move_block.add_precondition(
    And([
            Or(Not(Equals(r, w[0])), Not(Equals(c, w[1])))  # Position we want to move is not wall
            for w in walls
        ] + [
            Or(Not(Equals(r, w[0])), Not(Equals(c + m, w[1])))  # Next position is not wall
            for w in walls
        ])
)
move_block.add_precondition(Not(Equals(m, 0)))
move_block.add_precondition(Not(Equals(patterned[r][c], F)))
move_block.add_precondition(Equals(patterned[r][c + m], F))
move_block.add_effect(patterned[r][c], F)
move_block.add_effect(patterned[r][c + m], patterned[r][c])
puzznic_problem.add_action(move_block)

fall_block = InstantaneousAction('fall_block', r=IntType(0, rows - 2), c=IntType(0, columns - 1))
r = fall_block.parameter('r')
c = fall_block.parameter('c')
fall_block.add_precondition(falling_flag)
fall_block.add_precondition(
    And([
            Or(Not(Equals(r, w[0])), Not(Equals(c, w[1])))  # Position to fall is not a wall
            for w in walls
        ] + [
            Or(Not(Equals(r + 1, w[0])), Not(Equals(c, w[1])))  # Next position (below) is not a wall
            for w in walls
        ])
)
fall_block.add_precondition(Not(Equals(patterned[r][c], F)))
fall_block.add_precondition(Equals(patterned[r + 1][c], F))
fall_block.add_effect(patterned[r][c], F)
fall_block.add_effect(patterned[r + 1][c], patterned[r][c])
puzznic_problem.add_action(fall_block)

match_blocks = InstantaneousAction('match_blocks')
match_blocks.add_precondition(Not(falling_flag))
match_blocks.add_precondition(matching_flag)
for i in range(rows):
    for j in range(columns):
        if (i, j) not in walls:
            or_exp = Or(Equals(patterned[i + 1][j], patterned[i][j]),
                        Equals(patterned[i - 1][j], patterned[i][j]),
                        Equals(patterned[i][j + 1], patterned[i][j]),
                        Equals(patterned[i][j - 1], patterned[i][j]))
            match_blocks.add_effect(patterned[i][j], F, And(or_exp, Not(Equals(patterned[i][j], F))))
puzznic_problem.add_action(match_blocks)


puzznic_problem.add_goal(And(Equals(patterned[i][j], F) for i in range(rows) for j in range(columns)))

costs: Dict[Action, Expression] = {
    move_block: Int(1),
    match_blocks: Int(0),
    fall_block: Int(0),
}
puzznic_problem.add_quality_metric(MinimizeActionCosts(costs))


from unified_planning.engines import CompilationKind, PlanGenerationResultStatus

# The CompilationKind class is defined in the unified_planning/engines/mixins/compiler.py file
compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    #CompilationKind.ARRAYS_AND_INTEGERS_REMOVING,
    CompilationKind.ARRAYS_REMOVING,
    #CompilationKind.INT_ARRAYS_BITS_REMOVING,
    #CompilationKind.INTEGERS_REMOVING,
    CompilationKind.USERTYPE_FLUENTS_REMOVING,
]
problem = puzznic_problem
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

# ----------------------------------------------------- Solver ---------------------------------------------------------
print(problem.kind)
mid = time.time()
print("Preprocessing:", mid - start)

with OneshotPlanner(problem_kind=problem.kind) as planner:
    result = planner.solve(problem)
    plan = result.plan
    end = time.time()
    print(f"Solving: {end-mid} seconds")
    if plan is None:
        print("No plan found.")
        print(result.log_messages)
        if not planner.supports(problem.kind):
           for pk in problem.kind.features:
               if pk not in planner.supported_kind().features:
                   print(f"{pk} is not supported by the planner")
    else:
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )
        print("Moves:", len(compiled_plan.actions))
        print("Compiled plan: ", compiled_plan.actions)

# ------------------------------------------------------ PDDL ----------------------------------------------------------

#domain_file = 'Puzznic_domain.pddl'
#problem_file = 'Puzznic_problem.pddl'
#
#if os.path.exists(domain_file):
#    os.remove(domain_file)
#if os.path.exists(problem_file):
#    os.remove(problem_file)
#
#w = PDDLWriter(problem, rewrite_bool_assignments=True)
#w.write_domain(domain_file)
#w.write_problem(problem_file)
#
#mid = time.time()
#print("Preprocessing:", mid - start)
#
#command = [f"/Users/cds26/PycharmProjects/unified-planning/venv/lib/python3.9/site-packages/up_fast_downward/downward/fast-downward.py",
#           domain_file, problem_file, "--search", "astar(blind())"]
#result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
#
#stdout_lines = result.stdout.splitlines()
#plan_lines = [line for line in stdout_lines if "move_" in line]
#end = time.time()
#print(f"Solving: {end-mid} seconds")
#
#if plan_lines:
#    plan = []
#    for line in plan_lines:
#        plan.append(line.split()[0])
#    print("Moves:", len(plan))
#    print("Plan: ", plan)
#else:
#    print("Plan not found.")
#if result.stderr:
#    print("Errors:")
#    print(result.stderr)
