import os
import subprocess
import time
from unified_planning.engines import PlanGenerationResultStatus
from unified_planning.io import PDDLWriter
from unified_planning.shortcuts import *

compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    # CompilationKind.ARRAYS_AND_INTEGERS_REMOVING,
    CompilationKind.ARRAYS_REMOVING,
    # CompilationKind.INT_ARRAYS_BITS_REMOVING,
    # CompilationKind.COUNT_REMOVING,
    CompilationKind.INTEGERS_REMOVING,
    CompilationKind.USERTYPE_FLUENTS_REMOVING,
]

def compile_plan(old_problem):
    problem = old_problem
    results = []
    for ck in compilation_kinds_to_apply:
        print("Compiling", ck)
        params = {}
        if ck == CompilationKind.ARRAYS_REMOVING:
            # 'mode' should be 'strict' or 'permissive'
            params = {'mode': 'permissive'}
        # To get the Compiler from the factory we can use the Compiler operation mode.
        # It takes a problem_kind and a compilation_kind, and returns a compiler with the capabilities we need
        with Compiler(
                problem_kind=problem.kind,
                compilation_kind=ck,
                params=params
        ) as compiler:
            result = compiler.compile(
                problem,
                ck
            )
            results.append(result)
            problem = result.problem
    return problem, results

def execute_plan(problem, planner_name, results):
    with OneshotPlanner(name=planner_name) as planner:
        result = planner.solve(problem)
        plan = result.plan

        if plan is None:
            error_message = "No plan found."
            print(result.log_messages)
            if not planner.supports(problem.kind):
                for pk in problem.kind.features:
                    if pk not in planner.supported_kind().features:
                        error_message += f"{pk} is not supported by the planner"
            return error_message
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )

        return compiled_plan.actions

def execute_pddl(problem, problem_name, heuristics):
    domain_file = f'{problem_name}_domain.pddl'
    problem_file = f'{problem_name}_problem.pddl'
    if os.path.exists(domain_file):
        os.remove(domain_file)
    if os.path.exists(problem_file):
        os.remove(problem_file)

    w = PDDLWriter(problem, rewrite_bool_assignments=True)
    w.write_domain(domain_file)
    w.write_problem(problem_file)

    command = [
        f"/Users/cds26/PycharmProjects/unified-planning/venv/lib/python3.9/site-packages/up_fast_downward/downward/fast-downward.py",
        domain_file, problem_file, "--search", f"astar({heuristics}())"]
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    stdout_lines = result.stdout.splitlines()
    print(stdout_lines)
    plan_lines = [line for line in stdout_lines if "move_block" in line]
    if plan_lines:
        return plan_lines
    else:
        return "Plan not found.", result.stderr


# ------------------------------------------------ Problem -------------------------------------------------------------
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

T = Object('T', Pattern)  # Ready to count
M = Object('M', Pattern)  # Ready to match
puzznic_problem.add_objects([F, B, Y, G, R, L, O, V, P, T, M])

#R R#
#P B#
## ##
#   #
#B P#
#initial_state = {(0,0): R, (0,2): R, (1,0): P, (1,2): B, (4,0): B, (4,2): P}
#rows = 5
#columns = 3
#undefined = [(2,0),(2,2)]

#R R#
initial_state = {(0,0): R, (0,2): R}
rows = 1
columns = 3
undefined = []
n_blocks = rows*columns - len(undefined)

# ------------------------------------------------ Fluents -------------------------------------------------------------
patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns, Pattern)), undefined_positions=undefined)
puzznic_problem.add_fluent(patterned, default_initial_value=F)

for p in initial_state:
    puzznic_problem.set_initial_value(patterned[p[0]][p[1]], initial_state[p])

#score = Fluent('score', IntType(0, 20))
#puzznic_problem.add_fluent(score, default_initial_value=0)

step = Fluent('step', IntType(1, 4))
puzznic_problem.add_fluent(step, default_initial_value=1)

blocks_matched = Fluent('blocks_matched', IntType(0, n_blocks))
puzznic_problem.add_fluent(blocks_matched, default_initial_value=0)

falling_flag = Fluent('falling_flag', DerivedBoolType())
puzznic_problem.add_fluent(falling_flag, default_initial_value=False)
matching_flag = Fluent('matching_flag', DerivedBoolType())
puzznic_problem.add_fluent(matching_flag, default_initial_value=False)
counting_flag = Fluent('counting_flag', DerivedBoolType())
puzznic_problem.add_fluent(counting_flag, default_initial_value=False)
scoring_flag = Fluent('scoring_flag', DerivedBoolType())
puzznic_problem.add_fluent(scoring_flag, default_initial_value=False)


# ------------------------------------------------ Axioms -------------------------------------------------------------
# Falling
axiom_falling = Axiom('axiom_falling')
axiom_falling.set_head(falling_flag)
i = RangeVariable('i', 1, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_falling.add_body_condition(
    Exists(And(Not(Equals(patterned[i-1][j], F)), Not(Equals(patterned[i][j], M)), Equals(patterned[i][j], F)), i,j)
)
puzznic_problem.add_axiom(axiom_falling)

# Matching
axiom_matching = Axiom('axiom_matching')
axiom_matching.set_head(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-2)
matching_horizontal = Exists(
    And(Equals(patterned[i][j], patterned[i][j + 1]), Not(Equals(patterned[i][j], F)),
        Not(Equals(patterned[i][j], M)), Not(Equals(patterned[i][j], T))), i,j
)
i = RangeVariable('i', 0, rows-2)
j = RangeVariable('j', 0, columns-1)
matching_vertical = Exists(
    And(Equals(patterned[i][j], patterned[i + 1][j]), Not(Equals(patterned[i][j], F)),
        Not(Equals(patterned[i][j], M)), Not(Equals(patterned[i][j], T))), i,j
)
axiom_matching.add_body_condition(
    Or(matching_horizontal, matching_vertical)
)
puzznic_problem.add_axiom(axiom_matching)

# Counting
axiom_counting = Axiom('axiom_counting')
axiom_counting.set_head(counting_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_counting.add_body_condition(
    Exists(Equals(patterned[i][j], M), i,j)
)
puzznic_problem.add_axiom(axiom_counting)

# Scoring
axiom_scoring = Axiom('axiom_scoring')
axiom_scoring.set_head(scoring_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_scoring.add_body_condition(
    Exists(Equals(patterned[i][j], T), i,j)
)
puzznic_problem.add_axiom(axiom_scoring)

# ------------------------------------------------ Actions -------------------------------------------------------------
# Move Block
move_block = InstantaneousAction('move_block', r=IntType(0, rows - 1), c=IntType(0, columns - 1),
                                 m=IntType(-1, 1))
r = move_block.parameter('r')
c = move_block.parameter('c')
m = move_block.parameter('m')
move_block.add_precondition(Not(falling_flag))
move_block.add_precondition(Not(matching_flag))
move_block.add_precondition(Not(counting_flag))
move_block.add_precondition(Not(scoring_flag))
move_block.add_precondition(Not(Equals(m, 0)))
move_block.add_precondition(Not(Equals(patterned[r][c], F)))
move_block.add_precondition(Not(Equals(patterned[r][c], M)))
move_block.add_precondition(Not(Equals(patterned[r][c], T)))
move_block.add_precondition(Equals(patterned[r][c + m], F))
move_block.add_effect(patterned[r][c], F)
move_block.add_effect(patterned[r][c + m], patterned[r][c])
move_block.add_effect(step, 1)
puzznic_problem.add_action(move_block)

# Fall Block
fall_block = InstantaneousAction('fall_block', r=IntType(0, rows - 2), c=IntType(0, columns - 1))
r = fall_block.parameter('r')
c = fall_block.parameter('c')
fall_block.add_precondition(falling_flag)
fall_block.add_precondition(Not(counting_flag))
fall_block.add_precondition(Not(scoring_flag))
fall_block.add_precondition(Not(Equals(patterned[r][c], F)))
fall_block.add_precondition(Not(Equals(patterned[r][c], M)))
fall_block.add_precondition(Not(Equals(patterned[r][c], T)))
fall_block.add_precondition(Equals(patterned[r + 1][c], F))
fall_block.add_effect(patterned[r][c], F)
fall_block.add_effect(patterned[r + 1][c], patterned[r][c])
puzznic_problem.add_action(fall_block)

# Match Blocks
match_blocks = InstantaneousAction('match_blocks')
match_blocks.add_precondition(Not(falling_flag))
match_blocks.add_precondition(Not(scoring_flag))
match_blocks.add_precondition(Not(counting_flag))
match_blocks.add_precondition(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
match_blocks.add_effect(patterned[i][j], M, condition=And(
    Not(Equals(patterned[i][j], F)), Not(Equals(patterned[i][j], M)), Not(Equals(patterned[i][j], T)),
    Or(Equals(patterned[i + 1][j], patterned[i][j]),
       Equals(patterned[i - 1][j], patterned[i][j]),
       Equals(patterned[i][j + 1], patterned[i][j]),
       Equals(patterned[i][j - 1], patterned[i][j]))
), forall=[i,j])
puzznic_problem.add_action(match_blocks)

# Count Block
count_block = InstantaneousAction('count_block', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
count_block.add_precondition(Not(falling_flag))
count_block.add_precondition(Not(matching_flag))
count_block.add_precondition(counting_flag)
count_block.add_precondition(Equals(patterned[r][c], M))
count_block.add_effect(patterned[r][c], T)
count_block.add_increase_effect(blocks_matched, 1)
puzznic_problem.add_action(count_block)

# to modify the blocks
r = RangeVariable('r', 0, rows-1)
c = RangeVariable('c', 0, columns-1)
# Count Block - with Count Expression
#count_block = InstantaneousAction('count_block', n=IntType(2, n_blocks))
#n = count_block.parameter('n')
#count_block.add_precondition(Not(falling_flag))
#count_block.add_precondition(Not(matching_flag))
#count_block.add_precondition(counting_flag)
#n_count = []
#for i in range(rows):
#    for j in range(columns):
#        if (i,j) not in undefined:
#            n_count.append(Equals(patterned[i][j], M))
#count_block.add_precondition(Equals(Count(n_count), n))
#count_block.add_effect(patterned[r][c], T, condition=Equals(patterned[r][c], M), forall=[r,c])
#count_block.add_increase_effect(blocks_matched, n)
#puzznic_problem.add_action(count_block)

# ----------------------------------------------------------------------------------------------------------------------

# Calculate Score
    # Step 1 Matching 2
score_blocks_1_2 = InstantaneousAction('score_blocks_1_2')
score_blocks_1_2.add_precondition(Not(matching_flag))
score_blocks_1_2.add_precondition(Not(counting_flag))
score_blocks_1_2.add_precondition(scoring_flag)
score_blocks_1_2.add_precondition(Equals(step, 1))
score_blocks_1_2.add_precondition(Equals(blocks_matched, 2))
score_blocks_1_2.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
score_blocks_1_2.add_effect(step, 2)
score_blocks_1_2.add_effect(blocks_matched, 0)
puzznic_problem.add_action(score_blocks_1_2)

    # Step 2 Matching 2
score_blocks_2_2 = InstantaneousAction('score_blocks_2_2')
score_blocks_2_2.add_precondition(Not(matching_flag))
score_blocks_2_2.add_precondition(Not(counting_flag))
score_blocks_2_2.add_precondition(scoring_flag)
score_blocks_2_2.add_precondition(Equals(step, 2))
score_blocks_2_2.add_precondition(Equals(blocks_matched, 2))
score_blocks_2_2.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
score_blocks_2_2.add_effect(step, 3)
score_blocks_2_2.add_effect(blocks_matched, 0)
puzznic_problem.add_action(score_blocks_2_2)

# Step 3 Matching 2
score_blocks_3_2 = InstantaneousAction('score_blocks_3_2')
score_blocks_3_2.add_precondition(Not(matching_flag))
score_blocks_3_2.add_precondition(Not(counting_flag))
score_blocks_3_2.add_precondition(scoring_flag)
score_blocks_3_2.add_precondition(Equals(step, 3))
score_blocks_3_2.add_precondition(Equals(blocks_matched, 2))
score_blocks_3_2.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
score_blocks_3_2.add_effect(step, 4)
score_blocks_3_2.add_effect(blocks_matched, 0)
puzznic_problem.add_action(score_blocks_3_2)

# Step 4 Matching
score_blocks_4 = InstantaneousAction('score_blocks_4')
score_blocks_4.add_precondition(Not(matching_flag))
score_blocks_4.add_precondition(Not(counting_flag))
score_blocks_4.add_precondition(scoring_flag)
score_blocks_4.add_precondition(Equals(step, 4))
score_blocks_4.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
score_blocks_4.add_effect(blocks_matched, 0)
puzznic_problem.add_action(score_blocks_4)

if n_blocks >= 3:
    # Step 2 Matching 3 or 4
    score_blocks_2_3_4 = InstantaneousAction('score_blocks_2_3_4')
    score_blocks_2_3_4.add_precondition(Not(matching_flag))
    score_blocks_2_3_4.add_precondition(Not(counting_flag))
    score_blocks_2_3_4.add_precondition(scoring_flag)
    score_blocks_2_3_4.add_precondition(Equals(step, 2))
    score_blocks_2_3_4.add_precondition(Or(Equals(blocks_matched, 3), Equals(blocks_matched, 4)))
    score_blocks_2_3_4.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_2_3_4.add_effect(step, 3)
    score_blocks_2_3_4.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_2_3_4)

    # Step 1 Matching 3
    score_blocks_1_3 = InstantaneousAction('score_blocks_1_3')
    score_blocks_1_3.add_precondition(Not(matching_flag))
    score_blocks_1_3.add_precondition(Not(counting_flag))
    score_blocks_1_3.add_precondition(scoring_flag)
    score_blocks_1_3.add_precondition(Equals(step, 1))
    score_blocks_1_3.add_precondition(Equals(blocks_matched, 3))
    score_blocks_1_3.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_3.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_1_3)

    # Step 3 Matching 3
    score_blocks_3_3 = InstantaneousAction('score_blocks_3_3')
    score_blocks_3_3.add_precondition(Not(matching_flag))
    score_blocks_3_3.add_precondition(Not(counting_flag))
    score_blocks_3_3.add_precondition(scoring_flag)
    score_blocks_3_3.add_precondition(Equals(step, 3))
    score_blocks_3_3.add_precondition(Equals(blocks_matched, 3))
    score_blocks_3_3.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r, c])
    score_blocks_3_3.add_effect(step, 4)
    score_blocks_3_3.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_3_3)

if n_blocks >= 4:
    # Step 1 Matching 4
    score_blocks_1_4 = InstantaneousAction('score_blocks_1_4')
    score_blocks_1_4.add_precondition(Not(matching_flag))
    score_blocks_1_4.add_precondition(Not(counting_flag))
    score_blocks_1_4.add_precondition(scoring_flag)
    score_blocks_1_4.add_precondition(Equals(step, 1))
    score_blocks_1_4.add_precondition(Equals(blocks_matched, 4))
    score_blocks_1_4.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_4.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_1_4)

    # Step 3 Matching 4 or more
    score_blocks_3_4 = InstantaneousAction('score_blocks_3_4')
    score_blocks_3_4.add_precondition(Not(matching_flag))
    score_blocks_3_4.add_precondition(Not(counting_flag))
    score_blocks_3_4.add_precondition(scoring_flag)
    score_blocks_3_4.add_precondition(Equals(step, 3))
    score_blocks_3_4.add_precondition(GE(blocks_matched, 4))
    score_blocks_3_4.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r, c])
    score_blocks_3_4.add_effect(step, 4)
    score_blocks_3_4.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_3_4)

if n_blocks >= 5:
    # Step 1 Matching 5
    score_blocks_1_5 = InstantaneousAction('score_blocks_1_5')
    score_blocks_1_5.add_precondition(Not(matching_flag))
    score_blocks_1_5.add_precondition(Not(counting_flag))
    score_blocks_1_5.add_precondition(scoring_flag)
    score_blocks_1_5.add_precondition(Equals(step, 1))
    score_blocks_1_5.add_precondition(Equals(blocks_matched, 5))
    score_blocks_1_5.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_5.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_1_5)

    # Step 2 Matching 5
    score_blocks_2_5 = InstantaneousAction('score_blocks_2_5')
    score_blocks_2_5.add_precondition(Not(matching_flag))
    score_blocks_2_5.add_precondition(Not(counting_flag))
    score_blocks_2_5.add_precondition(scoring_flag)
    score_blocks_2_5.add_precondition(Equals(step, 2))
    score_blocks_2_2.add_precondition(Equals(blocks_matched, 5))
    score_blocks_2_5.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r, c])
    score_blocks_2_5.add_effect(step, 3)
    score_blocks_2_5.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_2_5)

if n_blocks >= 6:
    # Step 1 Matching 6
    score_blocks_1_6 = InstantaneousAction('score_blocks_1_6')
    score_blocks_1_6.add_precondition(Not(matching_flag))
    score_blocks_1_6.add_precondition(Not(counting_flag))
    score_blocks_1_6.add_precondition(scoring_flag)
    score_blocks_1_6.add_precondition(Equals(step, 1))
    score_blocks_1_6.add_precondition(Equals(blocks_matched, 6))
    score_blocks_1_6.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_6.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_1_6)

    # Step 2 Matching 6 or more
    score_blocks_2_6 = InstantaneousAction('score_blocks_2_6')
    score_blocks_2_6.add_precondition(Not(matching_flag))
    score_blocks_2_6.add_precondition(Not(counting_flag))
    score_blocks_2_6.add_precondition(scoring_flag)
    score_blocks_2_6.add_precondition(Equals(step, 2))
    score_blocks_2_6.add_precondition(GE(blocks_matched, 6))
    score_blocks_2_6.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r, c])
    score_blocks_2_6.add_effect(step, 3)
    score_blocks_2_6.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_2_6)

if n_blocks >= 7:
    # Step 1 Matching 7
    score_blocks_1_7 = InstantaneousAction('score_blocks_1_7')
    score_blocks_1_7.add_precondition(Not(matching_flag))
    score_blocks_1_7.add_precondition(Not(counting_flag))
    score_blocks_1_7.add_precondition(scoring_flag)
    score_blocks_1_7.add_precondition(Equals(step, 1))
    score_blocks_1_7.add_precondition(Equals(blocks_matched, 7))
    score_blocks_1_7.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_7.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_1_7)

if n_blocks >= 8:
    # Step 1 Matching 8 or more
    score_blocks_1_8 = InstantaneousAction('score_blocks_1_8')
    score_blocks_1_8.add_precondition(Not(matching_flag))
    score_blocks_1_8.add_precondition(Not(counting_flag))
    score_blocks_1_8.add_precondition(scoring_flag)
    score_blocks_1_8.add_precondition(Equals(step, 1))
    score_blocks_1_8.add_precondition(GE(blocks_matched, 8))
    score_blocks_1_8.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_8.add_effect(step, 2)
    score_blocks_1_8.add_effect(blocks_matched, 0)
    puzznic_problem.add_action(score_blocks_1_8)

# ----------------------------------------------------------------------------------------------------------------------
    # Step 1
#score_blocks1 = InstantaneousAction('score_blocks1', n=IntType(2, n_blocks))
#n = score_blocks1.parameter('n')
#score_blocks1.add_precondition(Not(matching_flag))
#score_blocks1.add_precondition(Not(counting_flag))
#score_blocks1.add_precondition(scoring_flag)
#score_blocks1.add_precondition(Equals(step, 1))
#score_blocks1.add_precondition(Equals(blocks_matched, n))
#score_blocks1.add_increase_effect(score, 1, Equals(n, 2))
#score_blocks1.add_increase_effect(score, 2, Equals(n, 3))
#score_blocks1.add_increase_effect(score, 4, Equals(n, 4))
#score_blocks1.add_increase_effect(score, 6, Equals(n, 5))
#score_blocks1.add_increase_effect(score, 10, Equals(n, 6))
#score_blocks1.add_increase_effect(score, 12, Equals(n, 7))
#score_blocks1.add_increase_effect(score, 20, GE(n, 8))
#score_blocks1.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], C), forall=[r,c])
#score_blocks1.add_effect(step, 2)
#score_blocks1.add_effect(blocks_matched, 0)
#puzznic_problem.add_action(score_blocks1)
#
#    # Step 2
#score_blocks2 = InstantaneousAction('score_blocks2', n=IntType(2, n_blocks))
#n = score_blocks2.parameter('n')
#score_blocks2.add_precondition(Not(matching_flag))
#score_blocks2.add_precondition(Not(counting_flag))
#score_blocks2.add_precondition(scoring_flag)
#score_blocks2.add_precondition(Equals(step, 2))
#score_blocks2.add_precondition(Equals(blocks_matched, n))
#score_blocks2.add_increase_effect(score, 6, Equals(n, 2))
#score_blocks2.add_increase_effect(score, 10, Or(Equals(n, 3), Equals(n, 4)))
#score_blocks2.add_increase_effect(score, 12, Equals(n, 5))
#score_blocks2.add_increase_effect(score, 20, GE(n, 6))
#score_blocks2.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], C), forall=[r,c])
#score_blocks2.add_effect(step, 3)
#score_blocks2.add_effect(blocks_matched, 0)
#puzznic_problem.add_action(score_blocks2)
#
#    # Step 3
#score_blocks3 = InstantaneousAction('score_blocks3', n=IntType(2, n_blocks))
#n = score_blocks3.parameter('n')
#score_blocks3.add_precondition(Not(matching_flag))
#score_blocks3.add_precondition(Not(counting_flag))
#score_blocks3.add_precondition(scoring_flag)
#score_blocks3.add_precondition(Equals(step, 3))
#score_blocks3.add_precondition(Equals(blocks_matched, n))
#score_blocks3.add_increase_effect(score, 10, Equals(n, 2))
#score_blocks3.add_increase_effect(score, 12, Equals(n, 3))
#score_blocks3.add_increase_effect(score, 20, GE(n, 4))
#score_blocks3.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], C), forall=[r,c])
#score_blocks3.add_effect(step, 4)
#score_blocks3.add_effect(blocks_matched, 0)
#puzznic_problem.add_action(score_blocks3)
#
#    # Step 4
#score_blocks4 = InstantaneousAction('score_blocks4')
#score_blocks4.add_precondition(Not(matching_flag))
#score_blocks4.add_precondition(Not(counting_flag))
#score_blocks4.add_precondition(scoring_flag)
#score_blocks4.add_precondition(Equals(step, 4))
#score_blocks4.add_increase_effect(score, 20)
#score_blocks4.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], C), forall=[r,c])
#score_blocks4.add_effect(blocks_matched, 0)
#puzznic_problem.add_action(score_blocks4)
#

# ----------------------------------------------------------------------------------------------------------------------
# all at once...
#score_blocks = InstantaneousAction('score_blocks', n=IntType(2, n_blocks))
#n = score_blocks.parameter('n')
#score_blocks.add_precondition(Not(counting_flag))
#score_blocks.add_precondition(scoring_flag)
#score_blocks.add_precondition(Equals(blocks_matched, n))
#score_blocks.add_increase_effect(score, 1, And(Equals(step, 1), Equals(n, 2)))
#score_blocks.add_increase_effect(score, 2, And(Equals(step, 1), Equals(n, 3)))
#score_blocks.add_increase_effect(score, 4, And(Equals(step, 1), Equals(n, 4)))
#score_blocks.add_increase_effect(score, 6, Or(And(Equals(step, 1), Equals(n, 5)),
#                                              And(Equals(step, 2), Equals(n, 2))))
#score_blocks.add_increase_effect(score, 10, Or(And(Equals(step, 1), Equals(n, 6)),
#                                               And(Equals(step, 2), Or(Equals(n, 3), Equals(n, 4))),
#                                               And(Equals(step, 3), Equals(n, 2))))
#score_blocks.add_increase_effect(score, 12, Or(And(Equals(step, 1), Equals(n, 7)),
#                                               And(Equals(step, 2), Equals(n, 5)),
#                                               And(Equals(step, 3), Equals(n, 3))))
#score_blocks.add_increase_effect(score, 20, Or(And(Equals(step, 1), GE(n, 8)),
#                                               And(Equals(step, 2), GE(n, 6)),
#                                               And(Equals(step, 3), GE(n, 4)),
#                                               GE(step, 4)))
#score_blocks.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], C), forall=[r,c])
#score_blocks.add_increase_effect(step, 1)
#score_blocks.add_effect(blocks_matched, 0)
#puzznic_problem.add_action(score_blocks)

# ------------------------------------------------ Goal -------------------------------------------------------------
for i in range(rows):
    for j in range(columns):
        if (i,j) not in undefined:
            puzznic_problem.add_goal(Equals(patterned[i][j], F))

# ------------------------------------------------ Costs -------------------------------------------------------------
maximum_cost = 100
costs: Dict[Action, Expression] = {
    move_block: Int(0),
    match_blocks: Int(0),
    fall_block: Int(0),
    count_block: Int(0),
    score_blocks_1_2: maximum_cost - Int(1),
    score_blocks_2_2: maximum_cost - Int(6),
    score_blocks_3_2: maximum_cost - Int(10),
    score_blocks_4: maximum_cost - Int(20),
}
if n_blocks >= 3:
    costs.update({
        score_blocks_2_3_4: maximum_cost - Int(10),
        score_blocks_1_3: maximum_cost - Int(2),
        score_blocks_3_3: maximum_cost - Int(12),
    })
if n_blocks >= 4:
    costs.update({
        score_blocks_1_4: maximum_cost - Int(4),
        score_blocks_3_4: maximum_cost - Int(20),
    })
if n_blocks >= 5:
    costs.update({
        score_blocks_1_5: maximum_cost - Int(6),
        score_blocks_2_5: maximum_cost - Int(12)
    })
if n_blocks >= 6:
    costs.update({
        score_blocks_1_6: maximum_cost - Int(10),
        score_blocks_2_6: maximum_cost - Int(20)})
if n_blocks >= 7:
    costs.update({score_blocks_1_7: maximum_cost - Int(12)})
if n_blocks >= 8:
    costs.update({score_blocks_1_8: maximum_cost - Int(20)})

puzznic_problem.add_quality_metric(MinimizeActionCosts(costs))

#maximize_score = MaximizeExpressionOnFinalState(score)
#puzznic_problem.add_quality_metric(maximize_score)

# ---------------------------------------------- Compilation -----------------------------------------------------------
problem, results = compile_plan(puzznic_problem)
print(problem)
# ------------------------------------------------ Solving -------------------------------------------------------------

solving = 'fast-downward'

if solving.split('_')[0] == 'any':
    solving = solving.split('_')[1]
    with AnytimePlanner(name=solving) as planner:
        for res in planner.get_solutions(problem,  timeout=1800):
            print(f'Plan found.\n{res}')
else:
    with OneshotPlanner(name=solving) as planner:
        result = planner.solve(problem)
        if result.status == PlanGenerationResultStatus.SOLVED_SATISFICING:
            print(f'Plan found.\n{result.plan}')
        else:
            print("No plan found.")
            print(result)


#solv_start = time.time()
#solver = 'symk'
#print("Solving...")
#if solver == 'pddl':
#    plan = execute_pddl(problem, 'puzznic', 'blind')
#    print(f"Moves: {len(plan)}")
#    print(f"Plan: {plan}")
#
#    domain_path = "puzznic_domain.pddl"
#    problem_path = "puzznic_problem.pddl"
#
#else:
#    compiled_actions = execute_plan(problem, planner_name=solver, results=results)
#    if isinstance(compiled_actions, str):
#        print(compiled_actions)
#    else:
#        print("Compiled plan: ", compiled_actions)
#        total_score = maximum_cost
#        for a in compiled_actions:
#            if str(a).split('_')[0] == 'score':
#                score_action = puzznic_problem.action(str(a))
#                for qm in puzznic_problem.quality_metrics:
#                    total_score -= qm.get_action_cost(score_action).simplify()
#
#        print(f"Total score: {total_score.simplify()}")
#
#solv_end = time.time()
#print(f"Solving time: {solv_end-solv_start:.3f} seconds")

#for result in reversed(results):
#    #print(result.plan.replace_action_instances())
#    print("from??:", result.map_back_action_instance)
#

#for action in plan:
#    print(f"Applying action: {action}")
#    state = puzznic_problem.apply_action(state, action)
#    print(f"Score after action: {state.get_value(score)}")