import subprocess
from unified_planning.shortcuts import *

compilation = 'up'
solving = 'symk'
instance_name = '5x3-trivial'

instance = subprocess.run(['python3', 'read_instance.py', instance_name], text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
output = instance.stdout.split("---")
initial_state = eval(output[0].strip())
undefined  = eval(output[1].strip())
rows = eval(output[2].strip())
columns = eval(output[3].strip())
n_blocks = rows*columns - len(undefined)

# ---------------------------------------------------- Problem ---------------------------------------------------------

puzznic_scoring_problem = unified_planning.model.Problem('puzznic_scoring_problem')
Pattern = UserType('Pattern')
F = Object('F', Pattern) # Free
B = Object('B', Pattern) # Blue
Y = Object('Y', Pattern) # Yellow
G = Object('G', Pattern) # Green
R = Object('R', Pattern) # Red
L = Object('L', Pattern) # Lightblue
O = Object('O', Pattern) # Orange
V = Object('V', Pattern) # Violet
P = Object('P', Pattern) # Pink
C = Object('C', Pattern) # Coal

T = Object('T', Pattern)  # Ready to count
M = Object('M', Pattern)  # Ready to match
puzznic_scoring_problem.add_objects([F, T, M])

patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns, Pattern)), undefined_positions=undefined)
puzznic_scoring_problem.add_fluent(patterned, default_initial_value=F)

for l, p in initial_state.items():
    if not puzznic_scoring_problem.has_object(p):
        puzznic_scoring_problem.add_object(eval(p))
    puzznic_scoring_problem.set_initial_value(patterned[l[0]][l[1]], eval(initial_state[l]))

step = Fluent('step', IntType(1, 4))
puzznic_scoring_problem.add_fluent(step, default_initial_value=1)

blocks_matched = Fluent('blocks_matched', IntType(0, n_blocks))
puzznic_scoring_problem.add_fluent(blocks_matched, default_initial_value=0)

falling_flag = Fluent('falling_flag', DerivedBoolType())
puzznic_scoring_problem.add_fluent(falling_flag, default_initial_value=False)
matching_flag = Fluent('matching_flag', DerivedBoolType())
puzznic_scoring_problem.add_fluent(matching_flag, default_initial_value=False)
counting_flag = Fluent('counting_flag', DerivedBoolType())
puzznic_scoring_problem.add_fluent(counting_flag, default_initial_value=False)
scoring_flag = Fluent('scoring_flag', DerivedBoolType())
puzznic_scoring_problem.add_fluent(scoring_flag, default_initial_value=False)

# ------------------------------------------------ Axioms -------------------------------------------------------------
# Falling
axiom_falling = Axiom('axiom_falling')
axiom_falling.set_head(falling_flag)
i = RangeVariable('i', 1, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_falling.add_body_condition(
    Exists(And(Not(Equals(patterned[i-1][j], F)), Not(Equals(patterned[i][j], M)), Equals(patterned[i][j], F)), i,j)
)
puzznic_scoring_problem.add_axiom(axiom_falling)

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
puzznic_scoring_problem.add_axiom(axiom_matching)

# Counting
axiom_counting = Axiom('axiom_counting')
axiom_counting.set_head(counting_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_counting.add_body_condition(
    Exists(Equals(patterned[i][j], M), i,j)
)
puzznic_scoring_problem.add_axiom(axiom_counting)

# Scoring
axiom_scoring = Axiom('axiom_scoring')
axiom_scoring.set_head(scoring_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_scoring.add_body_condition(
    Exists(Equals(patterned[i][j], T), i,j)
)
puzznic_scoring_problem.add_axiom(axiom_scoring)

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
puzznic_scoring_problem.add_action(move_block)

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
puzznic_scoring_problem.add_action(fall_block)

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
puzznic_scoring_problem.add_action(match_blocks)

# Count Block
count_block = InstantaneousAction('count_block', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
count_block.add_precondition(Not(falling_flag))
count_block.add_precondition(Not(matching_flag))
count_block.add_precondition(counting_flag)
count_block.add_precondition(Equals(patterned[r][c], M))
count_block.add_effect(patterned[r][c], T)
count_block.add_increase_effect(blocks_matched, 1)
puzznic_scoring_problem.add_action(count_block)

# --------------------------------------- Calculate Score ----------------------------------------------------------------------
# to modify the M blocks
r = RangeVariable('r', 0, rows-1)
c = RangeVariable('c', 0, columns-1)

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
puzznic_scoring_problem.add_action(score_blocks_1_2)

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
puzznic_scoring_problem.add_action(score_blocks_2_2)

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
puzznic_scoring_problem.add_action(score_blocks_3_2)

# Step 4 Matching
score_blocks_4 = InstantaneousAction('score_blocks_4')
score_blocks_4.add_precondition(Not(matching_flag))
score_blocks_4.add_precondition(Not(counting_flag))
score_blocks_4.add_precondition(scoring_flag)
score_blocks_4.add_precondition(Equals(step, 4))
score_blocks_4.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
score_blocks_4.add_effect(blocks_matched, 0)
puzznic_scoring_problem.add_action(score_blocks_4)

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
    puzznic_scoring_problem.add_action(score_blocks_2_3_4)

    # Step 1 Matching 3
    score_blocks_1_3 = InstantaneousAction('score_blocks_1_3')
    score_blocks_1_3.add_precondition(Not(matching_flag))
    score_blocks_1_3.add_precondition(Not(counting_flag))
    score_blocks_1_3.add_precondition(scoring_flag)
    score_blocks_1_3.add_precondition(Equals(step, 1))
    score_blocks_1_3.add_precondition(Equals(blocks_matched, 3))
    score_blocks_1_3.add_effect(patterned[r][c], F, condition=Equals(patterned[r][c], T), forall=[r,c])
    score_blocks_1_3.add_effect(blocks_matched, 0)
    puzznic_scoring_problem.add_action(score_blocks_1_3)

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
    puzznic_scoring_problem.add_action(score_blocks_3_3)

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
    puzznic_scoring_problem.add_action(score_blocks_1_4)

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
    puzznic_scoring_problem.add_action(score_blocks_3_4)

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
    puzznic_scoring_problem.add_action(score_blocks_1_5)

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
    puzznic_scoring_problem.add_action(score_blocks_2_5)

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
    puzznic_scoring_problem.add_action(score_blocks_1_6)

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
    puzznic_scoring_problem.add_action(score_blocks_2_6)

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
    puzznic_scoring_problem.add_action(score_blocks_1_7)

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
    puzznic_scoring_problem.add_action(score_blocks_1_8)

# ------------------------------------------------ Goal -------------------------------------------------------------
for i in range(rows):
    for j in range(columns):
        if (i,j) not in undefined:
            puzznic_scoring_problem.add_goal(Equals(patterned[i][j], F))

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
        score_blocks_2_6: maximum_cost - Int(20)
    })
if n_blocks >= 7:
    costs.update({score_blocks_1_7: maximum_cost - Int(12)})
if n_blocks >= 8:
    costs.update({score_blocks_1_8: maximum_cost - Int(20)})

puzznic_scoring_problem.add_quality_metric(MinimizeActionCosts(costs))

# -------------------------------------------------- Compilation -------------------------------------------------------
from unified_planning.engines import CompilationKind
if compilation == 'up':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
else:
    raise ValueError(f"Unsupported compilation type: {compilation}")

if solving == 'fast-downward-opt':
    compilation_kinds_to_apply.append(CompilationKind.CONDITIONAL_EFFECTS_REMOVING)

problem = puzznic_scoring_problem
results = []
for ck in compilation_kinds_to_apply:
    params = {}
    if ck == CompilationKind.ARRAYS_REMOVING:
        params = {'mode': 'permissive'}
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
up.shortcuts.get_environment().credits_stream = None

# ----------------------------------------------------- Solving --------------------------------------------------------

with OneshotPlanner(name=solving) as planner:
    result = planner.solve(problem)

    plan = result.plan
    if plan is not None:
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )

        total_score = maximum_cost
        for a in compiled_plan.actions:
            if str(a).split('_')[0] == 'score':
                score_action = puzznic_scoring_problem.action(str(a))
                for qm in puzznic_scoring_problem.quality_metrics:
                    total_score -= qm.get_action_cost(score_action).simplify()

        print(f'Steps: {len(compiled_plan.actions)}')
        print(f'Actions: {compiled_plan.actions}')
        print(f'Score: {total_score.simplify()}')

    else:
        print(result)

    if not planner.supports(problem.kind):
        unsupported_features = [
            f"{pk} is not supported by the planner"
            for pk in problem.kind.features if pk not in planner.supported_kind().features
        ]
        print("\n".join(unsupported_features))
