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

# ------------------------------------------------ Problem -------------------------------------------------------------
puzznic_problem = Problem('puzznic_problem')

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

puzznic_problem.add_object(F)

patterned = Fluent('patterned', ArrayType(rows, ArrayType(columns, Pattern)), undefined_positions=undefined)
puzznic_problem.add_fluent(patterned, default_initial_value=F)

for l, p in initial_state.items():
    if not puzznic_problem.has_object(p):
        puzznic_problem.add_object(eval(p))
    puzznic_problem.set_initial_value(patterned[l[0]][l[1]], eval(initial_state[l]))

falling_flag = Fluent('falling_flag', DerivedBoolType())
puzznic_problem.add_fluent(falling_flag, default_initial_value=False)
matching_flag = Fluent('matching_flag', DerivedBoolType())
puzznic_problem.add_fluent(matching_flag, default_initial_value=False)

# ------------------------------------------------ Axioms -------------------------------------------------------------
# Falling
axiom_falling = Axiom('axiom_falling')
axiom_falling.set_head(falling_flag)
i = RangeVariable('i', 1, rows-1)
j = RangeVariable('j', 0, columns-1)
axiom_falling.add_body_condition(
    Exists(And(Not(Equals(patterned[i-1][j], F)), Equals(patterned[i][j], F)), i,j)
)
puzznic_problem.add_axiom(axiom_falling)

# Matching
axiom_matching = Axiom('axiom_matching')
axiom_matching.set_head(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-2)
matching_horizontal = Exists(
    And(Equals(patterned[i][j], patterned[i][j + 1]), Not(Equals(patterned[i][j], F))), i,j
)
i = RangeVariable('i', 0, rows-2)
j = RangeVariable('j', 0, columns-1)
matching_vertical = Exists(
    And(Equals(patterned[i][j], patterned[i + 1][j]), Not(Equals(patterned[i][j], F))), i,j
)
axiom_matching.add_body_condition(
    Or(matching_horizontal, matching_vertical)
)
puzznic_problem.add_axiom(axiom_matching)

# ------------------------------------------------ Actions -------------------------------------------------------------
# Move Block
move_block = InstantaneousAction('move_block', r=IntType(0, rows - 1), c=IntType(0, columns - 1),
                                 m=IntType(-1, 1))
r = move_block.parameter('r')
c = move_block.parameter('c')
m = move_block.parameter('m')
move_block.add_precondition(Not(falling_flag))
move_block.add_precondition(Not(matching_flag))
move_block.add_precondition(Not(Equals(m, 0)))
move_block.add_precondition(Not(Equals(patterned[r][c], F)))
move_block.add_precondition(Equals(patterned[r][c + m], F))
move_block.add_effect(patterned[r][c], F)
move_block.add_effect(patterned[r][c + m], patterned[r][c])
puzznic_problem.add_action(move_block)

# Fall Block
fall_block = InstantaneousAction('fall_block', r=IntType(0, rows - 2), c=IntType(0, columns - 1))
r = fall_block.parameter('r')
c = fall_block.parameter('c')
fall_block.add_precondition(falling_flag)
fall_block.add_precondition(Not(Equals(patterned[r][c], F)))
fall_block.add_precondition(Equals(patterned[r + 1][c], F))
fall_block.add_effect(patterned[r][c], F)
fall_block.add_effect(patterned[r + 1][c], patterned[r][c])
puzznic_problem.add_action(fall_block)

# Match Blocks
matching_blocks = InstantaneousAction('matching_blocks')
matching_blocks.add_precondition(Not(falling_flag))
matching_blocks.add_precondition(matching_flag)
i = RangeVariable('i', 0, rows-1)
j = RangeVariable('j', 0, columns-1)
matching_blocks.add_effect(patterned[i][j], F, condition=And(
    Not(Equals(patterned[i][j], F)),
    Or(Equals(patterned[i + 1][j], patterned[i][j]),
       Equals(patterned[i - 1][j], patterned[i][j]),
       Equals(patterned[i][j + 1], patterned[i][j]),
       Equals(patterned[i][j - 1], patterned[i][j]))
), forall=[i,j])
puzznic_problem.add_action(matching_blocks)

# ------------------------------------------------ Goal -------------------------------------------------------------
for i in range(rows):
    for j in range(columns):
        if (i,j) not in undefined:
            puzznic_problem.add_goal(Equals(patterned[i][j], F))

# ------------------------------------------------ Costs -------------------------------------------------------------
costs: Dict[Action, Expression] = {
    move_block: Int(1),
    matching_blocks: Int(0),
    fall_block: Int(0)
}
puzznic_problem.add_quality_metric(MinimizeActionCosts(costs))

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