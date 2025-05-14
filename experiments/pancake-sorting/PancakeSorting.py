from unified_planning.shortcuts import *
from unified_planning.engines import CompilationKind
from experiments import compilation_solving

compilation = 'up'
solving = 'fast-downward'

n = 5
instance = [3,4,2,1,0]

# ------------------------------------------------ Problem -------------------------------------------------------------

pancake_problem = Problem('pancake_problem')

Number = UserType('Number')
pancake = Fluent('pancake', ArrayType(n, Number))

for i in range(n):
    o = Object(f'n{instance[i]}', Number)
    pancake_problem.add_object(o)
    if i == 0:
        pancake_problem.add_fluent(pancake, default_initial_value=o)
    pancake_problem.set_initial_value(pancake[i], o)

flip = InstantaneousAction('flip', f=IntType(1,n-1))
f = flip.parameter('f')
b = RangeVariable('b', 0, f)
flip.add_effect(pancake[b], pancake[f - b], forall=[b])
pancake_problem.add_action(flip)

for i in range(n):
    pancake_problem.add_goal(Equals(pancake[i], pancake_problem.object(f'n{i}')))

costs: Dict[Action, Expression] = {
    flip: Int(1),
}
pancake_problem.add_quality_metric(MinimizeActionCosts(costs))

if compilation == 'up':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
else:
    raise ValueError(f"Unsupported compilation type: {compilation}")

compilation_solving.compile_and_solve(pancake_problem, solving, compilation_kinds_to_apply=compilation_kinds_to_apply)