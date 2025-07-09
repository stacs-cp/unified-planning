import argparse

from experiments import compilation_solving
from unified_planning.shortcuts import *

# Parser
parser = argparse.ArgumentParser(description="Solve Folding")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

dimension = 15
nodes = 8
# ---------------------------------------------------- Problem ---------------------------------------------------------
folding = Problem('folding')

rows = Fluent('rows', ArrayType(nodes, IntType(0, dimension-1)))
cols = Fluent('cols', ArrayType(nodes, IntType(0, dimension-1)))

folding.add_fluent(rows, default_initial_value=0)
folding.add_fluent(cols, default_initial_value=0)

#folding.set_initial_value(rows[0], 2)
#folding.set_initial_value(cols[0], 0)
#folding.set_initial_value(rows[1], 1)
#folding.set_initial_value(cols[1], 0)
#folding.set_initial_value(rows[2], 0)
#folding.set_initial_value(cols[2], 0)
folding.set_initial_value(rows[0], 7)
folding.set_initial_value(cols[0], 7)
folding.set_initial_value(rows[1], 6)
folding.set_initial_value(cols[1], 7)
folding.set_initial_value(rows[2], 5)
folding.set_initial_value(cols[2], 7)
folding.set_initial_value(rows[3], 4)
folding.set_initial_value(cols[3], 7)
folding.set_initial_value(rows[4], 3)
folding.set_initial_value(cols[4], 7)
folding.set_initial_value(rows[5], 2)
folding.set_initial_value(cols[5], 7)
folding.set_initial_value(rows[6], 1)
folding.set_initial_value(cols[6], 7)
folding.set_initial_value(rows[7], 0)
folding.set_initial_value(cols[7], 7)

rotate_clockwise = InstantaneousAction('rotate_clockwise', n=IntType(0, nodes-1))
n = rotate_clockwise.parameter('n')
#rotate_clockwise.add_precondition(And(
#    LE(x, folding.initial_value(rows[0]) + n),
#    GE(x, folding.initial_value(rows[0]) - n),
#))
#rotate_clockwise.add_precondition(And(
#    LE(y, folding.initial_value(cols[0]) + n),
#    GE(y, folding.initial_value(cols[0]) - n),
#))
g = RangeVariable('g', n+1, nodes-1)
b = RangeVariable('b', 0, n-1)
rotate_clockwise.add_precondition(Forall(
    Or(Not(Equals(rows[b], rows[n] - cols[n] + cols[g])),
       Not(Equals(cols[b], cols[n] + rows[n] - rows[g]))), g,b))
rotate_clockwise.add_effect(rows[g], rows[n] - cols[n] + cols[g], forall=[g])
rotate_clockwise.add_effect(cols[g], cols[n] + rows[n] - rows[g], forall=[g])
folding.add_action(rotate_clockwise)

rotate_counter_clockwise = InstantaneousAction('rotate_counter_clockwise', n=IntType(0, nodes-1))
n = rotate_counter_clockwise.parameter('n')
#rotate_counter_clockwise.add_precondition(And(
#    LE(x, folding.initial_value(rows[0]) + n),
#    GE(x, folding.initial_value(rows[0]) - n),
#))
#rotate_counter_clockwise.add_precondition(And(
#    LE(y, folding.initial_value(cols[0]) + n),
#    GE(y, folding.initial_value(cols[0]) - n),
#))
g = RangeVariable('g', n+1, nodes-1)
b = RangeVariable('b', 0, n-1)
rotate_counter_clockwise.add_precondition(Forall(
    Or(Not(Equals(rows[b], rows[n] + cols[n] - cols[g])),
       Not(Equals(cols[b], cols[n] - rows[n] + rows[g]))), g,b))
rotate_counter_clockwise.add_effect(rows[g], rows[n] + cols[n] - cols[g], forall=[g])
rotate_counter_clockwise.add_effect(cols[g], cols[n] - rows[n] + rows[g], forall=[g])
folding.add_action(rotate_counter_clockwise)

#folding.add_goal(Equals(rows[1], 2))
#folding.add_goal(Equals(cols[1], 1))
#folding.add_goal(Equals(rows[2], 2))
#folding.add_goal(Equals(cols[2], 2))
folding.add_goal(Equals(rows[0], 7))
folding.add_goal(Equals(cols[0], 7))
folding.add_goal(Equals(rows[1], 6))
folding.add_goal(Equals(cols[1], 7))
folding.add_goal(Equals(rows[2], 5))
folding.add_goal(Equals(cols[2], 7))
folding.add_goal(Equals(rows[3], 5))
folding.add_goal(Equals(cols[3], 8))
folding.add_goal(Equals(rows[4], 6))
folding.add_goal(Equals(cols[4], 8))
folding.add_goal(Equals(rows[5], 6))
folding.add_goal(Equals(cols[5], 9))
folding.add_goal(Equals(rows[6], 7))
folding.add_goal(Equals(cols[6], 9))
folding.add_goal(Equals(rows[7], 7))
folding.add_goal(Equals(cols[7], 10))

costs: Dict[Action, Expression] = {
    rotate_clockwise: Int(1),
    rotate_counter_clockwise: Int(1),
}
folding.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['integers', 'ut-integers', 'logarithmic'], f"Unsupported compilation type: {compilation}"

#compilation_solving.compile_and_solve(folding, solving, compilation)

COMPILATION_PIPELINES = {
    'up': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ],
    'integers': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
    ],
    'ut-integers': [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
}
problem = folding
compilation_kinds_to_apply = COMPILATION_PIPELINES[compilation]
results = []
for ck in compilation_kinds_to_apply:
    print(f'Compilation kind: {ck}')
    params = {}
    if ck == CompilationKind.ARRAYS_REMOVING:
        params = {'mode': 'permissive'}
    with Compiler(problem_kind=problem.kind, compilation_kind=ck, params=params) as compiler:
        result = compiler.compile(
            problem,
            ck
        )
        results.append(result)
        problem = result.problem

print(problem)