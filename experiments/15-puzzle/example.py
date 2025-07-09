from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve 15-Puzzle")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# ---------------------------------------------------- Problem ---------------------------------------------------------

#problem = unified_planning.model.Problem('npuzzle_problem')
#
#value = Fluent('value', IntType(0,1))
#problem.add_fluent(value)
#
#sum = unified_planning.model.InstantaneousAction('sum')
#sum.add_precondition(Equals(value, 0))
#sum.add_effect(value, 1)
#
#problem.add_action(sum)
#problem.add_goal(Equals(value, 1))
#
#print(problem)
#compilation_solving.compile_and_solve(problem, solving, compilation)

problem = unified_planning.model.Problem('npuzzle_problem')

array = Fluent("array", ArrayType(5, IntType(0,4)), undefined_positions=[(1)])
problem.add_fluent(array, default_initial_value=0)

increment = InstantaneousAction("increment", c=IntType(0,4))
c = increment.parameter("c")
i = RangeVariable("i", 0, c-1)
increment.add_precondition(Forall(LE(array[i], array[c]), i))

problem.add_action(increment)
problem.add_goal(Equals(array, [0,1,2,3,4]))

print(problem)
compilation_solving.compile_and_solve(problem, solving, compilation)