import argparse
import time
from experiments import compilation_solving
from unified_planning.shortcuts import *

from unified_planning.io import PDDLReader

domain_filename = 'domain.pddl'
problem_filename = 'p01.pddl'

reader = PDDLReader()
problem = reader.parse_problem(domain_filename, problem_filename)

params = {
    'fast_downward_alias': 'seq-sat-fdss-2',
    'fast_downward_search_time_limit': '300'
}
with OneshotPlanner(name='fast-downward', params=params) as planner:
    result = planner.solve(problem)
    plan = result.plan
    if plan is not None:
        print(plan)
        #compiled_plan_str = str(compiled_plan)
        #moves = sum(1 for line in compiled_plan_str.splitlines() if 'move_' in line)
    else:
        print('No solution found.')
        print(result)
    if not planner.supports(problem.kind):
        unsupported_features = [
            f"{pk} is not supported by the planner"
            for pk in problem.kind.features if pk not in planner.supported_kind().features
        ]
        print("\n".join(unsupported_features))

print(f'Planner Time:', time.time())

