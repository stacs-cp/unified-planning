from experiments import compilation_solving
from unified_planning.io import PDDLReader

instance = 'plt0_2_4_2_1'
solving = 'fast-downward'

reader = PDDLReader()
domain_filename = f'handcrafted/domain.pddl'
problem_filename = f'handcrafted/{instance}.pddl'

problem = reader.parse_problem(domain_filename, problem_filename)

compilation_solving.compile_and_solve(problem, solving, compilation_kinds_to_apply=[])