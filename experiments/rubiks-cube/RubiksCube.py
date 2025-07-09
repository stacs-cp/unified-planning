from experiments import compilation_solving
from unified_planning.shortcuts import *

compilation = 'up'
solving = 'fast-downward'

rows = 6
columns = 6
# ---------------------------------------------------- Problem ---------------------------------------------------------
rubiks_cube = Problem('rubiks_cube')



costs: Dict[Action, Expression] = {

}
rubiks_cube.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(rubiks_cube, solving, compilation)