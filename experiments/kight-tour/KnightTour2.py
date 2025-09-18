from experiments import compilation_solving
from unified_planning.shortcuts import *

compilation = 'integers'
solving = 'enhsp'

rows = 4
columns = 4

# ---------------------------------------------------- Problem ---------------------------------------------------------
knight_problem = Problem('knight_problem')

Direction = UserType('Direction')
ur = Object('up-right', Direction)
ul = Object('up-left', Direction)
dr = Object('down-right', Direction)
dl = Object('down-left', Direction)
knight_problem.add_objects([ur, ul, dr, dl])

Status = UserType('Status')

visited = Fluent('visited', ArrayType(rows, ArrayType(columns, BoolType())))
at_row = Fluent('at_row', IntType(0, rows))
at_col = Fluent('at_col', IntType(0, columns))

knight_problem.add_fluent(visited, default_initial_value=False)
knight_problem.add_fluent(at_row, default_initial_value=0)
knight_problem.add_fluent(at_col, default_initial_value=0)

knight_problem.set_initial_value(at_row, 0)
knight_problem.set_initial_value(at_col, 3)
knight_problem.set_initial_value(visited[0][3], True)

move_2c_1r = InstantaneousAction('move_2c_1r', d=Direction, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_2c_1r.parameter('r')
c = move_2c_1r.parameter('c')
d = move_2c_1r.parameter('d')
move_2c_1r.add_precondition(Equals(at_row, r))
move_2c_1r.add_precondition(Equals(at_col, c))
move_2c_1r.add_precondition(And(
    Implies(Equals(d, ur), Not(visited[r-1][c+2])),
    Implies(Equals(d, ul), Not(visited[r-1][c-2])),
    Implies(Equals(d, dr), Not(visited[r+1][c+2])),
    Implies(Equals(d, dl), Not(visited[r+1][c-2])),
))
move_2c_1r.add_effect(visited[r][c], True)
move_2c_1r.add_effect(at_row, r-1, Or(Equals(d, ur), Equals(d, ul)))
move_2c_1r.add_effect(at_row, r+1, Or(Equals(d, dr), Equals(d, dl)))
move_2c_1r.add_effect(at_col, c-2, Or(Equals(d, ul), Equals(d, dl)))
move_2c_1r.add_effect(at_col, c+2, Or(Equals(d, ur), Equals(d, dr)))
knight_problem.add_action(move_2c_1r)

move_1c_2r = InstantaneousAction('move_1c_2r', d=Direction, r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_1c_2r.parameter('r')
c = move_1c_2r.parameter('c')
d = move_1c_2r.parameter('d')
move_1c_2r.add_precondition(Equals(at_row, r))
move_1c_2r.add_precondition(Equals(at_col, c))
move_1c_2r.add_precondition(And(
    Implies(Equals(d, ur), Not(visited[r-2][c+1])),
    Implies(Equals(d, ul), Not(visited[r-2][c-1])),
    Implies(Equals(d, dr), Not(visited[r+2][c+1])),
    Implies(Equals(d, dl), Not(visited[r+2][c-1])),
))
move_1c_2r.add_effect(visited[r][c], True)
move_1c_2r.add_effect(at_row, r-2, Or(Equals(d, ur), Equals(d, ul)))
move_1c_2r.add_effect(at_row, r+2, Or(Equals(d, dr), Equals(d, dl)))
move_1c_2r.add_effect(at_col, c-1, Or(Equals(d, ul), Equals(d, dl)))
move_1c_2r.add_effect(at_col, c+1, Or(Equals(d, ur), Equals(d, dr)))
knight_problem.add_action(move_1c_2r)


for c in range(columns):
    knight_problem.add_goal(visited[0][c])

costs: Dict[Action, Expression] = {
    move_2c_1r: Int(1),
    move_1c_2r: Int(1),
}
knight_problem.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up', 'integers'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(knight_problem, solving, compilation)