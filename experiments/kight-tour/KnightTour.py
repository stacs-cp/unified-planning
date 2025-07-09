from experiments import compilation_solving
from unified_planning.shortcuts import *

compilation = 'up'
solving = 'fast-downward'

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

visited = Fluent('visited', ArrayType(rows, ArrayType(columns)))
at_row = Fluent('at_row', IntType(0, rows))
at_col = Fluent('at_col', IntType(0, columns))

knight_problem.add_fluent(visited, default_initial_value=False)
knight_problem.add_fluent(at_row, default_initial_value=0)
knight_problem.add_fluent(at_col, default_initial_value=0)

knight_problem.set_initial_value(at_row, 0)
knight_problem.set_initial_value(at_col, 3)
knight_problem.set_initial_value(visited[0][3], True)

# ----------------------------------------------- 2 columns 1 row -----------------------------------------------
move_2c_1r_ur = InstantaneousAction('move_2c_1r_ur', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_2c_1r_ur.parameter('r')
c = move_2c_1r_ur.parameter('c')
move_2c_1r_ur.add_precondition(Equals(at_row, r))
move_2c_1r_ur.add_precondition(Equals(at_col, c))
move_2c_1r_ur.add_precondition(Not(visited[r-1][c+2]))
move_2c_1r_ur.add_effect(visited[r][c], True)
move_2c_1r_ur.add_effect(at_row, r-1)
move_2c_1r_ur.add_effect(at_col, c+2)
knight_problem.add_action(move_2c_1r_ur)


move_2c_1r_ul = InstantaneousAction('move_2c_1r_ul', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_2c_1r_ul.parameter('r')
c = move_2c_1r_ul.parameter('c')
move_2c_1r_ul.add_precondition(Equals(at_row, r))
move_2c_1r_ul.add_precondition(Equals(at_col, c))
move_2c_1r_ul.add_precondition(Not(visited[r-1][c-2]))
move_2c_1r_ul.add_effect(visited[r][c], True)
move_2c_1r_ul.add_effect(at_row, r-1)
move_2c_1r_ul.add_effect(at_col, c-2)
knight_problem.add_action(move_2c_1r_ul)

move_2c_1r_dr = InstantaneousAction('move_2c_1r_dr', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_2c_1r_dr.parameter('r')
c = move_2c_1r_dr.parameter('c')
move_2c_1r_dr.add_precondition(Equals(at_row, r))
move_2c_1r_dr.add_precondition(Equals(at_col, c))
move_2c_1r_dr.add_precondition(Not(visited[r+1][c+2]))
move_2c_1r_dr.add_effect(visited[r][c], True)
move_2c_1r_dr.add_effect(at_row, r+1)
move_2c_1r_dr.add_effect(at_col, c+2)
knight_problem.add_action(move_2c_1r_dr)

move_2c_1r_dl = InstantaneousAction('move_2c_1r_dl', r=IntType(0, rows - 1), c=IntType(0, columns - 1))
r = move_2c_1r_dl.parameter('r')
c = move_2c_1r_dl.parameter('c')
move_2c_1r_dl.add_precondition(Equals(at_row, r))
move_2c_1r_dl.add_precondition(Equals(at_col, c))
move_2c_1r_dl.add_precondition(Not(visited[r+1][c-2]))
move_2c_1r_dl.add_effect(visited[r][c], True)
move_2c_1r_dl.add_effect(at_row, r+1)
move_2c_1r_dl.add_effect(at_col, c-2)
knight_problem.add_action(move_2c_1r_dl)

# ----------------------------------------------- 2 rows 1 column -----------------------------------------------



for r in range(rows):
    for c in range(columns):
        knight_problem.add_goal(visited[r][c])

costs: Dict[Action, Expression] = {
    move_2c_1r_dl: Int(1),
    move_2c_1r_dr: Int(1),
    move_2c_1r_ul: Int(1),
    move_2c_1r_ur: Int(1),
}
knight_problem.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(knight_problem, solving, compilation)