from experiments import compilation_solving
from unified_planning.shortcuts import *
import argparse

# Parser
parser = argparse.ArgumentParser(description="Solve Rush Hour")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# Example 15Puzzle
instance = 'BBBCCLIDDKoLIAAKoMooJEEMFFJoooGGGoox'
undefined = [(5,5)]
columns = 6
rows = 6

# ---------------------------------------------------- Problem ---------------------------------------------------------

rush_hour_problem = unified_planning.model.Problem('rush_hour_problem')

Vehicle = UserType('Vehicle')
none = Object('none', Vehicle)
A = Object('A', Vehicle)
rush_hour_problem.add_objects([none, A])
occupied = Fluent('occupied', ArrayType(rows, ArrayType(columns, Vehicle)), undefined_positions=undefined)
is_car = Fluent('is_car', v=Vehicle)
rush_hour_problem.add_fluent(occupied, default_initial_value=none)
rush_hour_problem.add_fluent(is_car, default_initial_value=False)
rush_hour_problem.set_initial_value(is_car(A), True)

for i, char in enumerate(instance):
    r, c = divmod(i, columns)
    if char == 'o':
        rush_hour_problem.set_initial_value(occupied[r][c], none)
    elif char != 'x':
        obj = Object(f'{char}', Vehicle)
        if not rush_hour_problem.has_object(char):
            rush_hour_problem.add_object(obj)
            rush_hour_problem.set_initial_value(is_car(obj), instance.count(char) == 2)
        rush_hour_problem.set_initial_value(occupied[r][c], obj)

# ------------------------------------------------ Move Car Right ------------------------------------------------------
move_car_right = InstantaneousAction('move_car_right', v=Vehicle, r=IntType(0,rows-1),
                                     c=IntType(0,columns-1))
v = move_car_right.parameter('v')
r = move_car_right.parameter('r')
c = move_car_right.parameter('c')
# vehicle is not none
move_car_right.add_precondition(Not(Equals(v,none)))
# is a car
move_car_right.add_precondition(is_car(v))
# it is located in the position
move_car_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
# next position must be empty
move_car_right.add_precondition(Equals(occupied[r][c+2], none))

# control effects occupied positions
move_car_right.add_effect(occupied[r][c], none)
move_car_right.add_effect(occupied[r][c+2], v)

# ------------------------------------------------- Move Car Left ------------------------------------------------------
move_car_left = InstantaneousAction('move_car_left', v=Vehicle, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_car_left.parameter('v')
r = move_car_left.parameter('r')
c = move_car_left.parameter('c')
# vehicle is not none
move_car_left.add_precondition(Not(Equals(v,none)))
# is a car
move_car_left.add_precondition(is_car(v))
# is horizontal and it is located in the position
move_car_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
# next position must be empty
move_car_left.add_precondition(Equals(occupied[r][c-1], none))

# control effects occupied positions
move_car_left.add_effect(occupied[r][c-1], v)
move_car_left.add_effect(occupied[r][c+1], none)

# ------------------------------------------------- Move Car Up ------------------------------------------------------
move_car_up = InstantaneousAction('move_car_up', v=Vehicle, r=IntType(0,rows-1),
                                  c=IntType(0,columns-1))
v = move_car_up.parameter('v')
r = move_car_up.parameter('r')
c = move_car_up.parameter('c')
# vehicle is not none
move_car_up.add_precondition(Not(Equals(v,none)))
# is a car
move_car_up.add_precondition(is_car(v))
# is horizontal and it is located in the position
move_car_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
# next position must be empty
move_car_up.add_precondition(Equals(occupied[r-1][c], none))

# control effects occupied positions
move_car_up.add_effect(occupied[r-1][c], v)
move_car_up.add_effect(occupied[r+1][c], none)

# ------------------------------------------------- Move Car Down ------------------------------------------------------
move_car_down = InstantaneousAction('move_car_down', v=Vehicle, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_car_down.parameter('v')
r = move_car_down.parameter('r')
c = move_car_down.parameter('c')
# vehicle is not none
move_car_down.add_precondition(Not(Equals(v,none)))
# is a car
move_car_down.add_precondition(is_car(v))
# is horizontal and it is located in the position
move_car_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
# next position must be empty
move_car_down.add_precondition(Equals(occupied[r+2][c], none))

# control effects occupied positions
move_car_down.add_effect(occupied[r][c], none)
move_car_down.add_effect(occupied[r+2][c], v)

# ------------------------------------------------ Move Truck Right ------------------------------------------------------
move_truck_right = InstantaneousAction('move_truck_right', v=Vehicle, r=IntType(0,rows-1),
                                       c=IntType(0,columns-1))
v = move_truck_right.parameter('v')
r = move_truck_right.parameter('r')
c = move_truck_right.parameter('c')
# vehicle is not none
move_truck_right.add_precondition(Not(Equals(v,none)))
# is a truck
move_truck_right.add_precondition(Not(is_car(v)))
# it is located in the position
move_truck_right.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
# next position must be empty
move_truck_right.add_precondition(Equals(occupied[r][c+3], none))

# control effects occupied positions
move_truck_right.add_effect(occupied[r][c], none)
move_truck_right.add_effect(occupied[r][c+3], v)

# ------------------------------------------------ Move Truck Left ------------------------------------------------------
move_truck_left = InstantaneousAction('move_truck_left', v=Vehicle, r=IntType(0,rows-1),
                                      c=IntType(0,columns-1))
v = move_truck_left.parameter('v')
r = move_truck_left.parameter('r')
c = move_truck_left.parameter('c')
# vehicle is not none
move_truck_left.add_precondition(Not(Equals(v,none)))
# is a truck
move_truck_left.add_precondition(Not(is_car(v)))
# it is located in the position
move_truck_left.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
# next position must be empty
move_truck_left.add_precondition(Equals(occupied[r][c-1], none))

# control effects occupied positions
move_truck_left.add_effect(occupied[r][c+2], none)
move_truck_left.add_effect(occupied[r][c-1], v)

# ------------------------------------------------- Move Truck Up ------------------------------------------------------
move_truck_up = InstantaneousAction('move_truck_up', v=Vehicle, r=IntType(0,rows-1),
                                    c=IntType(0,columns-1))
v = move_truck_up.parameter('v')
r = move_truck_up.parameter('r')
c = move_truck_up.parameter('c')
# vehicle is not none
move_truck_up.add_precondition(Not(Equals(v,none)))
# is a truck
move_truck_up.add_precondition(Not(is_car(v)))
# is horizontal and it is located in the position
move_truck_up.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v),
                                   Equals(occupied[r+2][c], v)))
# next position must be empty
move_truck_up.add_precondition(Equals(occupied[r-1][c], none))

# control effects occupied positions
move_truck_up.add_effect(occupied[r+2][c], none)
move_truck_up.add_effect(occupied[r-1][c], v)

# ------------------------------------------------- Move Truck Down ------------------------------------------------------
move_truck_down = InstantaneousAction('move_truck_down', v=Vehicle, r=IntType(0,rows-1),
                                      c=IntType(0,columns-1))
v = move_truck_down.parameter('v')
r = move_truck_down.parameter('r')
c = move_truck_down.parameter('c')
# vehicle is not none
move_truck_down.add_precondition(Not(Equals(v,none)))
# is a truck
move_truck_down.add_precondition(Not(is_car(v)))
# is horizontal and it is located in the position
move_truck_down.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v),
                                     Equals(occupied[r+2][c], v)))
# next position must be empty
move_truck_down.add_precondition(Equals(occupied[r+3][c], none))

# control effects occupied positions
move_truck_down.add_effect(occupied[r][c], none)
move_truck_down.add_effect(occupied[r+3][c], v)


rush_hour_problem.add_actions([move_car_right, move_car_left, move_car_down, move_car_up,
                               move_truck_right, move_truck_left, move_truck_down, move_truck_up])

# ----------------------------------------------------- Goal -----------------------------------------------------------
rush_hour_problem.add_goal(Equals(occupied[2][4], A))
rush_hour_problem.add_goal(Equals(occupied[2][5], A))

# ------------------------------------------------- Action Costs -------------------------------------------------------
costs: Dict[Action, Expression] = {
    move_car_left: Int(1),
    move_car_right: Int(1),
    move_car_up: Int(1),
    move_car_down: Int(1),
    move_truck_right: Int(1),
    move_truck_left: Int(1),
    move_truck_up: Int(1),
    move_truck_down: Int(1),
}
rush_hour_problem.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(rush_hour_problem, solving, compilation)