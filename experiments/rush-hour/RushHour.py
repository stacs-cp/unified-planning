from unified_planning.shortcuts import *
import time
start = time.time()

rush_hour_problem = Problem('rush_hour_problem')

rows = 6
columns = 6

Vehicle = UserType('Vehicle')
none = Object('none', Vehicle)
A = Object('A', Vehicle)
rush_hour_problem.add_objects([none, A])

occupied = Fluent('occupied', ArrayType(rows, ArrayType(columns, Vehicle)))
is_car = Fluent('is_car', v=Vehicle)

rush_hour_problem.add_fluent(is_car, default_initial_value=True)
rush_hour_problem.set_initial_value(is_car(A), True)

grid = 'BBBKLMHCCKLMH.AALMDDJ....IJEE..IFFGG'
for i, char in enumerate(grid):
    r, c = divmod(i, columns)
    if char == '.':
        rush_hour_problem.set_initial_value(occupied[r][c], none)
    else:
        obj = Object(f'{char}', Vehicle)
        if not rush_hour_problem.has_object(char):
            rush_hour_problem.add_object(obj)
            rush_hour_problem.set_initial_value(is_car(obj), grid.count(char) == 2)
        rush_hour_problem.set_initial_value(occupied[r][c], obj)

# ----------------------- Move Horizontal Car ----------------------------
move_horizontal_car = unified_planning.model.InstantaneousAction('move_horizontal_car', v=Vehicle, r=IntType(0,rows-1), c=IntType(0,columns-2), m=IntType(-(columns-2), columns-2))
v = move_horizontal_car.parameter('v')
r = move_horizontal_car.parameter('r')
c = move_horizontal_car.parameter('c')
m = move_horizontal_car.parameter('m')
# vehicle is not none
move_horizontal_car.add_precondition(Not(Equals(v,none)))
# is a car
move_horizontal_car.add_precondition(is_car(v))
# is horizontal and it is located in the position
move_horizontal_car.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v)))
# control of the movements - positions within the movement have to be empty
for p in range(-(columns-2), columns-1):
    if p > 0:
        move_horizontal_car.add_precondition(Or(GT(p, m), Equals(occupied[r][c+1+p], none)))
    elif p < 0:
        move_horizontal_car.add_precondition(Or(LT(p, m), Equals(occupied[r][c+p], none)))

# control effects occupied positions
move_horizontal_car.add_effect(occupied[r][c+m], v, Not(Equals(m, 1)))
move_horizontal_car.add_effect(occupied[r][c+m+1], v, Not(Equals(m, -1)))
move_horizontal_car.add_effect(occupied[r][c], none, Not(Equals(m, -1)))
move_horizontal_car.add_effect(occupied[r][c+1], none, Not(Equals(m, 1)))

# ----------------------- Move Horizontal Truck ----------------------------
move_horizontal_truck = unified_planning.model.InstantaneousAction('move_horizontal_truck', v=Vehicle, r=IntType(0,rows-1), c=IntType(0,columns-3), m=IntType(-(columns-3), columns-3))
v = move_horizontal_truck.parameter('v')
r = move_horizontal_truck.parameter('r')
c = move_horizontal_truck.parameter('c')
m = move_horizontal_truck.parameter('m')
# vehicle is not none
move_horizontal_truck.add_precondition(Not(Equals(v,none)))
# is a truck
move_horizontal_truck.add_precondition(Not(is_car(v)))
# is horizontal and it is located in the position
move_horizontal_truck.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r][c+1], v), Equals(occupied[r][c+2], v)))
# control of the movements - positions within the movement have to be empty
for p in range(-(columns-3), columns-2):
    if p > 0:
        move_horizontal_truck.add_precondition(Or(GT(p, m), Equals(occupied[r][c+2+p], none)))
    elif p < 0:
        move_horizontal_truck.add_precondition(Or(LT(p, m), Equals(occupied[r][c+p], none)))

# control effects occupied positions
move_horizontal_truck.add_effect(occupied[r][c+m], v, Or(LT(m, 0), GT(m, 2)))
move_horizontal_truck.add_effect(occupied[r][c+m+1], v, Or(LT(m, -1), GT(m, 1)))
move_horizontal_truck.add_effect(occupied[r][c+m+2], v, Or(LT(m, -2), GT(m, 0)))
move_horizontal_truck.add_effect(occupied[r][c], none, Or(LT(m, -2), GT(m, 0)))
move_horizontal_truck.add_effect(occupied[r][c+1], none, Or(LT(m, -1), GT(m, 1)))
move_horizontal_truck.add_effect(occupied[r][c+2], none, Or(LT(m, 0), GT(m, 2)))

# ----------------------- Move Vertical Car ----------------------------
move_vertical_car = unified_planning.model.InstantaneousAction('move_vertical_car', v=Vehicle, r=IntType(0,rows-2), c=IntType(0,columns-1), m=IntType(-(rows-2), rows-2))
v = move_vertical_car.parameter('v')
r = move_vertical_car.parameter('r')
c = move_vertical_car.parameter('c')
m = move_vertical_car.parameter('m')
# vehicle is not none
move_vertical_car.add_precondition(Not(Equals(v,none)))
# is a car
move_vertical_car.add_precondition(is_car(v))
# is horizontal and it is located in the position
move_vertical_car.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v)))
# control of the movements - positions within the movement have to be empty
for p in range(-(rows-2), rows-1):
    if p > 0:
        move_vertical_car.add_precondition(Or(GT(p, m), Equals(occupied[r+1+p][c], none)))
    elif p < 0:
        move_vertical_car.add_precondition(Or(LT(p, m), Equals(occupied[r+p][c], none)))

# control effects occupied positions
move_vertical_car.add_effect(occupied[r+m][c], v, Not(Equals(m, 1)))
move_vertical_car.add_effect(occupied[r+m+1][c], v, Not(Equals(m, -1)))
move_vertical_car.add_effect(occupied[r][c], none, Not(Equals(m, -1)))
move_vertical_car.add_effect(occupied[r+1][c], none, Not(Equals(m, 1)))

# ----------------------- Move Vertical Truck ----------------------------
move_vertical_truck = unified_planning.model.InstantaneousAction('move_vertical_truck', v=Vehicle, r=IntType(0,rows-3), c=IntType(0,columns-1), m=IntType(-(rows-3), rows-3))
v = move_vertical_truck.parameter('v')
r = move_vertical_truck.parameter('r')
c = move_vertical_truck.parameter('c')
m = move_vertical_truck.parameter('m')
# vehicle is not none
move_vertical_truck.add_precondition(Not(Equals(v,none)))
# is a truck
move_vertical_truck.add_precondition(Not(is_car(v)))
# is horizontal and it is located in the position
move_vertical_truck.add_precondition(And(Equals(occupied[r][c], v), Equals(occupied[r+1][c], v), Equals(occupied[r+2][c], v)))
# control of the movements - positions within the movement have to be empty
for p in range(-(rows-3), rows-2):
    if p > 0:
        move_vertical_truck.add_precondition(Or(GT(p, m), Equals(occupied[r+2+p][c], none)))
    elif p < 0:
        move_vertical_truck.add_precondition(Or(LT(p, m), Equals(occupied[r+p][c], none)))

# control effects occupied positions
move_vertical_truck.add_effect(occupied[r+m][c], v, Or(LT(m, 0), GT(m, 2)))
move_vertical_truck.add_effect(occupied[r+m+1][c], v, Or(LT(m, -1), GT(m, 1)))
move_vertical_truck.add_effect(occupied[r+m+2][c], v, Or(LT(m, -2), GT(m, 0)))
move_vertical_truck.add_effect(occupied[r][c], none, Or(LT(m, -2), GT(m, 0)))
move_vertical_truck.add_effect(occupied[r+1][c], none, Or(LT(m, -1), GT(m, 1)))
move_vertical_truck.add_effect(occupied[r+2][c], none, Or(LT(m, 0), GT(m, 2)))

rush_hour_problem.add_actions([move_horizontal_car, move_horizontal_truck, move_vertical_car, move_vertical_truck])

rush_hour_problem.add_goal(Equals(occupied[2][4], A))
rush_hour_problem.add_goal(Equals(occupied[2][5], A))

from unified_planning.engines import CompilationKind
# The CompilationKind class is defined in the unified_planning/engines/mixins/compiler.py file
compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    CompilationKind.ARRAYS_REMOVING,
    CompilationKind.USERTYPE_FLUENTS_REMOVING,
]

problem = rush_hour_problem
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
mid = time.time()
print("Preprocessing", mid - start)

with OneshotPlanner(name='enhsp-opt') as planner:
    version = planner.version()
    print(version)
    result = planner.solve(problem)
    plan = result.plan
    if plan is None:
        print("No plan found.")
    else:
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )
        print("Compiled plan: ", compiled_plan.actions)