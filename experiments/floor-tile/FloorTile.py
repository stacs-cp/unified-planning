from experiments import compilation_solving
from unified_planning.shortcuts import *

compilation = 'up'
solving = 'fast-downward'

rows = 4
columns = 4
# A set of robots use different colors to paint patterns in floor tiles.
# The robots can move around the floor tiles in four directions (up, down, left, and right).
# Robots paint with one color at a time but can change their spray guns to any available color.
# However, robots can only paint the tile that is in front (up) and behind (down) them, and once a tile has been painted,
# no robot can stand on it.
# ---------------------------------------------------- Problem ---------------------------------------------------------
floor_tile = Problem('floor_tile')

Tile = UserType('Tile')
Robot = UserType('Robot', Tile)
Colour = UserType('Colour', Tile)

clear = Object('clear', Tile)
r1 = Object('r1', Robot)
r2 = Object('r2', Robot)
black = Object('black', Colour)
white = Object('white', Colour)

robot_has = Fluent('robot_has', r=Robot, c=Colour)
available_colour = Fluent('available_colour', c=Colour)

grid = Fluent('grid', ArrayType(rows, ArrayType(columns, Tile)))

floor_tile.add_fluent(robot_has, default_initial_value=False)
floor_tile.add_fluent(available_colour, default_initial_value=False)
floor_tile.add_fluent(grid, default_initial_value=clear)


change_colour = InstantaneousAction('change_colour', r=Robot, c=Colour, nc=Colour)
r = change_colour.parameter('r')
c = change_colour.parameter('c')
nc = change_colour.parameter('nc')
change_colour.add_precondition(robot_has(r, c))
change_colour.add_precondition(available_colour(nc))
change_colour.add_effect(robot_has(r, c), False)
change_colour.add_effect(robot_has(r, nc), False)
floor_tile.add_action(change_colour)

paint_up = InstantaneousAction('paint_up', r=Robot, c=Colour, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = paint_up.parameter('r')
c = paint_up.parameter('c')
x = paint_up.parameter('x')
y = paint_up.parameter('y')
paint_up.add_precondition(robot_has(r, c))
paint_up.add_precondition(Equals(grid(x, y), r))
paint_up.add_precondition(Equals(grid(x-1, y), clear))
paint_up.add_effect(grid(x-1, y), c)
floor_tile.add_action(paint_up)

paint_down = InstantaneousAction('paint_down', r=Robot, c=Colour, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = paint_down.parameter('r')
c = paint_down.parameter('c')
x = paint_down.parameter('x')
y = paint_down.parameter('y')
paint_down.add_precondition(robot_has(r, c))
paint_down.add_precondition(Equals(grid(x, y), r))
paint_down.add_precondition(Equals(grid(x+1, y), clear))
paint_down.add_effect(grid(x+1, y), c)
floor_tile.add_action(paint_down)

up = InstantaneousAction('up', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = up.parameter('r')
x = up.parameter('x')
y = up.parameter('y')
up.add_precondition(Equals(grid(x, y), r))
up.add_precondition(Equals(grid(x-1, y), clear))
up.add_effect(grid(x-1, y), c)
up.add_effect(grid(x, y), clear)
floor_tile.add_action(up)

down = InstantaneousAction('down', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = down.parameter('r')
x = down.parameter('x')
y = down.parameter('y')
down.add_precondition(Equals(grid(x, y), r))
down.add_precondition(Equals(grid(x+1, y), clear))
down.add_effect(grid(x+1, y), c)
down.add_effect(grid(x, y), clear)
floor_tile.add_action(down)

left = InstantaneousAction('left', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = left.parameter('r')
x = left.parameter('x')
y = left.parameter('y')
left.add_precondition(Equals(grid(x, y), r))
left.add_precondition(Equals(grid(x, y-1), clear))
left.add_effect(grid(x, y-1), c)
left.add_effect(grid(x, y), clear)
floor_tile.add_action(left)

right = InstantaneousAction('right', r=Robot, x=IntType(0, rows-1), y=IntType(0, columns-1))
r = right.parameter('r')
x = right.parameter('x')
y = right.parameter('y')
right.add_precondition(Equals(grid(x, y), r))
right.add_precondition(Equals(grid(x, y+1), clear))
right.add_effect(grid(x, y+1), c)
right.add_effect(grid(x, y), clear)
floor_tile.add_action(right)

for r in range(rows):
    for c in range(columns):
        floor_tile.add_goal(Not(Equals(grid[r][c], clear)))

costs: Dict[Action, Expression] = {
    change_colour: Int(5),
    paint_up: Int(2),
    paint_down: Int(2),
    up: Int(1),
    down: Int(1),
    left: Int(1),
    right: Int(1),
}
floor_tile.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(floor_tile, solving, compilation)