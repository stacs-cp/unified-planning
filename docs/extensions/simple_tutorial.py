"""
This example illustrates how to define a planning problem using high-level constructs:
- arrays
- sets
- integer parameters in actions
- range variables
- count
This is intentionally simple and does not represent a real problem.
"""

# Imports
from unified_planning.shortcuts import *

# ------------------------------------------------------------
# Problem definition
# ------------------------------------------------------------
problem = Problem("simple_example")

# ------------------------------------------------------------
# Types and objects
# ------------------------------------------------------------
# User-defined type
Colour = UserType("Colour")

# Objects of type Colour
blue = Object("blue", Colour)
red = Object("red", Colour)
green = Object("green", Colour)

problem.add_objects([blue, red, green])

# ------------------------------------------------------------
# Fluents
# ------------------------------------------------------------
# A small 2x2 array of colours
grid = Fluent("grid", ArrayType(2, ArrayType(2, Colour)))
problem.add_fluent(grid, default_initial_value=red)

# A set of colours
my_colours = Fluent("my_colours", SetType(Colour))
problem.add_fluent(my_colours, default_initial_value=set())

# ----------------------------------------------
# Action with Integer parameters + RangeVariable
# ----------------------------------------------
# Swap a cell with its right neighbor for a given row 'r'
swap_row_right = InstantaneousAction("swap_row_right", r=IntType(0,1))
r = swap_row_right.parameter("r")

# Define a range variable over columns 0 and 1 (to swap col 0 with col 1, col 1 with col 2)
c = RangeVariable("c", 0, 1)

# Precondition: adjacent cells are different
swap_row_right.add_precondition(Not(Equals(grid[r][c], grid[r][c+1])), forall=[c])

# Effect: exchange colours
swap_row_right.add_effect(grid[r][c], grid[r][c+1], forall=[c])
swap_row_right.add_effect(grid[r][c+1], grid[r][c], forall=[c])

problem.add_action(swap_row_right)

# ------------------------------------------------------------
# Goal using Count
# ------------------------------------------------------------
# Goal: at most 1 cell is not blue
not_blue_cells = [Not(Equals(grid[i][j], blue)) for i in range(2) for j in range(2)]
problem.add_goal(LE(Count(not_blue_cells), 1))
