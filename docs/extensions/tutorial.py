"""
Example demonstrating high-level constructs for planning:
- Arrays: multi-dimensional state representation
- Integer parameters: dynamic action instantiation
- Range variables: dynamic quantification dependent on parameters
- Set fluents: dynamic color tracking
- Count expressions: cardinality constraints on goals

This problem models painting a 3x3 grid where:
- Each cell starts blue by default
- Action paints rows 0..up_to with the same color
- Range variables iterate over a determined range
- Set tracks which colors have been used
- Goal: use at least 2 different colors
"""

from unified_planning.shortcuts import *

problem = Problem("grid_painting")

# ===== TYPES AND OBJECTS =====
Color = UserType("Color")
blue = Object("blue", Color)
red = Object("red", Color)
green = Object("green", Color)

problem.add_objects([blue, red, green])

# ===== FLUENTS =====
# 3x3 grid of colors (default: blue)
grid = Fluent("grid", ArrayType(3, ArrayType(3, Color)))
problem.add_fluent(grid, default_initial_value=blue)

# Set tracking which colors have been used
colors_used = Fluent("colors_used", SetType(Color))
problem.add_fluent(colors_used, default_initial_value=set())

# ===== ACTION: Paint Multiple Rows =====
# Paint rows from 0 to 'up_to' (inclusive) with the same color
# This demonstrates range variables dependent on integer parameters
paint_rows = InstantaneousAction("paint_rows", r_up_to=IntType(0, 2), color=Color)
r_up_to = paint_rows.parameter("r_up_to")
color = paint_rows.parameter("color")

# Range variable: rows from 0 to up_to (parameter-dependent)
r = RangeVariable("r", 0, r_up_to)
# Range variable: columns in each row
c = RangeVariable("c", 0, 2)

# Precondition: at least one cell in the range differs from target color
paint_rows.add_precondition(Exists(Not(Equals(grid[r][c], color)), c, r))

# Effects:
# - Paint all cells in rows 0..up_to with the color
paint_rows.add_effect(grid[r][c], color, forall=[r, c])

# - Track that this color has been used
paint_rows.add_effect(colors_used, SetAdd(color, colors_used))

problem.add_action(paint_rows)

# ===== GOALS =====
# Goal: use at least 2 different colors (count over set membership)
problem.add_goal(
    GE(
        Count([SetMember(blue, colors_used), SetMember(red, colors_used), SetMember(green, colors_used)]),
        2
    )
)

print(problem)