from unified_planning.shortcuts import *
import time

start = time.time()

plotting_problem = unified_planning.model.Problem('plotting_problem')
Colour = UserType('Colour')
R = Object('R', Colour)
B = Object('B', Colour)
G = Object('G', Colour)
Y = Object('Y', Colour)
V = Object('V', Colour)
O = Object('O', Colour)
W = Object('W', Colour)
N = Object('N', Colour)
plotting_problem.add_objects([R,B,G,V,O,Y,W,N])

initial_blocks = [[R,R,R,R,R,R],[G,R,B,B,G,B],[B,G,G,G,B,G],[B,B,B,B,B,G],[G,R,R,B,G,G]]
rows = len(initial_blocks)
columns = len(initial_blocks[0])

blocks = Fluent('blocks', ArrayType(rows, ArrayType(columns, Colour)))

lc = columns-1
lr = rows-1
hand = Fluent('hand', Colour)
plotting_problem.add_fluent(blocks)
plotting_problem.add_fluent(hand, default_initial_value = W)
plotting_problem.set_initial_value(blocks, initial_blocks)

################################################################################
#                                 ACTIONS                                      #
################################################################################

################################ SHOOT PARTIAL ROW #####################################
shoot_partial_row = unified_planning.model.InstantaneousAction('shoot_partial_row', p=Colour, r=IntType(0, rows-1), l=IntType(0, columns-2))
# p: colour of the blocks that we want to eliminate
# r: row
# l: last column affected (cannot be the last one) - next block is different than p and not none
p = shoot_partial_row.parameter('p')
r = shoot_partial_row.parameter('r')
l = shoot_partial_row.parameter('l')

      # Preconditions
# 'p' is never the wildcard or none
shoot_partial_row.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_partial_row.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
# The following block from the last 'l' is different than 'p' and not none
shoot_partial_row.add_precondition(Not(Or(Equals(blocks[r][l+1], p), Equals(blocks[r][l+1], N))))
same_colour = []
# All the blocks in the row 'r' before the last 'l' are none or 'p'
for c in range(0, columns-1):
  shoot_partial_row.add_precondition(Or(GT(c,l), Equals(blocks[r][c], p), Equals(blocks[r][c], N)))
  same_colour.append(And(Equals(blocks[r][c], p), LE(c,l)))
# At least one of the blocks of the row is 'p'
shoot_partial_row.add_precondition(Or(same_colour))

      # Effects
# The hand saves the colour of the following block from the last 'l'
shoot_partial_row.add_effect(hand, blocks[r][l+1])
# The following block from the last 'l' saves the colour of 'p'
shoot_partial_row.add_effect(blocks[r][l+1], p)
# Each block in the row 'r' until the last 'l' become its above block, except the ones in the first row, that become none
for c in range(0, columns-1):
  shoot_partial_row.add_effect(blocks[0][c], N, LE(c,l))
  for a in range(1, rows):
    shoot_partial_row.add_effect(blocks[a][c], blocks[a-1][c], And(LE(c,l), LE(a,r)))

plotting_problem.add_action(shoot_partial_row)


############################### SHOOT FULL ROW #################################
# shoot complete row and then going down until the next is not p and not none
shoot_full_row = unified_planning.model.InstantaneousAction('shoot_full_row', p=Colour, r=IntType(0, rows-2), l=IntType(0, rows-2))
# p: colour of the blocks that we want to eliminate
# r: complete row affected
# l: last row of the last column affected (can't be the last one) next block is different than p and not none
p = shoot_full_row.parameter('p')
r = shoot_full_row.parameter('r')
l = shoot_full_row.parameter('l')

    # Preconditions
# 'p' is never the wildcard or none
shoot_full_row.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_full_row.add_precondition(Or(Equals(p, hand), Equals(hand, W)))
# All the blocks of the row 'r' are none or 'p' and at least 1 is p
for c in range(0, columns-1):
  shoot_full_row.add_precondition(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)))
# The following block from the last 'l' is different to 'p' and not none --- l+1 CONTROLAT ALS PARAMETRES
shoot_full_row.add_precondition(Not(Or(Equals(blocks[l+1][lc], p), Equals(blocks[l+1][lc], N))))
# The blocks between 'r' and 'l' are 'p' or none
for b in range(0, rows-1):
  shoot_full_row.add_precondition(Or(LT(b,r), GT(b,l), Equals(blocks[b][lc], p), Equals(blocks[b][lc], N)))
# At least one of the blocks of the row or the column is 'p'
colour_in_row = [Equals(blocks[r][c], p) for c in range(columns-1)]
colour_in_column = [And(Equals(blocks[b][lc], p), GE(b, r), LE(b, l))
for b in range(rows-1)]
shoot_full_row.add_precondition(Or(colour_in_row + colour_in_column))

    # Effects
# The following block from the last 'l' stores the colour of 'p' --- l+1 CONTROLAT ALS PARAMETRES I A LES PRECONDICIONS
shoot_full_row.add_effect(blocks[l+1][lc], p)
# The hand stores the colour of the following block from the last 'l'
shoot_full_row.add_effect(hand, blocks[l+1][lc])
# Control the fall of the blocks from the previous row - last's previous blocks become its above block --- CONTROLAT PER EFECTE CONDICIONAL
for c in range(0, columns-1):
  for a in range(0, rows-1):
    if a == 0:
      shoot_full_row.add_effect(blocks[a][c], N)
    else:
      shoot_full_row.add_effect(blocks[a][c], blocks[a-1][c], LE(a,r))

# Control the fall of the blocks from the previous rows of the last column - affected vertically and x number of blocks must go down according to gravity
for b in range(0,rows-1):
    shoot_full_row.add_effect(blocks[b][lc], N, And(LE(b,l), LT(b-(l-r+1), 0)))
    shoot_full_row.add_effect(blocks[b][lc], blocks[b-(l-r+1)][lc], And(LE(b,l), GE(b-(l-r+1), 0)))

plotting_problem.add_action(shoot_full_row)


########################### SHOOT FULL ROW FULL COLUMN #########################
# shoot complete row and then going down until a=the next is not p and not none
shoot_full_row_full_column = unified_planning.model.InstantaneousAction('shoot_full_row_full_column', p=Colour, r=IntType(0, rows-1))
# p: colour of the blocks that we want to eliminate
# r: row
p = shoot_full_row_full_column.parameter('p')
r = shoot_full_row_full_column.parameter('r')

    # Preconditions
# 'p' is never the wildcard or none
shoot_full_row_full_column.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_full_row_full_column.add_precondition(Or(Equals(p, hand), Equals(hand, W)))
same_colour = []
# All the blocks of the row 'r' are none or 'p'
for c in range(0, columns):
  shoot_full_row_full_column.add_precondition(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)))
  same_colour.append(Equals(blocks[r][c], p))
# All the blocks of the last column under 'r' are none or p
for u in range(0, rows):
  shoot_full_row_full_column.add_precondition(Or(LT(u,r), Equals(blocks[u][lc], p), Equals(blocks[u][lc], N)))
  same_colour.append(And(Equals(blocks[r][lc], p), GE(u,r)))
# At least one of the blocks of the row or the column is 'p'
shoot_full_row_full_column.add_precondition(Or(same_colour))

    # Effects
# The hand stores the colour of 'p'
shoot_full_row_full_column.add_effect(hand, p)
# Control the fall of the blocks from the previous row - last's previous blocks become its above blocks
for c in range(0, columns-1):
  for a in range(0, rows):
    if a == 0:
      shoot_full_row_full_column.add_effect(blocks[a][c], N)
    else:
      shoot_full_row_full_column.add_effect(blocks[a][c], blocks[a-1][c], LE(a,r))
# Control the fall of the blocks from the previous rows of the last column - affected vertically and x number of blocks must go down according to gravity
for b in range(0,rows):
    shoot_full_row_full_column.add_effect(blocks[b][lc], N, LT(b-(rows-r), 0))
    shoot_full_row_full_column.add_effect(blocks[b][lc], blocks[b-(rows-r)][lc], GE(b-(rows-r), 0))

plotting_problem.add_action(shoot_full_row_full_column)


############################### SHOOT PARTIAL COLUMN ###################################
shoot_partial_column = unified_planning.model.InstantaneousAction('shoot_partial_column', p=Colour, c=IntType(0, columns-1), l=IntType(0, rows-2))
# p: colour of the blocks that we want to eliminate
# c: column
# l: last row affected - next block is different than p and not none
p = shoot_partial_column.parameter('p')
c = shoot_partial_column.parameter('c')
l = shoot_partial_column.parameter('l')

      # Preconditions
# 'p' is never the wildcard or none
shoot_partial_column.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_partial_column.add_precondition(Or(Equals(p, hand), Equals(W, hand)))
# The following block from the last 'l' is different to 'p' and not none
shoot_partial_column.add_precondition(Not(Or(Equals(blocks[l+1][c], p), Equals(blocks[l+1][c], N))))
same_colour = []
# The blocks of the row 'r' before the last 'l' are none or 'p'
for r in range(0, rows-1):
  shoot_partial_column.add_precondition(Or(GT(r,l), Equals(blocks[r][c], p), Equals(blocks[r][c], N)))
  same_colour.append(And(Equals(blocks[r][c], p), LE(r,l)))
# At least one of the blocks of the column is 'p'
shoot_partial_column.add_precondition(Or(same_colour))

      # Effects
# The hand stores the colour of the following block from the last 'l'.
shoot_partial_column.add_effect(hand, blocks[l+1][c])
# The following block from the last 'l' stores 'p'.
shoot_partial_column.add_effect(blocks[l+1][c], p)
# The blocks previous to the last 'l' become none.
for r in range(0, rows-1):
  shoot_partial_column.add_effect(blocks[r][c], N, LE(r,l))

plotting_problem.add_action(shoot_partial_column)


############################# SHOOT FULL COLUMN ################################
shoot_full_column = unified_planning.model.InstantaneousAction('shoot_full_column', p=Colour, c=IntType(0, columns-1))
# p: colour of the blocks that we want to eliminate
# c: column
p = shoot_full_column.parameter('p')
c = shoot_full_column.parameter('c')

      # Preconditions
# 'p' is never the wildcard or none
shoot_full_column.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_full_column.add_precondition(Or(Equals(p, hand), Equals(W, hand)))
same_colour = []
# All the blocks of the column 'c' are 'p' or none
for r in range(0, rows):
  shoot_full_column.add_precondition(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)))
  same_colour.append(Equals(blocks[r][c], p))
# At least one of the blocks of the column is 'p'
shoot_full_column.add_precondition(Or(same_colour))

      # Effects
# The hand stores 'p'
shoot_full_column.add_effect(hand, p)
# All the blocks of the column 'c' become none
for r in range(0, rows):
  shoot_full_column.add_effect(blocks[r][c], N)

plotting_problem.add_action(shoot_full_column)


################################################################################
#                                   GOALS                                      #
################################################################################
remaining_blocks = [Not(Equals(blocks[i][j], N)) for i in range(rows) for j in range(columns)]
plotting_problem.add_goal(LE(Count(remaining_blocks), 2))


from unified_planning.engines import CompilationKind
# The CompilationKind class is defined in the unified_planning/engines/mixins/compiler.py file
compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    CompilationKind.ARRAYS_REMOVING,
    CompilationKind.COUNT_REMOVING,
    CompilationKind.USERTYPE_FLUENTS_REMOVING,
]

problem = plotting_problem
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
    result = planner.solve(problem)
    plan = result.plan
    end = time.time()
    print(f"Solving: {end-mid} seconds")
    if plan is None:
        print("No plan found.")
    else:
      compiled_plan = plan
      for result in reversed(results):
        compiled_plan = compiled_plan.replace_action_instances(
            result.map_back_action_instance
        )
      print("Compiled plan: ", compiled_plan.actions)