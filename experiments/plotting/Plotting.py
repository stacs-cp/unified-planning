from unified_planning.shortcuts import *

compilation = 'count'
solving = 'fast-downward'

# ---------------------------------------------------- Problem ---------------------------------------------------------

plotting_problem = Problem('plotting_problem')
Colour = UserType('Colour')
R = Object('R', Colour)
B = Object('B', Colour)
G = Object('G', Colour)
Y = Object('Y', Colour)
V = Object('V', Colour)
O = Object('O', Colour)
W = Object('W', Colour)
N = Object('N', Colour)
plotting_problem.add_objects([R,G,W,N])

initial_blocks = [[R,R,R,G],[R,G,G,G]]
rows = len(initial_blocks)
columns = len(initial_blocks[0])
lc = columns-1
lr = rows-1

blocks = Fluent('blocks', ArrayType(rows, ArrayType(columns, Colour)))
hand = Fluent('hand', Colour)
plotting_problem.add_fluent(blocks)
plotting_problem.add_fluent(hand, default_initial_value = W)
plotting_problem.set_initial_value(blocks, initial_blocks)

########################################################################################################################
#                                                     ACTIONS                                                          #
########################################################################################################################

################################################ SHOOT PARTIAL ROW #####################################################
shoot_partial_row = InstantaneousAction('shoot_partial_row', p=Colour, r=IntType(0, rows-1), l=IntType(0, columns-2))
# p: colour of the blocks that we want to eliminate
# r: row
# l: last column affected (cannot be the last one) - next block is different from p and not none
p = shoot_partial_row.parameter('p')
r = shoot_partial_row.parameter('r')
l = shoot_partial_row.parameter('l')

      # Preconditions
# 'p' is never the wildcard or none
shoot_partial_row.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_partial_row.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
# The following block from the last 'l' is different from 'p' and not none
shoot_partial_row.add_precondition(Not(Or(Equals(blocks[r][l+1], p), Equals(blocks[r][l+1], N))))
# All blocks in the row 'r' before the last 'l' are 'null' or 'p'
b = RangeVariable('b', 0, l)
shoot_partial_row.add_precondition(Forall(Or(Equals(blocks[r][b], p), Equals(blocks[r][b], N)), b))
# At least one of the blocks of the row 'r' until 'l' is 'p'
shoot_partial_row.add_precondition(Exists(Equals(blocks[r][b], p), b))

      # Effects
# The hand saves the colour of the following block from the last 'l'
shoot_partial_row.add_effect(hand, blocks[r][l+1])
# The following block from the last 'l' saves the colour of 'p'
shoot_partial_row.add_effect(blocks[r][l+1], p)
# Each block in the row 'r' until the last 'l' become its above block, except the ones in the first row, that become none
shoot_partial_row.add_effect(blocks[0][b], N, forall=[b])
a = RangeVariable('a', 1, r)
shoot_partial_row.add_effect(blocks[a][b], blocks[a-1][b], forall=[a,b])

plotting_problem.add_action(shoot_partial_row)

################################################### SHOOT FULL ROW #####################################################
# shoot complete row and then going down until the next is not p and not none
shoot_full_row = InstantaneousAction('shoot_full_row', p=Colour, r=IntType(0, rows-2), l=IntType(0, rows-2))
# p: colour of the blocks that we want to eliminate
# r: complete row affected
# l: last row of the last column affected (can't be the last one) next block is different than p and not none
p = shoot_full_row.parameter('p')
r = shoot_full_row.parameter('r')
l = shoot_full_row.parameter('l')

    # Preconditions
# 'l' has to be greater or equal than 'r'
shoot_full_row.add_precondition(GE(l, r))
# 'p' is never the wildcard or none
shoot_full_row.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_full_row.add_precondition(Or(Equals(hand, p), Equals(hand, W)))
# The following block from the last 'l' is different to 'p' and not none
shoot_full_row.add_precondition(Not(Or(Equals(blocks[l+1][lc], p), Equals(blocks[l+1][lc], N))))
# All the blocks of the row 'r' are none or 'p'
c = RangeVariable('c', 0, lc)
shoot_full_row.add_precondition(Forall(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)), c))
# The blocks between 'r' and 'l' are 'p' or none
b = RangeVariable('b', r+1, l)
shoot_full_row.add_precondition(Forall(Or(Equals(blocks[b][lc], p), Equals(blocks[b][lc], N)), b))
# At least one of the blocks of the row or the column is 'p'
a = RangeVariable('a', 0, lc)
shoot_full_row.add_precondition(Or(
    Exists(Equals(blocks[r][a], p), a),
    Exists(Equals(blocks[b][lc], p), b)
))

    # Effects
# The following block from the last 'l' stores the colour of 'p'
shoot_full_row.add_effect(blocks[l+1][lc], p)
# The hand stores the colour of the following block from the last 'l'
shoot_full_row.add_effect(hand, blocks[l+1][lc])
# Control the fall of the blocks from the previous row - last's previous blocks become its above block
a = RangeVariable('a', 1, r)
c = RangeVariable('c', 0, lc-1)
shoot_full_row.add_effect(blocks[0][c], N, forall=[c])
shoot_full_row.add_effect(blocks[a][c], blocks[a-1][c], forall=[a,c])

# Control the fall of the blocks from the previous rows of the last column - affected vertically and x number of blocks
# must go down according to gravity
n = RangeVariable('n', 0, r-l) # blocks that will become N (from top until r-l)
b = RangeVariable('b', r, l) # blocs that will fall down
shoot_full_row.add_effect(blocks[n][lc], N, forall=[n])
shoot_full_row.add_effect(blocks[b][lc], blocks[b-(r-l+1)][lc], forall=[b])

plotting_problem.add_action(shoot_full_row)

########################### SHOOT FULL ROW FULL COLUMN #########################
# shoot complete row and complete last column
shoot_full_row_full_column = InstantaneousAction('shoot_full_row_full_column', p=Colour, r=IntType(0, rows-1))
# p: colour of the blocks that we want to eliminate
# r: row
p = shoot_full_row_full_column.parameter('p')
r = shoot_full_row_full_column.parameter('r')

    # Preconditions
# 'p' is never the wildcard or none
shoot_full_row_full_column.add_precondition(Not(Or(Equals(p, W), Equals(p, N))))
# 'p' is the same colour as the hand or the hand is the wildcard
shoot_full_row_full_column.add_precondition(Or(Equals(p, hand), Equals(hand, W)))
# All the blocks of the row 'r' are none or 'p'
c = RangeVariable('c', 0, lc)
shoot_full_row_full_column.add_precondition(Forall(Or(Equals(blocks[r][c], p), Equals(blocks[r][c], N)), c))
 # All the blocks of the last column under 'r' are none or p
a = RangeVariable('a', r, lr)
shoot_full_row_full_column.add_precondition(Forall(Or(Equals(blocks[a][lc], p), Equals(blocks[a][lc], N)), a))
# At least one of the blocks of the row or the column is 'p'
shoot_full_row_full_column.add_precondition(Or(
    Exists(Equals(blocks[r][c], p), c),
    Exists(Equals(blocks[r][a], p), a),
))

    # Effects
# The hand stores the colour of 'p'
shoot_full_row_full_column.add_effect(hand, p)
# Control the fall of the blocks from the previous rows - last's previous blocks become its above blocks
a = RangeVariable('a', 1, r) # Number of rows that have to fall
for c in range(0, columns-1):
    shoot_full_row_full_column.add_effect(blocks[0][c], N)
    shoot_full_row_full_column.add_effect(blocks[a][c], blocks[a-1][c], forall=[a])
# Control the fall of the blocks from the previous rows of the last column - affected vertically and x number of blocks
# must go down according to gravity
n = RangeVariable('n', 0, lr-r) # blocks that will become N
shoot_full_row_full_column.add_effect(blocks[n][lc], N, forall=[n])
b = RangeVariable('b', 0, r-1) # blocs that will fall down
shoot_full_row_full_column.add_effect(blocks[lr-b][lc], blocks[r-1-b][lc], forall=[b])

plotting_problem.add_action(shoot_full_row_full_column)


############################### SHOOT PARTIAL COLUMN ###################################
shoot_partial_column = InstantaneousAction('shoot_partial_column', p=Colour, c=IntType(0, columns-1), l=IntType(0, rows-2))
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
shoot_full_column = InstantaneousAction('shoot_full_column', p=Colour, c=IntType(0, columns-1))
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

########################################################################################################################
#                                                       GOALS                                                          #
########################################################################################################################
remaining_blocks = [Not(Equals(blocks[i][j], N)) for i in range(rows) for j in range(columns)]
plotting_problem.add_goal(LE(Count(remaining_blocks), 1))

# -------------------------------------------------- Compilation -------------------------------------------------------

from unified_planning.engines import CompilationKind
if compilation == 'count':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
elif compilation == 'count-int':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        CompilationKind.INTEGERS_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
elif compilation == 'count-int-numeric':
    compilation_kinds_to_apply = [
        CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
        CompilationKind.ARRAYS_REMOVING,
        CompilationKind.COUNT_INT_REMOVING,
        CompilationKind.USERTYPE_FLUENTS_REMOVING,
    ]
else:
    raise ValueError(f"Unsupported compilation type: {compilation}")

if solving == 'fast-downward-opt':
    compilation_kinds_to_apply.append(CompilationKind.CONDITIONAL_EFFECTS_REMOVING)

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

# ---------------------------------------------------- Solving ---------------------------------------------------------

with OneshotPlanner(name=solving) as planner:
    result = planner.solve(problem)

    plan = result.plan
    if plan is not None:
        compiled_plan = plan
        for result in reversed(results):
            compiled_plan = compiled_plan.replace_action_instances(
                result.map_back_action_instance
            )
        print(compiled_plan)
    else:
        print(result)

    if not planner.supports(problem.kind):
        unsupported_features = [
            f"{pk} is not supported by the planner"
            for pk in problem.kind.features if pk not in planner.supported_kind().features
        ]
        print("\n".join(unsupported_features))