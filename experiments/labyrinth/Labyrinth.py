from experiments import compilation_solving
from unified_planning.shortcuts import *

compilation = 'up'
solving = 'fast-downward'

rows = 6
columns = 6
# ---------------------------------------------------- Problem ---------------------------------------------------------
labyrinth = Problem('labyrinth')

Card = UserType('Card')
card1 = Object('card1', Card)
card2 = Object('card2', Card)
card3 = Object('card3', Card)
card4 = Object('card4', Card)
card5 = Object('card5', Card)
card6 = Object('card6', Card)
card7 = Object('card7', Card)
card8 = Object('card8', Card)
card9 = Object('card9', Card)

Direction = UserType('Direction')
N = Object('N', Direction)
S = Object('S', Direction)
W = Object('W', Direction)
E = Object('E', Direction)

card_at = Fluent('card_at', ArrayType(rows, ArrayType(columns, Card)))
robot_at = Fluent('robot_at', c=Card)
blocked = Fluent('blocked', c=Card, d=Direction)
left = Fluent('left')
#cards_moving = Fluent('cards_moving')
#cards_moving_west = Fluent('cards_moving_west')
#cards_moving_east = Fluent('cards_moving_east')
#cards_moving_south = Fluent('cards_moving_south')
#cards_moving_north = Fluent('cards_moving_north')
#next_moving_card = Fluent('next_moving_card')
#next_headtail_card = Fluent('next_headtail_card')
labyrinth.add_fluent(card_at)
labyrinth.set_initial_value(card_at[0][0], card1)
labyrinth.set_initial_value(card_at[0][1], card2)
labyrinth.set_initial_value(card_at[0][2], card3)
labyrinth.set_initial_value(card_at[1][0], card5)
labyrinth.set_initial_value(card_at[1][1], card6)
labyrinth.set_initial_value(card_at[1][2], card4)
labyrinth.set_initial_value(card_at[2][0], card7)
labyrinth.set_initial_value(card_at[2][1], card8)
labyrinth.set_initial_value(card_at[2][2], card9)
labyrinth.add_fluent(robot_at)
labyrinth.add_fluent(blocked)
labyrinth.add_fluent(left)

move_west = InstantaneousAction('move_west', cfrom=Card, xfrom=IntType(0, rows-1),
                                yfrom=IntType(0, columns-1), cto=Card)
cfrom = move_west.parameter('cfrom')
xfrom = move_west.parameter('xfrom')
yfrom = move_west.parameter('yfrom')
cto = move_west.parameter('cto')
#move_west.add_precondition(Not(cards_moving))
move_west.add_precondition(robot_at(cfrom))
move_west.add_precondition(Equals(card_at[xfrom][yfrom], cfrom))
move_west.add_precondition(Equals(card_at[xfrom][yfrom-1], cto))
move_west.add_precondition(Not(blocked(cfrom, W)))
move_west.add_precondition(Not(blocked(cto, E)))
move_west.add_effect(robot_at(cfrom), False)
move_west.add_effect(robot_at(cto), True)
labyrinth.add_action(move_west)

move_east = InstantaneousAction('move_east', cfrom=Card, xfrom=IntType(0, rows-1),
                                yfrom=IntType(0, columns-1), cto=Card)
cfrom = move_east.parameter('cfrom')
xfrom = move_east.parameter('xfrom')
yfrom = move_east.parameter('yfrom')
cto = move_east.parameter('cto')
#move_east.add_precondition(Not(cards_moving))
move_east.add_precondition(robot_at(cfrom))
move_east.add_precondition(Equals(card_at[xfrom][yfrom], cfrom))
move_east.add_precondition(Equals(card_at[xfrom][yfrom+1], cto))
move_east.add_precondition(Not(blocked(cfrom, E)))
move_east.add_precondition(Not(blocked(cto, W)))
move_east.add_effect(robot_at(cfrom), False)
move_east.add_effect(robot_at(cto), True)
labyrinth.add_action(move_east)

move_north = InstantaneousAction('move_north', cfrom=Card, xfrom=IntType(0, rows-1),
                                yfrom=IntType(0, columns-1), cto=Card)
cfrom = move_north.parameter('cfrom')
xfrom = move_north.parameter('xfrom')
yfrom = move_north.parameter('yfrom')
cto = move_north.parameter('cto')
#move_north.add_precondition(Not(cards_moving))
move_north.add_precondition(robot_at(cfrom))
move_north.add_precondition(Equals(card_at[xfrom][yfrom], cfrom))
move_north.add_precondition(Equals(card_at[xfrom-1][yfrom], cto))
move_north.add_precondition(Not(blocked(cfrom, N)))
move_north.add_precondition(Not(blocked(cto, S)))
move_north.add_effect(robot_at(cfrom), False)
move_north.add_effect(robot_at(cto), True)
labyrinth.add_action(move_north)

move_south = InstantaneousAction('move_south', cfrom=Card, xfrom=IntType(0, rows-1),
                                yfrom=IntType(0, columns-1), cto=Card)
cfrom = move_south.parameter('cfrom')
xfrom = move_south.parameter('xfrom')
yfrom = move_south.parameter('yfrom')
cto = move_south.parameter('cto')
#move_south.add_precondition(Not(cards_moving))
move_south.add_precondition(robot_at(cfrom))
move_south.add_precondition(Equals(card_at[xfrom][yfrom], cfrom))
move_south.add_precondition(Equals(card_at[xfrom+1][yfrom], cto))
move_south.add_precondition(Not(blocked(cfrom, S)))
move_south.add_precondition(Not(blocked(cto, N)))
move_south.add_effect(robot_at(cfrom), False)
move_south.add_effect(robot_at(cto), True)
labyrinth.add_action(move_south)

# moure les cartes d'una fila o columna amb un forall
move_cards_west = InstantaneousAction('move_cards_west', x=IntType(0, rows-1))
x = move_cards_west.parametex('x')
#move_cards_west.add_precondition(Not(cards_moving)) # ?
i = RangeVariable('i', 0, columns-2)
move_cards_west.add_effect(card_at[x][i], card_at[x][i+1], forall=[i])
move_cards_west.add_effect(card_at[x][columns-1], card_at[x][0])
labyrinth.add_action(move_cards_west)

move_cards_east = InstantaneousAction('move_cards_east', x=IntType(0, rows-1))
x = move_cards_east.parameter('x')
#move_cards_east.add_precondition(Not(cards_moving)) # ?
i = RangeVariable('i', 1, columns-1)
move_cards_east.add_effect(card_at[x][i], card_at[x][i-1])
move_cards_east.add_effect(card_at[x][0], card_at[x][columns-1])
labyrinth.add_action(move_cards_east)

move_cards_north = InstantaneousAction('move_cards_north', y=IntType(0, columns-1))
y = move_cards_north.parameter('y')
#move_cards_north.add_precondition(Not(cards_moving)) # ?
i = RangeVariable('i', 0, rows-2)
move_cards_north.add_effect(card_at[y][i], card_at[y][i+1])
move_cards_north.add_effect(card_at[y][rows-1], card_at[y][0])
labyrinth.add_action(move_cards_north)

move_cards_south = InstantaneousAction('move_cards_south', y=IntType(0, columns-1))
y = move_cards_south.parameter('y')
#move_cards_south.add_precondition(Not(cards_moving)) # ?
i = RangeVariable('i', 1, rows-1)
move_cards_south.add_effect(card_at[i][y], card_at[i-1][y])
move_cards_south.add_effect(card_at[0][y], card_at[rows-1][y])
labyrinth.add_action(move_cards_south)

leave = InstantaneousAction('leave', c=Card)
c = leave.parameter('c')
leave.add_precondition(robot_at(c))
leave.add_precondition(Equals(card_at[rows-1][columns-1], c))
leave.add_precondition(Or(Not(blocked(c, S)), Not(blocked(c, E))))
leave.add_effect(left, True)
labyrinth.add_action(leave)

costs: Dict[Action, Expression] = {
    move_west: Int(1),
    move_north: Int(1),
    move_south: Int(1),
    move_east: Int(1),
    move_cards_east: Int(1),
    move_cards_north: Int(1),
    move_cards_south: Int(1),
    move_cards_west: Int(1),
    leave: Int(1)
}
labyrinth.add_quality_metric(MinimizeActionCosts(costs))

assert compilation in ['up'], f"Unsupported compilation type: {compilation}"

compilation_solving.compile_and_solve(labyrinth, solving, compilation)