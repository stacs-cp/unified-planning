import unified_planning
from unified_planning.shortcuts import *
from unified_planning.plans import SequentialPlan, ActionInstance
from unified_planning.engines import CompilationKind
from unified_planning.io import PDDLWriter
from random import randrange
import math
import datetime
##

# defining types: Tile, Block-Types


# Tile = UserType('Tile') # tile holds blocks
Block = UserType('Block') #blocks apply logic based on sentence rules on the board (e.g. BABA IS YOU)
baba = unified_planning.model.Object("baba", Block)
flag = unified_planning.model.Object("flag", Block)
wall = unified_planning.model.Object("wall", Block)
# NOUN blocks for corresponding object blocks (e.g. BABA is text block which represents baba)
baba_text = unified_planning.model.Object("BABA", Block)
flag_text = unified_planning.model.Object("FLAG", Block)
wall_text = unified_planning.model.Object("WALL", Block)
you = unified_planning.model.Object("YOU_", Block)
push = unified_planning.model.Object("PUSH", Block)
stop = unified_planning.model.Object("STOP", Block)
win = unified_planning.model.Object("WIN_", Block)
is_block = unified_planning.model.Object("_IS_", Block)
blocks = [baba, flag, wall, baba_text, flag_text, wall_text, you, push, stop, win, is_block]
text_blocks = [baba_text, flag_text, wall_text, you, push, stop, win, is_block]
non_text_blocks = [baba, flag, wall]
nouns = len(non_text_blocks)

# set up tile system
rows = 6
cols = 6
total_tiles = rows*cols
#tiles = Fluent('tiles', ArrayType(total_tiles, Block))
# tiles = [];
# for i in range(cols):
#   for j in range(rows):
#     # tiles.append(unified_planning.model.Object("t"+str(i)+str(j), Tile))
#     tiles.append(0)

# tiles are Tiles that blocks may occupy, they are not fluents. the blocks themselves (wall, baba, etc), are fluent as they can change attributes and move around the board.

# connected states whether two tiles are adjacent.
# connected = unified_planning.model.Fluent('connected', BoolType(), l_from=Tile, l_to=Tile)
# block_at = unified_planning.model.Fluent('block_at', BoolType(), l=Tile) #where a block is
same_col = unified_planning.model.Fluent('same_col', BoolType(), l_from=IntType(0, (total_tiles)-1), l_to=IntType(0, (total_tiles)-1))
same_row = unified_planning.model.Fluent('same_row', BoolType(), l_from=IntType(0, (total_tiles)-1), l_to=IntType(0, (total_tiles)-1))
# for handling cascading push-logic.
# right_of_index = unified_planning.model.Fluent('right_of_index', IntType(), l_from=Tile)
# left_of_index = unified_planning.model.Fluent('left_of_index', IntType(), l_from=Tile)
# above_of_index = unified_planning.model.Fluent('above_of_index', IntType(), l_from=Tile)
# below_of_index = unified_planning.model.Fluent('below_of_index', IntType(), l_from=Tile)
# curr_tile_index = unified_planning.model.Fluent('curr_tile_index', IntType(), l_from=Tile)
free_space_found = unified_planning.model.Fluent('free_space_found', BoolType(), l_from=IntType(0, (total_tiles)-1))
no_free_space_found = unified_planning.model.Fluent('free_space_found', BoolType())
push_init_pos = unified_planning.model.Fluent('push_init_pos', IntType(0, total_tiles))
push_in_progress_last_l_to_index = unified_planning.model.Fluent('push_in_progress_last_l_to_index', IntType(0, total_tiles))
push_in_progress_up_1 = unified_planning.model.Fluent('push_in_progress_up_1', BoolType())
push_in_progress_up_2 = unified_planning.model.Fluent('push_in_progress_up_2', BoolType())
push_in_progress_down_1 = unified_planning.model.Fluent('push_in_progress_down_1', BoolType())
push_in_progress_down_2 = unified_planning.model.Fluent('push_in_progress_down_2', BoolType())
push_in_progress_left_1 = unified_planning.model.Fluent('push_in_progress_left_1', BoolType())
push_in_progress_left_2 = unified_planning.model.Fluent('push_in_progress_left_2', BoolType())
push_in_progress_right_1 = unified_planning.model.Fluent('push_in_progress_right_1', BoolType())
push_in_progress_right_2 = unified_planning.model.Fluent('push_in_progress_right_2', BoolType())
free_space_found_counter = unified_planning.model.Fluent('free_space_found_counter', IntType(0, total_tiles))

# for collective movement of all YOU blocks
move_all_up = unified_planning.model.Fluent('move_all_up', BoolType())
move_all_down = unified_planning.model.Fluent('move_all_down', BoolType())
move_all_left = unified_planning.model.Fluent('move_all_left', BoolType())
move_all_right = unified_planning.model.Fluent('move_all_right', BoolType())
you_tile_reviewed = unified_planning.model.Fluent('you_tile_reviewed', BoolType(), you_tile=IntType(0, (total_tiles)-1))

# sentence construction
scan_over_in_next_call = unified_planning.model.Fluent('scan_over_in_next_call', BoolType())
check_for_new_sentences = unified_planning.model.Fluent('check_for_new_sentences', BoolType())
check_for_win = unified_planning.model.Fluent('check_for_win', BoolType())
## below defines which blocks ar eimpacted by which rules
push_applied = unified_planning.model.Fluent('push_applied', BoolType(), block=Block)
stop_applied = unified_planning.model.Fluent('stop_applied', BoolType(), block=Block)
you_applied = unified_planning.model.Fluent('you_applied', BoolType(), block=Block)
win_applied = unified_planning.model.Fluent('win_applied', BoolType(), block=Block)

#meta
num_nouns = unified_planning.model.Fluent('num_nouns', IntType(0, len(text_blocks)))
nouns_reviewed_counter = unified_planning.model.Fluent('nouns_reviewed_counter', IntType(0, len(text_blocks)))
noun_reviewed = unified_planning.model.Fluent('noun_reviewed', BoolType(), noun=Block)
failure = unified_planning.model.Fluent('failure', BoolType())
success = unified_planning.model.Fluent('success', BoolType())
# below returns if block currently 'held' by tile - moultiple blocks can occupy the same tile.
tile_holds = unified_planning.model.Fluent('tile_holds', BoolType(), l_from=IntType(0, (total_tiles)), block=Block)
carry_tile_holds = unified_planning.model.Fluent('carry_tile_holds', BoolType(), l_from=IntType(0, (total_tiles)), block=Block)
block_text_match = unified_planning.model.Fluent('block_text_match', BoolType(), block_1=Block, block_2=Block)
is_text = unified_planning.model.Fluent('is_text', BoolType(), block=Block)

# defining action which states what direction of movement the solver is exploring. once chosen, all YOU-applied tiles must (attempt to) move in that direction before selecting another direction of movement.
move_all_select_up = unified_planning.model.InstantaneousAction('move_all_select_up')
move_all_select_up.add_precondition(Not(check_for_new_sentences)) # only start movement once sentences have been reviewed on the board.
move_all_select_up.add_precondition(Not(Or(move_all_up, move_all_down, move_all_left, move_all_right))) # solver has not yet indicated what direction to pursue for all YOU blocks.
move_all_select_up.add_effect(move_all_up, True)

move_all_select_down = unified_planning.model.InstantaneousAction('move_all_select_down')
move_all_select_down.add_precondition(Not(check_for_new_sentences)) # only start movement once sentences have been reviewed on the board.
move_all_select_down.add_precondition(Not(Or(move_all_up, move_all_down, move_all_left, move_all_right))) # solver has not yet indicated what direction to pursue for all YOU blocks.
move_all_select_down.add_effect(move_all_down, True)

move_all_select_left = unified_planning.model.InstantaneousAction('move_all_select_left')
move_all_select_left.add_precondition(Not(check_for_new_sentences)) # only start movement once sentences have been reviewed on the board.
move_all_select_left.add_precondition(Not(Or(move_all_up, move_all_down, move_all_left, move_all_right))) # solver has not yet indicated what direction to pursue for all YOU blocks.
move_all_select_left.add_effect(move_all_left, True)

move_all_select_right = unified_planning.model.InstantaneousAction('move_all_select_right')
move_all_select_right.add_precondition(Not(check_for_new_sentences)) # only start movement once sentences have been reviewed on the board.
move_all_select_right.add_precondition(Not(Or(move_all_up, move_all_down, move_all_left, move_all_right))) # solver has not yet indicated what direction to pursue for all YOU blocks.
move_all_select_right.add_effect(move_all_right, True)

# defining action for moving one YOU-applied block at a time.
move_up = unified_planning.model.InstantaneousAction('move_up', l_from=IntType(0,(total_tiles)-1), you_block=Block)
move_down = unified_planning.model.InstantaneousAction('move_down', l_from=IntType(0,(total_tiles)-1), you_block=Block)
move_left = unified_planning.model.InstantaneousAction('move_left', l_from=IntType(0,(total_tiles)-1), you_block=Block)
move_right = unified_planning.model.InstantaneousAction('move_right', l_from=IntType(0,(total_tiles)-1), you_block=Block)
def move_action_builder(move_action, up_down_left_right):
  l_from = move_action.parameter('l_from')
  # l_to = move_action.parameter('l_to')
  you_block = move_action.parameter('you_block')
  move_action.add_precondition(Not(push_in_progress_up_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_up_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_down_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_down_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_left_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_left_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_right_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(push_in_progress_right_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  move_action.add_precondition(Not(check_for_new_sentences()))

  # for i in range(total_tiles):
  #   for j in range(len(blocks)):
  #     move_action.add_precondition(Not(And(tile_holds(i, blocks[i]), Or(push_applied(blocks[i]), stop_applied(blocks[i]))))) #you can only move to l_to if the tile does not hold a block that has push_applied or stop_applied.

  move_action.add_precondition(Not(you_tile_reviewed(l_from))) #not a previously reviewed tile (you_tile_reviewed resets after every complete move_all action)
  move_action.add_precondition(And(tile_holds(l_from, you_block), you_applied(you_block))) #checks that tiles holds specified block, and thast it is a player-block. OLD: And(tile_holds(l_from, you_block), you_applied(you_block)))
  if(up_down_left_right == 0):
    move_action.add_precondition(move_all_up)
  elif(up_down_left_right == 1):
    move_action.add_precondition(move_all_down)
  elif(up_down_left_right == 2):
    move_action.add_precondition(move_all_left)
  elif(up_down_left_right == 3):
    move_action.add_precondition(move_all_right)
  # if(up_down_left_right == 0): # up
  #   next_over = -rows;
  #   direction = above_of_index(l_from)
  #   # move_action.add_precondition(Not(Exists(And(tile_holds(possible_tile, possible_block), you_applied(possible_block), LT(curr_tile_index(possible_tile), curr_tile_index(l_from))), possible_tile, possible_block))) #checks that we are currently considering the furthermost tile with a you-applied block.
  # elif(up_down_left_right == 1): # down
  #   next_over = rows;
  #   direction = below_of_index(l_from)
  #   # move_action.add_precondition(Not(Exists(And(tile_holds(possible_tile, possible_block), you_applied(possible_block), GT(curr_tile_index(possible_tile), curr_tile_index(l_from))), possible_tile, possible_block))) #checks that we are currently considering the furthermost tile with a you-applied block.
  # elif(up_down_left_right == 2): # left
  #   next_over = -1;
  #   direction = left_of_index(l_from)#
  #   # move_action.add_precondition(Not(Exists(And(tile_holds(possible_tile, possible_block), you_applied(possible_block), LT(curr_tile_index(possible_tile), curr_tile_index(l_from))), possible_tile, possible_block))) #checks that we are currently considering the furthermost tile with a you-applied block.
  # elif(up_down_left_right == 3): # right
  #   next_over = 1;
  #   direction = right_of_index(l_from)
  #   # move_action.add_precondition(Not(Exists(And(tile_holds(possible_tile, possible_block), you_applied(possible_block), GT(curr_tile_index(possible_tile), curr_tile_index(l_from))), possible_tile, possible_block))) #checks that we are currently considering the furthermost tile with a you-applied block.
  next_over = 1
  # direction = above_of_index(l_from)
  for i in range(total_tiles):
    for j in range(len(blocks)):
      if(up_down_left_right == 0): # up
        next_over = -rows;
        # direction = above_of_index(l_from)
        move_action.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), LT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.
      elif(up_down_left_right == 1): # down
        next_over = rows;
        # direction = below_of_index(l_from)
        move_action.add_precondition(Not(And(Not(you_tile_reviewed(i)),tile_holds(i, blocks[j]), you_applied(blocks[j]), GT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.
      elif(up_down_left_right == 2): # left
        next_over = -1;
        # direction = left_of_index(l_from)#
        move_action.add_precondition(Not(And(Not(you_tile_reviewed(i)),tile_holds(i, blocks[j]), you_applied(blocks[j]), LT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.
      elif(up_down_left_right == 3): # right
        next_over = 1;
        # direction = right_of_index(l_from)
        move_action.add_precondition(Not(And(Not(you_tile_reviewed(i)),tile_holds(i, blocks[j]), you_applied(blocks[j]), GT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.

  # move_action.add_precondition(Equals(direction, curr_tile_index(Plus(l_from, next_over)))) #checks move is in correct direction


  for i in range(len(blocks)): #loop for checking that l_to does not contain any push-able or stop-applied blocks, which would make the move action invalid.
    move_action.add_precondition(Not(And(tile_holds(Plus(l_from, next_over), blocks[i]), Or(push_applied(blocks[i]), stop_applied(blocks[i])))))  #checks that tile we are moving to does not contain a stop or push block.


  # for i in range(len(non_text_blocks)):
  #     move_action.add_effect(tile_holds(l_to, non_text_blocks[i]), True, And(tile_holds(l_from, non_text_blocks[i]), you_applied(non_text_blocks[i]))) # we move whatever you-applied blocks are in l_from to l_to IF: l_to has no push/stop block, j is YOU and is held by i
  #     move_action.add_effect(tile_holds(l_from, non_text_blocks[i]), False, And(tile_holds(l_from, non_text_blocks[i]), you_applied(non_text_blocks[i]))) # we move whatever you-applied blocks are in l_from to l_to IF: l_to has no push/stop block, j is YOU and is held by i

  move_action.add_effect(tile_holds(Plus(l_from, next_over), you_block), True) # we move whatever you-applied blocks are in l_from to l_to IF: l_to has no push/stop block, j is YOU and is held by i
  move_action.add_effect(tile_holds(l_from, you_block), False) # we move whatever you-applied blocks are in l_from to l_to IF: l_to has no push/stop block, j is YOU and is held by i



  move_action.add_effect(you_tile_reviewed(l_from), True)
  move_action.add_effect(you_tile_reviewed(Plus(l_from, next_over)), True)

move_action_builder(move_up, 0)
move_action_builder(move_down, 1)
move_action_builder(move_left, 2)
move_action_builder(move_right, 3)

# defining actions for cascading push (i.e. you can push multiple blocks at once in the same direction iff: block has push_applied rule and there is one free space available in that direction (no push/stop blocks on it)).

push_up_1 = unified_planning.model.InstantaneousAction('push_up_1', l_from=IntType(0,(total_tiles)-1), you_block=Block, push_block=Block)
push_up_2 = unified_planning.model.InstantaneousAction('push_up_2', l_to=IntType(0,(total_tiles)-1))
push_up_3 = unified_planning.model.InstantaneousAction('push_up_3', l_from=IntType(0,(total_tiles)-1))
push_down_1 = unified_planning.model.InstantaneousAction('push_down_1', l_from=IntType(0,(total_tiles)-1), you_block=Block, push_block=Block)
push_down_2 = unified_planning.model.InstantaneousAction('push_down_2', l_to=IntType(0,(total_tiles)-1))
push_down_3 = unified_planning.model.InstantaneousAction('push_down_3', l_from=IntType(0,(total_tiles)-1))
push_left_1 = unified_planning.model.InstantaneousAction('push_left_1', l_from=IntType(0,(total_tiles)-1), you_block=Block, push_block=Block)
push_left_2 = unified_planning.model.InstantaneousAction('push_left_2', l_to=IntType(0,(total_tiles)-1))
push_left_3 = unified_planning.model.InstantaneousAction('push_left_3', l_from=IntType(0,(total_tiles)-1))
push_right_1 = unified_planning.model.InstantaneousAction('push_right_1', l_from=IntType(0,(total_tiles)-1), you_block=Block, push_block=Block)
push_right_2 = unified_planning.model.InstantaneousAction('push_right_2', l_to=IntType(0,(total_tiles)-1))
push_right_3 = unified_planning.model.InstantaneousAction('push_right_3', l_from=IntType(0,(total_tiles)-1))

def push_action_builder_1(push_action_1, up_down_left_right):
  l_from = push_action_1.parameter('l_from')
  # l_to = push_action_1.parameter('l_to')
  you_block = push_action_1.parameter('you_block')
  push_block = push_action_1.parameter('push_block')
  # push_action_1.add_precondition(connected(l_from, l_to))
  # push_action_1.add_precondition(Or(move_all_up, move_all_down, move_all_left, move_all_right)) # solver indicated what direction to pursue for all YOU blocks.
  # direction = above_of_index(l_from)
  next_over = 1
  reachable = same_col(l_from, Plus(l_from,(next_over*2)))
  if(up_down_left_right == 0): # up
    # direction = above_of_index(l_from)
    next_over = -rows
    reachable = same_col(l_from, Plus(l_from,(next_over*2)))
  elif(up_down_left_right == 1): # down
    # direction = below_of_index(l_from)
    next_over = rows
    reachable = same_col(l_from, Plus(l_from,(next_over*2)))
  elif(up_down_left_right == 2): # left
    # direction = left_of_index(l_from)
    next_over = -1
    reachable = same_row(l_from, Plus(l_from,(next_over*2)))
  elif(up_down_left_right == 3): # right
    # direction = right_of_index(l_from)
    next_over = 1
    reachable = same_row(l_from, Plus(l_from,(next_over*2)))
  # push_action_1.add_precondition(Equals(direction,curr_tile_index(l_))) #to is above from
  # for i in range(len(blocks)):
  #   push_action_1.add_precondition(Implies(tile_holds(l_to, blocks[i]), push_applied(blocks[i]))) # you can only attempt to push a tile if it contains a block that has push_applied on.
  push_action_1.add_precondition(Not(free_space_found(Plus(l_from, next_over)))) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_up_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_up_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_down_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_down_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_left_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_left_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_right_1())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(push_in_progress_right_2())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_1.add_precondition(Not(check_for_new_sentences())) # if push is complete, must check for new sentences before continuing.
  push_action_1.add_precondition(GE(Plus(l_from,(next_over*2)), -1)) # checking that push would be within bounds
  push_action_1.add_precondition(LE(Plus(l_from,(next_over*2)), total_tiles)) #checking the push would be within bounds
  push_action_1.add_precondition(reachable) #checking the push would be reachable given direction.

  # push_action_1.add_precondition(tile_holds(l_from, baba))

  # push_action_1.add_precondition(Not(Exists(And(tile_holds(possible_tile, possible_block), you_applied(possible_block), LT(curr_tile_index(possible_tile), curr_tile_index(l_from))), possible_tile, possible_block))) #checks that we are currently considering the furthermost tile with a you-applied block.

  #specifying direction allowed given move_all boolean solver selected.
  push_action_1.add_precondition(Not(you_tile_reviewed(l_from)))

  push_action_1.add_precondition(And(tile_holds(l_from, you_block), you_applied(you_block))) #checks that tiles holds specified block, and thast it is a player-block. OLD: And(tile_holds(l_from, you_block), you_applied(you_block)))
  push_action_1.add_precondition(And(tile_holds(Plus(l_from, next_over), push_block), push_applied(push_block))) #checks l_to tile  (And(tile_holds(l_to, push_block), push_applied(push_block)))


  if(up_down_left_right == 0):
    push_action_1.add_precondition(move_all_up)
  elif(up_down_left_right == 1):
    push_action_1.add_precondition(move_all_down)
  elif(up_down_left_right == 2):
    push_action_1.add_precondition(move_all_left)
  elif(up_down_left_right == 3):
    push_action_1.add_precondition(move_all_right)

  for i in range(total_tiles):
    for j in range(len(blocks)):
      if(up_down_left_right == 0): # up
        push_action_1.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), LT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.
      elif(up_down_left_right == 1): # down
        push_action_1.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), GT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.
      elif(up_down_left_right == 2): # left
        push_action_1.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), LT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.
      elif(up_down_left_right == 3): # right
        push_action_1.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), GT(i, l_from))))  #checks that we are currently considering the furthermost tile with a you-applied block.


  # for i in range(len(blocks)): #checks that from has at leats 1 YOU-applied block, and to has at least 1 push-applied block.
  #   for j in range(len(blocks)):
  #     if (i!= j):
  #       if(up_down_left_right == 0): # up
  #         push_action_1.add_effect(push_in_progress_up_1(), True, And(tile_holds(l_to, blocks[i]), push_applied(blocks[i]), tile_holds(l_from, blocks[j]), you_applied(blocks[j]))) #only indicate that we move to part 2 of push if l_to contains a pushable block
  #       elif(up_down_left_right == 1): # down
  #         push_action_1.add_effect(push_in_progress_down_1(), True, And(tile_holds(l_to, blocks[i]), push_applied(blocks[i]), tile_holds(l_from, blocks[j]), you_applied(blocks[j]))) #CHANGE NAME
  #       elif(up_down_left_right == 2): # left
  #         push_action_1.add_effect(push_in_progress_left_1(), True, And(tile_holds(l_to, blocks[i]), push_applied(blocks[i]), tile_holds(l_from, blocks[j]), you_applied(blocks[j]))) #CHANGE NAME
  #       elif(up_down_left_right == 3): # right
  #         push_action_1.add_effect(push_in_progress_right_1(), True, And(tile_holds(l_to, blocks[i]), push_applied(blocks[i]), tile_holds(l_from, blocks[j]), you_applied(blocks[j]))) #CHANGE NAME

    if(up_down_left_right == 0): # up
      push_action_1.add_effect(push_in_progress_up_1(), True) #only indicate that we move to part 2 of push if l_to contains a pushable block
    elif(up_down_left_right == 1): # down
      push_action_1.add_effect(push_in_progress_down_1(), True)
    elif(up_down_left_right == 2): # left
      push_action_1.add_effect(push_in_progress_left_1(), True)
    elif(up_down_left_right == 3): # right
      push_action_1.add_effect(push_in_progress_right_1(), True)


    push_action_1.add_effect(free_space_found(Plus(l_from, next_over)), True)
    push_action_1.add_effect(free_space_found_counter, 1)
    push_action_1.add_effect(push_init_pos, l_from)




def push_action_builder_2(push_action_2, up_down_left_right): #up = 0, down = 1, left = 2, right = 3
  # this action iterates one tile at a time in a given row/column to determine if a push is valid. if we reach a stop_applied block, or the border of the board, then we gracefully fail.
  l_to = push_action_2.parameter('l_to')
  # l_to_to = push_action_2.parameter('l_to_to')
  push_action_2.add_precondition(free_space_found(l_to)) #already initiated a push before, should not be allowed to start another until resolving previous push.
  # direction = above_of_index(l_to)
  next_over = 1
  progress_check = push_in_progress_up_1
  progress_check_2 = push_in_progress_up_2
  if(up_down_left_right == 0): # up
    # direction = above_of_index(l_to)
    next_over = -rows
    progress_check = push_in_progress_up_1
    progress_check_2 = push_in_progress_up_2
  elif(up_down_left_right == 1): # down
    # direction = below_of_index(l_to)
    next_over = rows
    progress_check = push_in_progress_down_1
    progress_check_2 = push_in_progress_down_2
  elif(up_down_left_right == 2): # left
    # direction = left_of_index(l_to)
    next_over = -1
    progress_check = push_in_progress_left_1
    progress_check_2 = push_in_progress_left_2
  elif(up_down_left_right == 3): # right
    # direction = right_of_index(l_to)
    next_over = 1
    progress_check = push_in_progress_right_1
    progress_check_2 = push_in_progress_right_2

  push_action_2.add_precondition(progress_check()) #already initiated a push before, should not be allowed to start another until resolving previous push.
  # push_action_2.add_precondition(Equals(direction, curr_tile_index(l_to_to))) # check that tile we are pushing to is in appropriate direction to direction of push. Implicitly also checks that l_to is not a border tile and thus has no valid push-neighbour.

  #Or check that we reached end of push last time. Then direction check is not required as we will not handle cascading push logic anymore.

  push_action_2.add_decrease_effect(free_space_found_counter(), 1) # if it reaches 0 between iterations, then for one reason or another, we do not handle the cascading logic anymore and move to part 3 of the push action.

  loop_over_tiles_builder_2(push_action_2, up_down_left_right, l_to, l_to+next_over, next_over) #employs conditional effects based on all tiles and all possible objects that a tile might be able to hold.

  push_action_2.add_effect(progress_check(), False, LE(free_space_found_counter(), 0)) #previous call of this function indicated push ended
  push_action_2.add_effect(progress_check_2(), True, LE(free_space_found_counter(), 0)) #previous call of this function indicated push ended

def loop_over_tiles_builder_2(push_action_2, up_down_left_right, l_to, l_to_to, next_over):
  direction_check = Plus(l_to_to,next_over)
  reachable = same_col(l_to_to, Plus(l_to_to, next_over))
  if(up_down_left_right == 0):
    reachable = same_col(l_to_to, Plus(l_to_to, next_over))
  elif(up_down_left_right == 1):
    reachable = same_col(l_to_to, Plus(l_to_to, next_over))
  elif(up_down_left_right == 2):
    reachable = same_row(l_to_to, Plus(l_to_to, next_over))
  elif(up_down_left_right == 3):
    reachable = same_row(l_to_to, Plus(l_to_to, next_over))
  for j in range(len(blocks)):
    push_action_2.add_increase_effect(free_space_found_counter, 1, And(tile_holds(l_to_to, blocks[j]), push_applied(blocks[j]))) # checking if tile is the end of push-loop or not: if there is such a block in l_to_to that is pushable, increase a counter. if in the next call of this action the counter != 0, continue. otherwise, end it.
    push_action_2.add_decrease_effect(free_space_found_counter, 1, Or(Or(LE(direction_check, -1), GE(direction_check, total_tiles), Not(reachable)),And(tile_holds(l_to_to, blocks[j]), stop_applied(blocks[j])))) # failure condition met - reached border of board without any free space.
    push_action_2.add_effect(carry_tile_holds(l_to_to, blocks[j]), True, And(tile_holds(l_to, blocks[j]), push_applied(blocks[j]), GT(free_space_found_counter(), 0))) #effect: pushes block that is pushable from a tile to tile above it conditions: object being inspected is pushable (i.e. tile holds block-type which is push_applied), l_to_to is directly above l_to
    push_action_2.add_effect(free_space_found(l_to), False, And(tile_holds(l_to_to, blocks[j]), push_applied(blocks[j]), Not(Or(Or(LE(direction_check, -1), GE(direction_check, total_tiles), Not(reachable)),And(tile_holds(l_to_to, blocks[j]), stop_applied(blocks[j])))))) #consider the next tile in cascade if it also contains a pushable block.
    push_action_2.add_effect(free_space_found(l_to_to), True, And(tile_holds(l_to_to, blocks[j]), push_applied(blocks[j]), GT(free_space_found_counter(), 0), Not(Or(Or(LE(direction_check, -1), GE(direction_check, total_tiles), Not(reachable)),And(tile_holds(l_to_to, blocks[j]), stop_applied(blocks[j])))))) #sets focus for next call of this action to be tile which was just pushed into.

  for i in range(len(blocks)):
    push_action_2.add_effect(failure, True, And(tile_holds(l_to_to, blocks[i]), Or(push_applied(blocks[i]), stop_applied(blocks[i])), Or(LE(direction_check, -1), GE(direction_check, total_tiles), Not(reachable)))) #push fails if we reach a stop-applied block, or reached border without seeing a blank space.

# builder_3 only cleans up fluents
def push_action_builder_3(push_action_3, up_down_left_right): #up = 0, down = 1, left = 2, right = 3
  l_from = push_action_3.parameter('l_from')
  progress_check = push_in_progress_up_1
  progress_check_2 = push_in_progress_up_2
  next_over = 1
  if(up_down_left_right == 0): # up
    next_over = -rows
    # direction = above_of_index(l_from)
    progress_check = push_in_progress_up_1
    progress_check_2 = push_in_progress_up_2
  elif(up_down_left_right == 1): # down
    next_over = rows
    # direction = below_of_index(l_from)
    progress_check = push_in_progress_down_1
    progress_check_2 = push_in_progress_down_2
  elif(up_down_left_right == 2): # left
    next_over = -1
    # direction = left_of_index(l_from)
    progress_check = push_in_progress_left_1
    progress_check_2 = push_in_progress_left_2
  elif(up_down_left_right == 3): # right
    next_over = 1
    # direction = right_of_index(l_from)
    progress_check = push_in_progress_right_1
    progress_check_2 = push_in_progress_right_2
  push_action_3.add_precondition(Not(progress_check())) #already initiated a push before, should not be allowed to start another until resolving previous push.
  push_action_3.add_precondition(progress_check_2()) #already initiated a push before, should not be allowed to start another until resolving previous push.
  # push_action_3.add_precondition(connected(l_from, l_to))
  # push_action_3.add_precondition(Equals(direction,curr_tile_index(l_to))) #to is above from

  push_action_3.add_precondition(Equals(push_init_pos, l_from))

  push_action_3.add_effect(tile_holds(l_from, baba), False, Not(failure))
  push_action_3.add_effect(tile_holds(Plus(l_from, next_over), baba), True, Not(failure))


  for i in range(total_tiles):
    push_action_3.add_effect(free_space_found(i), False)
    for j in range(len(blocks)):
      if (i + next_over < total_tiles and i + next_over > -1):
        push_action_3.add_effect(tile_holds(i, blocks[j]), False, And(carry_tile_holds(i+next_over, blocks[j]), Not(failure))) #effect: pushes block that is pushable from a tile to tile above it conditions: object being inspected is pushable (i.e. tile holds block-type which is push_applied), l_to_to is directly above l_to
      push_action_3.add_effect(tile_holds(i, blocks[j]), True, And(carry_tile_holds(i, blocks[j]), Not(failure))) #effect: pushes block that is pushable from a tile to tile above it conditions: object being inspected is pushable (i.e. tile holds block-type which is push_applied), l_to_to is directly above l_to
      push_action_3.add_effect(carry_tile_holds(i, blocks[j]), False)
      push_action_3.add_effect(check_for_new_sentences, True, And(Not(failure), carry_tile_holds(i, blocks[j]), is_text(blocks[j]))) #only need to check for change in rules if we pushed (either directly or indirectly) a text block.

  push_action_3.add_effect(free_space_found_counter, 0)
  push_action_3.add_effect(progress_check_2(), False)
  push_action_3.add_effect(failure, False)
  push_action_3.add_effect(you_tile_reviewed(l_from), True)
  push_action_3.add_effect(you_tile_reviewed(Plus(l_from, next_over)), True)
  # push_action_3.add_effect(check_for_new_sentences, True)

push_action_builder_1(push_up_1, 0)
push_action_builder_2(push_up_2, 0)
push_action_builder_3(push_up_3, 0)

push_action_builder_1(push_down_1, 1)
push_action_builder_2(push_down_2, 1)
push_action_builder_3(push_down_3, 1)

push_action_builder_1(push_left_1, 2)
push_action_builder_2(push_left_2, 2)
push_action_builder_3(push_left_3, 2)

push_action_builder_1(push_right_1, 3)
push_action_builder_2(push_right_2, 3)
push_action_builder_3(push_right_3, 3)

## check for new sentences

sentences_scan =  unified_planning.model.InstantaneousAction('sentences_scan', noun=Block, object_block=Block)
noun = sentences_scan.parameter('noun')
object_block = sentences_scan.parameter('object_block')
sentences_scan.add_precondition(check_for_new_sentences)
sentences_scan.add_precondition(Not(Or(move_all_up, move_all_down, move_all_left, move_all_right)))
sentences_scan.add_precondition(is_text(noun))
sentences_scan.add_precondition(block_text_match(noun, object_block))
sentences_scan.add_precondition(Not(noun_reviewed(noun)))
#clear before applying new rules
sentences_scan.add_effect(push_applied(object_block), False)
sentences_scan.add_effect(stop_applied(object_block), False)
sentences_scan.add_effect(win_applied(object_block), False)
sentences_scan.add_effect(you_applied(object_block), False)

for i in range(total_tiles-2): #horizontal checks (sentences read from left to right)
  sentences_scan.add_effect(push_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+1, is_block),tile_holds(i+2, push), same_row(i, i+2))) #checks: first tile is a noun, second is Is, third is adjective
  sentences_scan.add_effect(win_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+1, is_block),tile_holds(i+2, win), same_row(i, i+2))) #checks: first tile is a noun, second is Is, third is adjective
  sentences_scan.add_effect(you_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+1, is_block),tile_holds(i+2, you), same_row(i, i+2))) #checks: first tile is a noun, second is Is, third is adjective
  sentences_scan.add_effect(stop_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+1, is_block),tile_holds(i+2, stop), same_row(i, i+2))) #checks: first tile is a noun, second is Is, third is adjective

for i in range(total_tiles-(rows*2)): #vertical checks (sentences read from up to down)
  sentences_scan.add_effect(push_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+rows, is_block),tile_holds(i+(rows*2), push))) #checks: first tile is a noun, second is Is, third is adjective. no need for check if in same col because it is implicit in index definition.
  sentences_scan.add_effect(win_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+rows, is_block),tile_holds(i+(rows*2), win))) #checks: first tile is a noun, second is Is, third is adjective
  sentences_scan.add_effect(you_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+rows, is_block),tile_holds(i+(rows*2), you))) #checks: first tile is a noun, second is Is, third is adjective
  sentences_scan.add_effect(stop_applied(object_block), True, And(tile_holds(i, noun),tile_holds(i+rows, is_block),tile_holds(i+(rows*2), stop))) #checks: first tile is a noun, second is Is, third is adjective

sentences_scan.add_increase_effect(nouns_reviewed_counter, 1)
sentences_scan.add_effect(noun_reviewed(noun), True, Not(scan_over_in_next_call))
for i in range(len(text_blocks)):
  sentences_scan.add_effect(noun_reviewed(text_blocks[i]), False, scan_over_in_next_call) #cleanup when sentence reviews are over.
sentences_scan.add_decrease_effect(nouns_reviewed_counter, Plus(nouns_reviewed_counter,1), scan_over_in_next_call) #end check
sentences_scan.add_effect(scan_over_in_next_call, True, Equals(Plus(nouns_reviewed_counter,2), num_nouns))
sentences_scan.add_effect(check_for_new_sentences, False, scan_over_in_next_call) #end check
sentences_scan.add_effect(scan_over_in_next_call, False, scan_over_in_next_call) #end check
sentences_scan.add_effect(move_all_up, False, scan_over_in_next_call) #signal to solver to select new movement direction
sentences_scan.add_effect(move_all_down, False, scan_over_in_next_call) #signal to solver to select new movement direction
sentences_scan.add_effect(move_all_left, False, scan_over_in_next_call) #signal to solver to select new movement direction
sentences_scan.add_effect(move_all_right, False, scan_over_in_next_call) #signal to solver to select new movement direction

do_nothing =  unified_planning.model.InstantaneousAction('do_nothing', l_from=IntType(0,(total_tiles)-1)) #used in case YOU block is in a border tile
l_from = do_nothing.parameter('l_from')
#do_nothing.add_precondition(Or(move_all_up, move_all_down, move_all_left, move_all_right)) # solver indicated what direction to pursue for all YOU blocks.
border_condition_up = And(move_all_up, LE(Plus(l_from, -rows),-1)) # effect condition: set reviewed to True if move_all_up is True and above_of_index == -1.
border_condition_down = And(move_all_down, GE(Plus(l_from, rows),total_tiles)) # effect condition: set reviewed to True if move_all_up is True and above_of_index == -1.
border_condition_left = And(move_all_left, LE(Plus(l_from, -1),-1)) # effect condition: set reviewed to True if move_all_up is True and above_of_index == -1.
border_condition_right = And(move_all_right, GE(Plus(l_from, 1),total_tiles)) # effect condition: set reviewed to True if move_all_up is True and above_of_index == -1.
do_nothing.add_precondition(Or(border_condition_up, border_condition_down, border_condition_left, border_condition_right))
for i in range(total_tiles):
    for j in range(len(blocks)):
        do_nothing.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), LT(i, l_from), move_all_up)))  #checks that we are currently considering the furthermost tile with a you-applied block.
        do_nothing.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), GT(i, l_from), move_all_down)))  #checks that we are currently considering the furthermost tile with a you-applied block.
        do_nothing.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), LT(i, l_from), move_all_left)))  #checks that we are currently considering the furthermost tile with a you-applied block.
        do_nothing.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]), GT(i, l_from), move_all_right)))  #checks that we are currently considering the furthermost tile with a you-applied block.
#for i in range(0, 1):
  #for j in range(len(blocks)):
    # direction_has_free_tile_check_up = And(tile_holds(i, blocks[j]), Or(push_applied(blocks[j]), stop_applied(blocks[j])), same_col(l_from, tiles[i]),  GT(curr_tile_index(l_from), curr_tile_index(tiles[i])), move_all_up) #condition says: it is not the case that there exists a free tile (no stop or push blocks) in the push direction specified by move all. (in this case for move_up, there is no free space above current tile (same_col, and with a lower index)).
    # direction_has_free_tile_check_down = And(tile_holds(i, blocks[j]), Or(push_applied(blocks[j]), stop_applied(blocks[j])), same_col(l_from, tiles[i]),  LT(curr_tile_index(l_from), curr_tile_index(tiles[i])), move_all_down) #condition says: it is not the case that there exists a free tile (no stop or push blocks) in the push direction specified by move all. (in this case for move_up, there is no free space above current tile (same_col, and with a lower index)).
    # direction_has_free_tile_check_left = And(tile_holds(i, blocks[j]), Or(push_applied(blocks[j]), stop_applied(blocks[j])), same_row(l_from, tiles[i]),  GT(curr_tile_index(l_from), curr_tile_index(tiles[i])), move_all_left) #condition says: it is not the case that there exists a free tile (no stop or push blocks) in the push direction specified by move all. (in this case for move_up, there is no free space above current tile (same_col, and with a lower index)).
    #direction_has_free_tile_check_right = And(Implies(And(same_row(l_from, tiles[i]),  LT(curr_tile_index(l_from), curr_tile_index(tiles[i]))), And(tile_holds(i, blocks[j]), Or(push_applied(blocks[j]), stop_applied(blocks[j])))), move_all_right) #condition says: it is not the case that there exists a free tile (no stop or push blocks) in the push direction specified by move all. (in this case for move_up, there is no free space above current tile (same_col, and with a lower index)).
    #direction_has_free_tile_check_right = And(Implies(And(False), And(False)), move_all_right) #condition says: it is not the case that there exists a free tile (no stop or push blocks) in the push direction specified by move all. (in this case for move_up, there is no free space above current tile (same_col, and with a lower index)).
    #do_nothing.add_precondition(direction_has_free_tile_check_right) #either block must be in border or no valid push available
do_nothing.add_effect(you_tile_reviewed(l_from), True)


reset_all =  unified_planning.model.InstantaneousAction('reset_all') #called after we complete all relevant movements from previous move_all action. Not must select new direction.
reset_all.add_precondition(Not(check_for_new_sentences))
for i in range(total_tiles):
  for j in range(len(blocks)):
    reset_all.add_precondition(Not(And(Not(you_tile_reviewed(i)), tile_holds(i, blocks[j]), you_applied(blocks[j]))))  #checks there are no more tiles left to consider for current move_all
for i in range(total_tiles):
  reset_all.add_effect(you_tile_reviewed(i), False)
reset_all.add_effect(move_all_up, False)
reset_all.add_effect(move_all_down, False)
reset_all.add_effect(move_all_left, False)
reset_all.add_effect(move_all_right, False)

# defining the problem
problem = unified_planning.model.Problem('baba_is_you')
# problem.add_fluent(connected, default_initial_value=False)
# problem.add_fluent(tiles, default_initial_value=baba)
problem.add_fluent(same_col, default_initial_value=False)
problem.add_fluent(same_row, default_initial_value=False)
# problem.add_fluent(left_of_index, default_initial_value=-1)
# problem.add_fluent(right_of_index, default_initial_value=-1)
# problem.add_fluent(above_of_index, default_initial_value=-1)
# problem.add_fluent(below_of_index, default_initial_value=-1)
# problem.add_fluent(curr_tile_index, default_initial_value=-1)
problem.add_fluent(tile_holds, default_initial_value=False)
problem.add_fluent(carry_tile_holds, default_initial_value=False)
problem.add_fluent(num_nouns, default_initial_value=nouns)
problem.add_fluent(nouns_reviewed_counter, default_initial_value=0)
problem.add_fluent(failure, default_initial_value=False)
problem.add_fluent(success, default_initial_value=False)
problem.add_fluent(free_space_found_counter, default_initial_value=0)
problem.add_fluent(free_space_found, default_initial_value=False)
problem.add_fluent(push_init_pos, default_initial_value=0)
problem.add_fluent(push_in_progress_last_l_to_index, default_initial_value=0)
problem.add_fluent(push_in_progress_up_1, default_initial_value=False)
problem.add_fluent(push_in_progress_up_2, default_initial_value=False)
problem.add_fluent(push_in_progress_down_1, default_initial_value=False)
problem.add_fluent(push_in_progress_down_2, default_initial_value=False)
problem.add_fluent(push_in_progress_left_1, default_initial_value=False)
problem.add_fluent(push_in_progress_left_2, default_initial_value=False)
problem.add_fluent(push_in_progress_right_1, default_initial_value=False)
problem.add_fluent(push_in_progress_right_2, default_initial_value=False)
problem.add_fluent(move_all_up, default_initial_value=False)
problem.add_fluent(move_all_down, default_initial_value=False)
problem.add_fluent(move_all_left, default_initial_value=False)
problem.add_fluent(move_all_right, default_initial_value=False)
problem.add_fluent(you_tile_reviewed, default_initial_value=False)
problem.add_fluent(check_for_new_sentences, default_initial_value=True) #by default when starting a new level, always check rules first before first move.
problem.add_fluent(scan_over_in_next_call, default_initial_value=False) #by default when starting a new level, always check rules first before first move.
problem.add_fluent(check_for_win, default_initial_value=False) #by default when starting a new level, always check rules first before first move.
problem.add_fluent(noun_reviewed, default_initial_value=False)
problem.add_fluent(push_applied, default_initial_value=False)
problem.add_fluent(stop_applied, default_initial_value=False)
problem.add_fluent(win_applied, default_initial_value=False)
problem.add_fluent(you_applied, default_initial_value=False)
problem.add_fluent(block_text_match, default_initial_value=False)
problem.add_fluent(is_text, default_initial_value=False)
problem.add_action(move_all_select_up)
problem.add_action(move_all_select_down)
problem.add_action(move_all_select_left)
problem.add_action(move_all_select_right)
problem.add_action(move_up)
problem.add_action(move_down)
problem.add_action(move_left)
problem.add_action(move_right)
problem.add_action(push_up_1)
problem.add_action(push_up_2)
problem.add_action(push_up_3)
problem.add_action(push_down_1)
problem.add_action(push_down_2)
problem.add_action(push_down_3)
problem.add_action(push_left_1)
problem.add_action(push_left_2)
problem.add_action(push_left_3)
problem.add_action(push_right_1)
problem.add_action(push_right_2)
problem.add_action(push_right_3)
problem.add_action(sentences_scan)
problem.add_action(do_nothing)
problem.add_action(reset_all)


# problem.add_objects(tiles)
problem.add_objects(blocks)

# defining tile connectivity

problem.set_initial_value(tile_holds(0,baba), True) # INITIAL POSITION OF BABA
for i in range(cols*rows - 1): #establishing horizontal adjacency - left, right moving
    # if((i+1) % rows != 0): #if i == right-most col, skip
      # problem.set_initial_value(connected(tiles[i], tiles[i+1]), True)
      # problem.set_initial_value(connected(tiles[i+1], tiles[i]), True)
      # problem.set_initial_value(left_of_index(tiles[i+1]), i) #i+1 is to the right of i
      # problem.set_initial_value(right_of_index(tiles[i]), i+1) #i is to the left of i+1
      # problem.set_initial_value(curr_tile_index(tiles[i]), i)
      # problem.set_initial_value(curr_tile_index(tiles[i+1]), i+1)
    for j in range(cols*rows): # can be made more efficient TODO
      #identify all tiles[j] that are in the same row as tiles[i]
      if(((math.floor(float(i)/float(rows))) == (math.floor(float(j)/float(rows)))) and i != j): #if the result (ignoring remainder) is the same, then they are in the same row
        problem.set_initial_value(same_row(i, j), True)
        problem.set_initial_value(same_row(j, i), True)

for i in range(cols*(rows-1)): #establishing vertical adjacency - up, down moving
    # problem.set_initial_value(connected(tiles[i], tiles[i+rows]), True)
    # problem.set_initial_value(connected(tiles[i+rows], tiles[i]), True)
    # problem.set_initial_value(below_of_index(tiles[i]), i+rows) #i+rows is below i
    # problem.set_initial_value(above_of_index(tiles[i+rows]), i) #i is above i+rows
    for j in range(cols*rows):
      #identify all tiles[j] that are in the same column as tiles[i]
      if(((i%rows) == (j%rows)) and i != j): #if modulo rows is the same for both, then they are in the same column
        problem.set_initial_value(same_col(i, j), True)
        problem.set_initial_value(same_col(j, i), True)

# SENTENCE SET UP

problem.set_initial_value(block_text_match(baba_text, baba), True)
problem.set_initial_value(block_text_match(wall_text, wall), True)
problem.set_initial_value(block_text_match(flag_text, flag), True)

for i in range(len(text_blocks)):
  problem.set_initial_value(push_applied(text_blocks[i]), True) #all text blocks can be pushed.
  problem.set_initial_value(is_text(text_blocks[i]), True) #all text are text.


win_loc = 23
for i in range(total_tiles):
  if (i % rows == 1 or i % rows == 2):
    problem.set_initial_value(tile_holds(i, wall), True)

# TEST SENTENCE POSITIONS
problem.set_initial_value(tile_holds(3, wall_text), True)
problem.set_initial_value(tile_holds(4, is_block), True)
problem.set_initial_value(tile_holds(5, push), True)
problem.set_initial_value(tile_holds(9, baba_text), True)
problem.set_initial_value(tile_holds(10, is_block), True)
problem.set_initial_value(tile_holds(11, you), True)
problem.set_initial_value(tile_holds(15, flag_text), True)
problem.set_initial_value(tile_holds(16, is_block), True)
problem.set_initial_value(tile_holds(17, win), True)

problem.set_initial_value(tile_holds(6, flag), True)
problem.set_initial_value(tile_holds(win_loc, flag), False)


#print(win_loc)

possible_tile = Variable("possible_tile", IntType(0, total_tiles))
possible_block_1 = Variable("possible_block_1", Block)
possible_block_2 = Variable("possible_block_2", Block)

# problem.add_goal(And(tile_holds(tiles[6], baba), tile_holds(tiles[6], flag), you_applied(baba), win_applied(flag))) #block is in same location as win

problem.add_goal(Exists(And(tile_holds(possible_tile, possible_block_1), tile_holds(possible_tile, possible_block_2), you_applied(possible_block_1), win_applied(possible_block_2)), possible_tile, possible_block_1, possible_block_2)) #block is in same location as win
# print(problem)

## Compilation of problem
now = datetime.datetime.now()
original_problem = problem
original_problem_kind = problem.kind
## below adapted form https://github.com/stacs-cp/unified-planning/blob/new_types2/experiments/plotting/Plotting.py#L247
compilation_kinds_to_apply = [
    # CompilationKind.ARRAYS_REMOVING,
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    # CompilationKind.BOUNDED_TYPES_REMOVING,#INTEGERS_REMOVING
    CompilationKind.INTEGERS_REMOVING,#INTEGERS_REMOVING
    #CompilationKind.USERTYPE_FLUENTS_REMOVING,
]

iteration = 0
results = []
for ck in compilation_kinds_to_apply:
    print(str(iteration) + " COMPILATION START " + str(now))
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
    now = datetime.datetime.now()
    print(str(iteration) + " COMPILATION DONE " + str(now))
    iteration += 1



## end of adapted code


# writer = PDDLWriter(problem)
# writer.write_domain("proto_domain.pddl")
# writer.write_problem("proto_problem.pddl")

# TEMP SIM

array_of_moves = []

def add_reset_all():
  array_of_moves.append(ActionInstance(reset_all))
  # add_sentence_review()

def add_NOOP(tileNum1):
  array_of_moves.append(ActionInstance(do_nothing, (tileNum1)))

# def add_win_scan():
#   array_of_moves.append(ActionInstance(win_scan, (tiles[6], baba, flag)))

def add_sentence_review():
  array_of_moves.append(ActionInstance(sentences_scan, (baba_text, baba)))
  array_of_moves.append(ActionInstance(sentences_scan, (wall_text, wall)))
  array_of_moves.append(ActionInstance(sentences_scan, (flag_text, flag)))

def add_move_all_up():
  array_of_moves.append(ActionInstance(move_all_select_up))

def add_move_all_down():
  array_of_moves.append(ActionInstance(move_all_select_down))

def add_move_all_left():
  array_of_moves.append(ActionInstance(move_all_select_left))

def add_move_all_right():
  array_of_moves.append(ActionInstance(move_all_select_right))

def add_move_up(tileNum1, tileNum2):
  array_of_moves.append(ActionInstance(move_up, (tileNum1, baba)))

def add_move_down(tileNum1, tileNum2):
  array_of_moves.append(ActionInstance(move_down, (tileNum1, baba)))

def add_move_left(tileNum1, tileNum2):
  array_of_moves.append(ActionInstance(move_left, (tileNum1, baba)))

def add_move_right(tileNum1, tileNum2):
  array_of_moves.append(ActionInstance(move_right, (tileNum1, baba)))

def add_push_right_1(tileNum1, tileNum2):
  array_of_moves.append(ActionInstance(push_right_1, (tileNum1, baba, wall)))

def add_push_right_2(tileNum2, tileNum3):
  array_of_moves.append(ActionInstance(push_right_2, (tileNum2, tileNum3)))

def add_push_right_3(tileNum1, tileNum2):
  array_of_moves.append(ActionInstance(push_right_3, (tileNum1)))


add_sentence_review()
# add_move_all_right()
# #add_NOOP(0)
# add_push_right_1(0,1)
# add_push_right_2(1,2)
# add_push_right_2(2,3)
# add_push_right_2(3,4)
# add_push_right_2(4,5)
# add_push_right_2(4,5)
# add_push_right_3(0,1)
# add_reset_all()
# add_move_all_left()
# add_NOOP(0)
# add_reset_all()
add_move_all_down()
add_move_down(0, 6)

plan = SequentialPlan(
        array_of_moves
      )
def plan_debug(plan, debug):
  with SequentialSimulator(problem=problem) as simulator:
      # print(problem.kind)
      initial_state = simulator.get_initial_state()
      current_state = initial_state
      # We also store the states to plot the metrics later
      states = [current_state]
      for action_instance in plan.actions:
          possible_moves = simulator.get_applicable_actions(current_state)
          # for move in (possible_moves):
          #   print(move)
          print(f'NEXT______: {action_instance}')
          current_state = simulator.apply(current_state, action_instance)

          if current_state is None:
              print(f'Error in applying: {action_instance}')
              break
          states.append(current_state)
          if (debug == TRUE):
            print("counter free: " + str(current_state.get_value(free_space_found_counter()).constant_value()))
            print("counter noun: " + str(current_state.get_value(nouns_reviewed_counter()).constant_value()))
            print("push_in_progress_up_1: " + str(current_state.get_value(push_in_progress_up_1()).constant_value()))
            print("push_in_progress_up_2: " + str(current_state.get_value(push_in_progress_up_2()).constant_value()))
            print("push_in_progress_down_1: " + str(current_state.get_value(push_in_progress_down_1()).constant_value()))
            print("push_in_progress_down_2: " + str(current_state.get_value(push_in_progress_down_2()).constant_value()))
            print("push_in_progress_left_1: " + str(current_state.get_value(push_in_progress_left_1()).constant_value()))
            print("push_in_progress_left_2: " + str(current_state.get_value(push_in_progress_left_2()).constant_value()))
            print("push_in_progress_right_1: " + str(current_state.get_value(push_in_progress_right_1()).constant_value()))
            print("push_in_progress_right_2: " + str(current_state.get_value(push_in_progress_right_2()).constant_value()))
            print("sentence scan to do: " + str(current_state.get_value(check_for_new_sentences()).constant_value()))
            print("scan over in next call: " + str(current_state.get_value(scan_over_in_next_call()).constant_value()))
            print("win scan to do: " + str(current_state.get_value(check_for_win()).constant_value()))
            print("failed: " + str(current_state.get_value(failure()).constant_value()))
            print("won: " + str(current_state.get_value(success()).constant_value()))
            for i in range(len(non_text_blocks)):
              print(str(non_text_blocks[i]) + ": push:" + str(current_state.get_value(push_applied(non_text_blocks[i])).constant_value()) + "/ stop:" + str(current_state.get_value(stop_applied(non_text_blocks[i])).constant_value()) + "/ win:" + str(current_state.get_value(win_applied(non_text_blocks[i])).constant_value()) + "/ you:" + str(current_state.get_value(you_applied(non_text_blocks[i])).constant_value())) #candidate_for_pull
          output = ""
          print()
          for i in range(total_tiles):
            number = 0;
            for j in range(len(blocks)):
              if(current_state.get_value(tile_holds(i, blocks[j])).constant_value()): #if tile holds block type, print name
                number += 1
                if (number > 1):
                  output += "/" + str(blocks[j])
                else:
                  output += str(blocks[j])
            if (number == 0):
              output += "____"
            output += ","
            if (i != 0 and i % rows == rows-1):
              print(output)
              output = ""
          # in current_battery_value we inspect the State
          print()
          if (simulator.is_goal(current_state)):
            print("!COMPLETED LEVEL!")
            break



####
# plan_debug(plan, FALSE)

# ## solve
# # with OneshotPlanner(names=['enhsp','enhsp'], params=[{'heuristic': 'hadd'}, {'heuristic': 'hmax'}]) as planner:
# with OneshotPlanner(problem_kind=problem.kind) as planner:
#     result = planner.solve(problem)
#     if result.plan is not None:
#         print("Plan returned: %s" % result.plan)
#         print("Plan logs: %s" % result.log_messages)
#         plan_debug(result.plan)
#     else:
#         print("No plan found. %s" % result.log_messages)

# more complicated problem - hide flag behind wall.
# problem.set_initial_value(tile_holds(tiles[6], flag), False)
# problem.set_initial_value(tile_holds(tiles[win_loc], flag), True)

# TEMP SIM

# array_of_moves = [];

# add_sentence_review()
# add_move_all_down()
# add_move_down(0, 6)
# add_reset_all()
# add_move_all_down()
# add_move_down(6, 12)
# add_reset_all()
# add_move_all_down()
# add_move_down(12, 18)
# add_reset_all()
# add_move_all_down()
# add_move_down(18, 24)
# add_reset_all()
# add_move_all_down()
# add_move_down(24, 30)
# add_reset_all()
# add_move_all_right()
# add_push_right_1(30, 31)
# add_push_right_2(31, 32)
# add_push_right_2(32, 33)
# add_push_right_2(32, 33)
# add_push_right_3(30, 31)
# add_reset_all()
# add_move_all_right()
# add_push_right_1(31, 32)
# add_push_right_2(32, 33)
# add_push_right_2(33, 34)
# add_push_right_2(33, 34)
# add_push_right_3(31, 32)
# add_reset_all()
# add_move_all_right()
# add_push_right_1(32, 33)
# add_push_right_2(33, 34)
# add_push_right_2(34, 35)
# add_push_right_2(34, 35)
# add_push_right_3(32, 33)
# add_reset_all()
# add_move_all_up()
# add_move_up(33, 27)
# add_reset_all()
# add_move_all_up()
# add_move_up(27, 21)
# add_reset_all()
# add_move_all_right()
# add_move_right(21, 22)
# add_reset_all()
# add_move_all_right()
# add_move_right(22, 23)


# plan = SequentialPlan(
#         array_of_moves
#       )

####
#plan_debug(plan, FALSE)

## solve
# with OneshotPlanner(names=['enhsp','enhsp'], params=[{'heuristic': 'hadd'}, {'heuristic': 'hmax'}]) as planner:
#with OneshotPlanner(problem_kind=problem.kind) as planner:
#    result = planner.solve(problem)
#    if result.plan is not None:
#        print("Plan returned: %s" % result.plan)
#        print("Plan logs: %s" % result.log_messages)
#        plan_debug(result.plan)
#    else:
#        print("No plan found. %s" % result.log_messages)
