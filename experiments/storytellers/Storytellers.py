from unified_planning.shortcuts import *
from experiments import compilation_solving
import argparse

# Run: python -m experiments.storytellers.Storytellers --compilation sets --solving fast-downward

# A set of storytellers tell their stories to a collection of different audiences.
# The storytellers know different (possibly intersecting) sets of stories.
# The audiences begin having heard none of the stories.
# Entertaining an audience leaves it having heard all the stories a storyteller knows.
# A storyteller might tell stories an audience has already heard, adding nothing to the stories the audience knows.

# Parser
parser = argparse.ArgumentParser(description="Solve Storytellers")
parser.add_argument('--compilation', type=str, help='Compilation strategy to apply')
parser.add_argument('--solving', type=str, help='Planner to use')

args = parser.parse_args()
compilation = args.compilation
solving = args.solving

# ------------------------------------------------ Problem -------------------------------------------------------------
storytellers_problem = unified_planning.model.Problem('storytellers_problem')

# (:functions
    # (pos ?tru - truck) - location
    # (at ?loc - location) - set of package
    # (in ?tru - truck) - set of package
    # (connects ?loc - location) - set of location)

Storyteller = UserType('Storyteller')
st1 = Object('st1', Storyteller)
st2 = Object('st2', Storyteller)
st3 = Object('st3', Storyteller)
st4 = Object('st4', Storyteller)
st5 = Object('st5', Storyteller)

Audience = UserType('Audiences')
a1 = Object('a1', Audience)
a2 = Object('a2', Audience)

Stories = UserType('Stories')
s1 = Object('s1', Stories)
s2 = Object('s2', Stories)
s3 = Object('s3', Stories)
s4 = Object('s4', Stories)
s5 = Object('s5', Stories)
s6 = Object('s6', Stories)
s7 = Object('s7', Stories)
s8 = Object('s8', Stories)
s9 = Object('s9', Stories)
s10 = Object('s10', Stories)

storytellers_problem.add_objects([st1, st2, st3, st4, st5, a1, a2])
objects = [s1, s2, s3, s4, s5, s6, s7, s8, s9, s10]
storytellers_problem.add_objects(objects)

known = Fluent('known', SetType(Stories), st=Storyteller)
heard = Fluent('heard', SetType(Stories), a=Audience)
story_set = Fluent('story_set', SetType(Stories))

storytellers_problem.add_fluent(known, default_initial_value=set())
storytellers_problem.add_fluent(heard, default_initial_value=set())
storytellers_problem.add_fluent(story_set, default_initial_value=set())

# initial state
storytellers_problem.set_initial_value(known(st1), {s1,s2})
storytellers_problem.set_initial_value(known(st2), {s2,s9})
storytellers_problem.set_initial_value(known(st3), {s3,s4})
storytellers_problem.set_initial_value(known(st4), {s5,s9,s3})
storytellers_problem.set_initial_value(known(st5), {s6,s7,s8,s2,s10})
storytellers_problem.set_initial_value(known(st5), {s6,s7,s8,s2,s10})
storytellers_problem.set_initial_value(story_set, {*objects})

# (:action entertain
# :parameters (?t - storyteller ?a - audience)
# :precondition (true)
# :effect ((heard ?a) := (union (heard ?a) (known ?t))))
entertain = InstantaneousAction('entertain', st=Storyteller, a=Audience)
st = entertain.parameter('st')
a = entertain.parameter('a')
entertain.add_effect(heard(a), SetUnion(heard(a), known(st)))
storytellers_problem.add_action(entertain)

# 2 different goals with the same basic problems:
# Both of these goals require the audiences to hear at least half of the stories.
a_var = Variable('a_var', Audience)
# possible to do with: SetCardinality(story_set) / len({*objects})
storytellers_problem.add_goal(
    Forall(
        GE(
            SetCardinality(heard(a_var)), 5
        ),
        a_var
    )
)

#   - saturation: all the stories have to be heard by at least one of the audiences.
#storytellers_problem.add_goal(
#    Exists(
#        Equals(heard(a_var), story_set),
#        a_var)
#)
#   - equality: all the audiences hear the same stories.
a_var2 = Variable('a_var2', Audience)
storytellers_problem.add_goal(
    Forall(
        Equals(heard(a_var), heard(a_var2)),
        a_var, a_var2
    )
)

compilation_solving.compile_and_solve(storytellers_problem, solving, compilation)