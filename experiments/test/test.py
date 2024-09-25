from unified_planning.shortcuts import *

problem = Problem('problem')

Person = UserType('Person')
carla = Object('carla', Person)
judith = Object('judith', Person)
problem.add_objects([carla, judith])
enters = Fluent('enters', IntType(0,5), p=Person)
problem.add_fluent(enters, default_initial_value=0)

track = Fluent('track', BoolType(), km=IntType(0,20), fuel=IntType(0, 4))
problem.add_fluent(track, default_initial_value=False)
problem.set_initial_value(track(20, 4), True)

puzzle = Fluent('puzzle', ArrayType(3, IntType(0,8)))
problem.add_fluent(puzzle)
problem.set_initial_value(puzzle, [8,0,6])

from unified_planning.engines import CompilationKind
# The CompilationKind class is defined in the unified_planning/engines/mixins/compiler.py file
compilation_kinds_to_apply = [
    CompilationKind.INT_PARAMETER_ACTIONS_REMOVING,
    CompilationKind.ARRAYS_REMOVING,
    CompilationKind.INTEGERS_REMOVING,
    #CompilationKind.USERTYPE_FLUENTS_REMOVING,
    #CompilationKind.CONDITIONAL_EFFECTS_REMOVING,
]

original_problem = problem
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
      # After we have the compiler, we get the compilation result
      result = compiler.compile(
          problem,
          ck
      )
      results.append(result)
      problem = result.problem