from unified_planning.shortcuts import *
from unified_planning.engines import CompilationKind
import signal
import time

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

def compile_and_solve(problem, solving, compilation_kinds_to_apply=None):
    start_time = time.time()
    signal.signal(signal.SIGALRM, handler)
    up.shortcuts.get_environment().credits_stream = None
    results = []
    timeout_seconds = 1800

    try:
        signal.alarm(timeout_seconds)

        if solving == 'fast-downward-opt':
            compilation_kinds_to_apply.append(CompilationKind.CONDITIONAL_EFFECTS_REMOVING)

        # Compilation
        for ck in compilation_kinds_to_apply:
            params = {}
            if ck == CompilationKind.ARRAYS_REMOVING:
                params = {'mode': 'permissive'}
            with Compiler(problem_kind=problem.kind, compilation_kind=ck, params=params) as compiler:
                result = compiler.compile(
                    problem,
                    ck
                )
                results.append(result)
                problem = result.problem

        signal.alarm(0)
        mid_time = time.time()
        print(f'UP Compilation Time:', mid_time - start_time)

        # Solving
        try:
            if solving in ['enhsp-any', 'any_fast-downward', 'any_symk']:
                solver_name = solving.split('_')[1]
                with AnytimePlanner(name=solver_name) as planner:
                    for res in planner.get_solutions(problem, timeout=timeout_seconds):
                        compiled_plan = res.plan
                        for result in reversed(results):
                            compiled_plan = compiled_plan.replace_action_instances(result.map_back_action_instance)
                        print(compiled_plan)
                print(f'Planner Time:', time.time() - mid_time)

            elif solving in ['fast-downward', 'symk', 'enhsp', 'fast-downward-opt', 'symk-opt', 'enhs-opt']:
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
                        print('No solution found.')
                    if not planner.supports(problem.kind):
                        unsupported_features = [
                            f"{pk} is not supported by the planner"
                            for pk in problem.kind.features if pk not in planner.supported_kind().features
                        ]
                        print("\n".join(unsupported_features))
                signal.alarm(0)
                print(f'Planner Time:', time.time() - mid_time)

        except TimeoutException:
            print('Planner Time: timeout')
        except Exception as e:
            print(f"Error encountered: {e}")
            exit(1)

    except TimeoutException:
        print(f'UP Compilation Time: timeout')

    except Exception as e:
        print(f"Error encountered: {e}")
        exit(1)