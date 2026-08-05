from agents.random_agent import RandomAgent
from agents.map_aware_agent import MapAwareAgent
from agents.rule_based_agent import RuleBasedAgent
from agents.dyna_q_agent import DynaQFinalisedAgent, DynaQLearningAgent
from runner.torcs_runner import TorcsRunner
from storage import RaceRepository


DEFAULT_TRACK_NAME = "g-track-3"


def print_results(results):

    print("\n==============================")
    print("Race Results")
    print("==============================")

    print(f"Steps       : {results['steps']}")
    print(f"Total Score : {results['total_score']:.2f}")
    print(f"Max Speed   : {results['max_speed'] * 50:.2f} km/h")
    print(f"Avg Speed   : {results['avg_speed'] * 50:.2f} km/h")
    print(f"Off Track   : {results['off_track']}")
    print(f"Laps        : {results['laps_completed']}")
    if results["best_lap_time_seconds"] is not None:
        print(f"Best Lap    : {results['best_lap_time_seconds']:.3f} seconds")
    print(f"Ended By    : {results['termination_reason']}")

    print("==============================\n")


def menu():

    repository = RaceRepository()

    while True:

        print("========================================")
        print(" Enhanced AI Racing")
        print("========================================\n")

        print("Choose an agent")
        print("[1] Random Agent")
        print("[2] Rule-Based Anti-Spin Agent")
        print("[3] Map-Aware Racing-Line Agent")
        print("[4] Dyna-Q Learning Agent")
        print("[5] Dyna-Q Finalised Agent")
        print("[0] Exit\n")

        choice = input("Enter choice: ")

        if choice in {"1", "2", "3", "4", "5"}:

            runner = TorcsRunner()

            agents = {
                "1": RandomAgent,
                "2": RuleBasedAgent,
                "3": MapAwareAgent,
                "4": DynaQLearningAgent,
                "5": DynaQFinalisedAgent,
            }
            agent = agents[choice]()

            agent_id = repository.register_agent(
                name=agent.name,
                agent_type=agent.agent_type,
                version=agent.version,
                config=agent.config,
            )

            try:
                runner.launch()

                runner.connect()

                runner.load_track(DEFAULT_TRACK_NAME)

                results = runner.run(agent)
            finally:
                runner.shutdown()

            run_id = repository.record_run(
                agent_id=agent_id,
                track=DEFAULT_TRACK_NAME,
                seed=agent.seed,
                results=results,
            )
            repository.record_run_telemetry(
                run_id,
                results.get("telemetry_samples", []),
            )

            print_results(results)

            print(f"Saved as race run #{run_id}.\n")

            input("Press ENTER to continue...")

        elif choice == "0":

            break

        else:

            print("\nInvalid option.\n")


if __name__ == "__main__":
    menu()
