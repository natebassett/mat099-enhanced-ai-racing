from agents.random_agent import RandomAgent
from agents.rule_based_agent import RuleBasedAgent
from runner.torcs_runner import TorcsRunner
from storage import RaceRepository


def print_results(results):

    print("\n==============================")
    print("Race Results")
    print("==============================")

    print(f"Steps       : {results['steps']}")
    print(f"Total Score : {results['total_score']:.2f}")
    print(f"Max Speed   : {results['max_speed']:.2f}")
    print(f"Avg Speed   : {results['avg_speed']:.2f}")
    print(f"Off Track   : {results['off_track']}")
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
        print("[0] Exit\n")

        choice = input("Enter choice: ")

        if choice in {"1", "2"}:

            runner = TorcsRunner()

            agent = RandomAgent() if choice == "1" else RuleBasedAgent()

            agent_id = repository.register_agent(
                name=agent.name,
                agent_type=agent.agent_type,
                version=agent.version,
                config=agent.config,
            )

            try:
                runner.launch()

                runner.connect()

                runner.load_track("corkscrew")

                results = runner.run(agent)
            finally:
                runner.shutdown()

            run_id = repository.record_run(
                agent_id=agent_id,
                track="corkscrew",
                seed=agent.seed,
                results=results,
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
