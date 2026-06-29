from agents.random_agent import RandomAgent
from runner.torcs_runner import TorcsRunner


def print_results(results):

    print("\n==============================")
    print("Race Results")
    print("==============================")

    print(f"Steps       : {results['steps']}")
    print(f"Total Score : {results['reward']:.2f}")
    print(f"Max Speed   : {results['max_speed']:.2f}")
    print(f"Avg Speed   : {results['avg_speed']:.2f}")
    print(f"Off Track   : {results['off_track']}")

    print("==============================\n")


def menu():

    while True:

        print("========================================")
        print(" Enhanced AI Racing")
        print("========================================\n")

        print("Choose an agent")
        print("[1] Random Agent")
        print("[0] Exit\n")

        choice = input("Enter choice: ")

        if choice == "1":

            runner = TorcsRunner()

            agent = RandomAgent()

            try:
                runner.launch()

                runner.connect()

                runner.load_track("corkscrew")

                results = runner.run(agent)
            finally:
                runner.shutdown()

            print_results(results)

            input("Press ENTER to continue...")

        elif choice == "0":

            break

        else:

            print("\nInvalid option.\n")


if __name__ == "__main__":
    menu()
