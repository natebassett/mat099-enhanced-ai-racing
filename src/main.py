# src/main/main.py

from agents.random_agent import RandomAgent


def main():
    print("=" * 40)
    print("Enhanced AI Racing")
    print("=" * 40)

    print("\nChoose an agent:")
    print("[1] Random Agent")
    print("[0] Exit")

    choice = input("\nEnter choice: ")

    if choice == "1":
        agent = RandomAgent(seed=42)
        print("\nRandom Agent selected.")
        print("Launching TORCS on Corkscrew...")
        
        # Later this becomes:
        # run_torcs(agent, track="corkscrew")

        for step in range(10):
            action = agent.act(None)
            print(f"Step {step}: steering={action[0]:.2f}, throttle={action[1]:.2f}, brake={action[2]:.2f}")

    elif choice == "0":
        print("Exiting.")
    else:
        print("Invalid choice.")


if __name__ == "__main__":
    main()