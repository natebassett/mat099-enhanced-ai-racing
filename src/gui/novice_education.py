from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NoviceAgentGuide:
    badge: str
    headline: str
    driving_story: tuple[str, ...]
    decision_steps: tuple[str, ...]
    learning_story: tuple[str, ...]
    input_signals: tuple[str, ...]
    key_takeaways: tuple[str, ...]
    strengths: tuple[str, ...]
    failure_signs: tuple[str, ...]


def build_novice_agent_guide(agent_type: str) -> NoviceAgentGuide:
    if agent_type == "map_aware":
        return NoviceAgentGuide(
            badge="Plans ahead with a track guide",
            headline=(
                "Think of this driver as having corner notes. It knows the planned "
                "route, then uses live sensors to stay safe while following it."
            ),
            driving_story=(
                "Before the race, a route is prepared for this specific track.",
                "During the race, the driver looks a little way ahead instead of "
                "only reacting to what is under the car.",
                "It aims for a sensible corner entry, inside point, and exit, then "
                "adjusts if the sensors show danger.",
            ),
            decision_steps=(
                "Find the planned point for this part of the lap.",
                "Look ahead to see whether a corner is coming.",
                "Compare the car's position with the planned route.",
                "Choose steering and a safe target speed.",
                "Use live sensors to soften anything unsafe.",
            ),
            learning_story=(
                "This driver does not train a neural network.",
                "Its behaviour comes from a prepared route plus hand-written control rules.",
                "That makes its decisions repeatable and easier to trace, but it "
                "needs a route file for each track.",
            ),
            input_signals=(
                "Where the car is around the lap.",
                "The prepared route and upcoming corner shape.",
                "Speed, direction, sideways movement, and road position.",
                "Road sensors that warn when the planned move is unsafe.",
            ),
            key_takeaways=(
                "It follows a plan; it does not learn while racing.",
                "The track guide suggests a route, while sensors provide a safety check.",
            ),
            strengths=(
                "Fast and predictable on tracks with a good route file.",
                "Plans for corners before reaching them.",
                "Its route can be displayed and explained directly.",
            ),
            failure_signs=(
                "No matching route exists for the selected track.",
                "The car leaves the planned route and corrects too late.",
                "A target speed is too ambitious for a sharp corner.",
            ),
        )

    if agent_type == "rule_based":
        return NoviceAgentGuide(
            badge="Drives with a clear rulebook",
            headline=(
                "This driver is like a careful checklist: look at the road, classify "
                "the situation, then apply the matching steering and speed rules."
            ),
            driving_story=(
                "It reads the road sensors on every simulator update.",
                "Straight road, gentle bends, sharp corners, wheel spin, and recovery "
                "each trigger different rules.",
                "There is no hidden training process; every action comes from code "
                "written by the developer.",
            ),
            decision_steps=(
                "Read the road and the car's movement.",
                "Decide whether the road is straight or turning.",
                "Choose a target speed for that situation.",
                "Steer toward the clearest safe space.",
                "Reduce power or recover if the car becomes unstable.",
            ),
            learning_story=(
                "This driver does not learn from previous races.",
                "Improvements happen when a developer changes its rules or thresholds.",
                "Its advantage is transparency: a person can follow the same logic step by step.",
            ),
            input_signals=(
                "Nineteen road-distance sensors.",
                "Forward and sideways speed.",
                "Car angle, road position, and wheel spin.",
                "Stuck, reversing, and off-track warning signals.",
            ),
            key_takeaways=(
                "It reacts to the road it can currently see.",
                "It is engineered rather than trained.",
            ),
            strengths=(
                "Easy to understand and debug.",
                "Reliable when its rules match the track conditions.",
                "Does not require training data or a route file.",
            ),
            failure_signs=(
                "A new situation falls between the rules.",
                "The car reacts too late because it cannot plan beyond its sensors.",
                "A threshold that works on one corner is unsuitable for another.",
            ),
        )

    if agent_type in {"dyna_q_learning", "dyna_q_finalised", "dyna_q"}:
        finalised = agent_type == "dyna_q_finalised"
        return _dyna_q_guide(finalised)

    if agent_type == "td3_scratch":
        return _td3_guide(mode="scratch")

    if agent_type == "n_step_td3":
        return _td3_guide(mode="route")

    if agent_type == "sensor_n_step_td3":
        return _td3_guide(mode="sensor")

    if agent_type == "random":
        return NoviceAgentGuide(
            badge="A deliberately unskilled comparison",
            headline=(
                "This driver makes random control choices. It gives us a simple "
                "starting point for proving that the other approaches add real value."
            ),
            driving_story=(
                "It receives the simulator state but does not try to understand it.",
                "Steering and throttle are sampled from safe, limited ranges.",
                "Poor and inconsistent driving is expected; that is the purpose of this baseline.",
            ),
            decision_steps=(
                "Receive the latest simulator update.",
                "Pick a random steering value.",
                "Pick a random throttle value.",
                "Send both values to the car.",
                "Record the result for comparison.",
            ),
            learning_story=(
                "It does not learn, remember, plan, or improve.",
                "A fixed random seed can repeat the same sequence for a fair experiment.",
            ),
            input_signals=(
                "A random-number generator.",
                "Fixed limits for steering and throttle.",
                "No useful interpretation of the road sensors.",
            ),
            key_takeaways=(
                "This is a control experiment, not a serious racing method.",
                "Useful agents should beat it clearly and repeatedly.",
            ),
            strengths=(
                "Very simple and repeatable with a fixed seed.",
                "Provides an honest low-skill comparison.",
            ),
            failure_signs=(
                "Frequent crashes and sudden steering changes are normal.",
                "It has no recovery strategy or understanding of corners.",
            ),
        )

    return NoviceAgentGuide(
        badge="Project racing agent",
        headline="This driver can be observed through its controls and saved telemetry.",
        driving_story=("Watch the live dashboard to see how it responds to the road.",),
        decision_steps=(
            "Read the simulator.",
            "Interpret the current situation.",
            "Choose an action.",
            "Send controls to the car.",
            "Record the result.",
        ),
        learning_story=("No beginner explanation has been added for this driver yet.",),
        input_signals=("Simulator telemetry and agent configuration.",),
        key_takeaways=("Use the Technical Guide for implementation details.",),
        strengths=("Its run can be reviewed and compared with other agents.",),
        failure_signs=("Watch for off-track, stuck, or crashed outcomes.",),
    )


def _dyna_q_guide(finalised: bool) -> NoviceAgentGuide:
    mode_note = (
        "This finalised version uses its saved score table and no longer explores."
        if finalised
        else (
            "This learning version sometimes tries a different action to discover "
            "whether it works better."
        )
    )
    return NoviceAgentGuide(
        badge="Learns with a score table and memory",
        headline=(
            "Imagine a notebook that scores each driving choice. Dyna-Q updates the "
            "score after a real action, then practises remembered situations again."
        ),
        driving_story=(
            "The continuous sensor readings are grouped into a manageable driving situation.",
            "The agent checks which action has the best score for that situation.",
            mode_note,
        ),
        decision_steps=(
            "Turn the sensor readings into a simple situation label.",
            "Look up the scores for the available actions.",
            "Choose the best action, or occasionally explore while learning.",
            "See the reward from the next simulator update.",
            "Update the score and practise remembered examples.",
        ),
        learning_story=(
            "A reward tells the agent whether the last choice helped or hurt.",
            "The Q-table is its scorebook: higher values mean better expected long-term results.",
            "Its memory model replays earlier situations, so one real experience "
            "can teach it more than once.",
        ),
        input_signals=(
            "Lap section and speed range.",
            "Road position and car direction.",
            "Whether the sensors show open road or danger.",
            "The reward and next situation after each action.",
        ),
        key_takeaways=(
            "It learns action scores rather than neural-network weights.",
            "Replay means practising from memory without taking another simulator step.",
        ),
        strengths=(
            "The score table can be inspected directly.",
            "Learns more from each real experience by replaying it.",
            "The finalised version behaves consistently without exploration.",
        ),
        failure_signs=(
            "Too many situation categories make useful experience too sparse.",
            "Too few categories hide important differences between corners.",
            "The learning version may make a poor choice while exploring.",
        ),
    )


def _td3_guide(mode: str) -> NoviceAgentGuide:
    if mode == "route":
        badge = "Neural driver with a route preview"
        headline = (
            "Agent 7 learns the controls for itself, but it can also see where the "
            "prepared route goes next. The route is a hint, not a copied steering command."
        )
        context = (
            "It sees live car sensors plus the shape and position of the prepared route.",
            "The route provides advance warning of corners but never supplies "
            "throttle, brake, or steering labels.",
        )
        inputs = (
            "The car's movement and recent actions.",
            "All nineteen road-distance sensors.",
            "Road position, wheel spin, and direction.",
            "The prepared route position and upcoming corner shape.",
        )
        takeaway = "It is reinforcement learning with route context, not action imitation."
    elif mode == "sensor":
        badge = "Neural driver using sensors only"
        headline = (
            "Agent 8 learns without a prepared route or teacher. It must discover how "
            "to steer, accelerate, brake, and recover using sensors and reward."
        )
        context = (
            "It sees the car's recent movement and the road sensors, but no ideal route.",
            "Successful and failed driving both help its two training-time judges "
            "learn which choices are dependable.",
        )
        inputs = (
            "The car's movement and recent actions.",
            "All nineteen road-distance sensors.",
            "Road position, wheel spin, and direction.",
            "No prepared route, target speed, or teacher action.",
        )
        takeaway = "It discovers its own route from reward and sensor feedback."
    else:
        badge = "Neural driver learning from reward"
        headline = (
            "Agent 6 starts without a teacher. Its neural network tries continuous "
            "controls, receives rewards, and gradually changes the choices it prefers."
        )
        context = (
            "It sees driving telemetry, road sensors, and a small amount of recent history.",
            "Training examples are stored in replay memory so they can be studied again later.",
        )
        inputs = (
            "Speed, direction, road position, and recent control.",
            "Road-distance sensors and wheel spin.",
            "Damage, lap progress, and reward.",
            "No teacher action or prepared racing route.",
        )
        takeaway = "It learns continuous controls from reward, with no copied driver."

    return NoviceAgentGuide(
        badge=badge,
        headline=headline,
        driving_story=(
            *context,
            "When the race is running, the trained neural network turns those inputs "
            "directly into steering and pedal intent.",
        ),
        decision_steps=(
            "Read the car, road sensors, and recent movement.",
            "Scale the readings into numbers the network can handle.",
            "Pass those numbers through the neural driver.",
            "Turn its two outputs into steering, throttle, and brake.",
            "Select the gear automatically and send the controls to TORCS.",
        ),
        learning_story=(
            "The actor is the neural driver that chooses the controls.",
            "Two critics act like independent coaches: both estimate how useful a "
            "choice will be over time.",
            "Training adjusts the critics first, then makes smaller, less frequent "
            "changes to the actor.",
            "Evaluation removes exploration noise, so the saved driver uses its "
            "learned choice directly.",
        ),
        input_signals=inputs,
        key_takeaways=(
            takeaway,
            "Training uses trial, reward, and replay; running the saved model does "
            "not keep changing it.",
        ),
        strengths=(
            "Can learn smooth combinations of steering and pedal control.",
            "Replay lets it learn from both successful and unsuccessful moments.",
            "The saved neural network can be evaluated without exploration noise.",
        ),
        failure_signs=(
            "A single very fast lap may be luck rather than a reliable policy.",
            "Repeated swaying can mean the network is over-correcting.",
            "Repeated failure at the same corner means it has not learned a robust response there.",
        ),
    )
