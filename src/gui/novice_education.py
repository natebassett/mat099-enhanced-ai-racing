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


def build_novice_agent_guide(
    agent_type: str,
    language: str = "en",
) -> NoviceAgentGuide:
    if language == "cy":
        return _build_welsh_novice_agent_guide(agent_type)

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


def _build_welsh_novice_agent_guide(agent_type: str) -> NoviceAgentGuide:
    if agent_type == "map_aware":
        return NoviceAgentGuide(
            badge="Yn cynllunio ymlaen gyda chanllaw trac",
            headline=(
                "Mae gan y gyrrwr hwn nodiadau ar gyfer pob cornel. Mae'n dilyn "
                "llwybr sydd wedi'i baratoi ac yn defnyddio synwyryddion byw i "
                "gadw'r car yn ddiogel."
            ),
            driving_story=(
                "Cyn y ras, caiff llwybr ei baratoi ar gyfer y trac penodol.",
                "Yn ystod y ras, mae'r gyrrwr yn edrych ymlaen at y gornel nesaf.",
                "Mae'n anelu at fynedfa, apex ac allanfa resymol, ac yn addasu os "
                "yw'r synwyryddion yn dangos perygl.",
            ),
            decision_steps=(
                "Dod o hyd i'r pwynt cynlluniedig ar gyfer y rhan hon o'r lap.",
                "Edrych ymlaen i weld a oes cornel yn dod.",
                "Cymharu safle'r car a'r llwybr cynlluniedig.",
                "Dewis llywio a chyflymder targed diogel.",
                "Defnyddio'r synwyryddion i leddfu unrhyw benderfyniad anniogel.",
            ),
            learning_story=(
                "Nid yw'r gyrrwr hwn yn hyfforddi rhwydwaith niwral.",
                "Daw ei ymddygiad o lwybr parod a rheolau rheoli a ysgrifennwyd "
                "gan ddatblygwr.",
                "Mae hynny'n ei wneud yn gyson, ond mae angen ffeil llwybr ar "
                "gyfer pob trac.",
            ),
            input_signals=(
                "Safle'r car o amgylch y lap.",
                "Y llwybr parod a siâp y corneli sydd o'i flaen.",
                "Cyflymder, cyfeiriad, symudiad i'r ochr a safle ar y ffordd.",
                "Synwyryddion ffordd sy'n rhybuddio am berygl.",
            ),
            key_takeaways=(
                "Mae'n dilyn cynllun; nid yw'n dysgu wrth rasio.",
                "Mae'r llwybr yn awgrymu cyfeiriad ac mae'r synwyryddion yn "
                "gwirio diogelwch.",
            ),
            strengths=(
                "Cyflym a rhagweladwy pan fo'r ffeil llwybr yn dda.",
                "Mae'n paratoi ar gyfer corneli cyn eu cyrraedd.",
                "Gellir dangos ac esbonio ei lwybr yn uniongyrchol.",
            ),
            failure_signs=(
                "Nid oes llwybr cyfatebol ar gyfer y trac.",
                "Mae'r car yn gadael y llwybr ac yn cywiro'n rhy hwyr.",
                "Mae'r cyflymder targed yn rhy uchel ar gyfer cornel lem.",
            ),
        )

    if agent_type == "rule_based":
        return NoviceAgentGuide(
            badge="Yn gyrru gyda llyfr rheolau clir",
            headline=(
                "Mae'r gyrrwr hwn fel rhestr wirio: darllen y ffordd, adnabod y "
                "sefyllfa, ac yna defnyddio'r rheol llywio a chyflymder briodol."
            ),
            driving_story=(
                "Mae'n darllen y synwyryddion ffordd ar bob diweddariad.",
                "Mae ffordd syth, tro ysgafn, cornel lem a llithro olwynion yn "
                "sbarduno rheolau gwahanol.",
                "Nid oes proses hyfforddi gudd; daw pob gweithred o'r cod.",
            ),
            decision_steps=(
                "Darllen y ffordd a symudiad y car.",
                "Penderfynu a yw'r ffordd yn syth neu'n troi.",
                "Dewis cyflymder targed ar gyfer y sefyllfa.",
                "Llywio tuag at y gofod diogel cliriaf.",
                "Lleihau pwer neu adfer os yw'r car yn ansefydlog.",
            ),
            learning_story=(
                "Nid yw'r gyrrwr hwn yn dysgu o rasys blaenorol.",
                "Mae'n gwella pan fydd datblygwr yn newid y rheolau neu'r trothwyon.",
                "Mae pob penderfyniad yn weladwy ac yn hawdd i'w olrhain.",
            ),
            input_signals=(
                "Pedwar ar bymtheg o synwyryddion pellter ffordd.",
                "Cyflymder ymlaen ac i'r ochr.",
                "Ongl y car, safle ar y ffordd a llithro olwynion.",
                "Rhybuddion pan fo'r car yn sownd, yn bacio neu oddi ar y trac.",
            ),
            key_takeaways=(
                "Mae'n ymateb i'r ffordd y gall ei gweld ar hyn o bryd.",
                "Mae wedi'i beiriannu yn hytrach na'i hyfforddi.",
            ),
            strengths=(
                "Hawdd ei ddeall a'i brofi.",
                "Dibynadwy pan fo'r rheolau'n cyfateb i'r trac.",
                "Nid oes angen data hyfforddi na ffeil llwybr.",
            ),
            failure_signs=(
                "Nid yw sefyllfa newydd yn cyd-fynd ag unrhyw reol.",
                "Mae'r car yn ymateb yn rhy hwyr oherwydd na all gynllunio ymhell ymlaen.",
                "Nid yw trothwy sy'n gweithio ar un gornel yn addas ar gyfer un arall.",
            ),
        )

    if agent_type in {"dyna_q_learning", "dyna_q_finalised", "dyna_q"}:
        return _welsh_dyna_q_guide(agent_type == "dyna_q_finalised")

    if agent_type == "n_step_td3":
        return _welsh_td3_guide(mode="route")

    if agent_type == "sensor_n_step_td3":
        return _welsh_td3_guide(mode="sensor")

    if agent_type == "td3_scratch":
        return _welsh_td3_guide(mode="scratch")

    if agent_type == "random":
        return NoviceAgentGuide(
            badge="Cymhariaeth syml heb sgiliau",
            headline=(
                "Mae'r gyrrwr hwn yn dewis rheolyddion ar hap. Mae'n fan cychwyn "
                "syml sy'n dangos a yw'r dulliau eraill yn ychwanegu gwerth go iawn."
            ),
            driving_story=(
                "Mae'n derbyn cyflwr y car ond nid yw'n ceisio ei ddeall.",
                "Dewisir llywio a chyflymu ar hap o fewn terfynau bach.",
                "Disgwylir gyrru gwael ac anghyson; dyna bwrpas y llinell sylfaen.",
            ),
            decision_steps=(
                "Derbyn y diweddariad diweddaraf gan TORCS.",
                "Dewis gwerth llywio ar hap.",
                "Dewis gwerth cyflymu ar hap.",
                "Anfon y ddau werth i'r car.",
                "Cofnodi'r canlyniad i'w gymharu.",
            ),
            learning_story=(
                "Nid yw'n dysgu, cofio, cynllunio na gwella.",
                "Gall hedyn ar hap sefydlog ailadrodd yr un dilyniant ar gyfer prawf teg.",
            ),
            input_signals=(
                "Cynhyrchydd rhifau ar hap.",
                "Terfynau sefydlog ar gyfer llywio a chyflymu.",
                "Dim dehongliad defnyddiol o'r synwyryddion ffordd.",
            ),
            key_takeaways=(
                "Arbrawf rheoli yw hwn, nid dull rasio go iawn.",
                "Dylai asiant defnyddiol ei guro'n glir ac yn gyson.",
            ),
            strengths=(
                "Syml iawn ac ailadroddadwy gyda hedyn sefydlog.",
                "Mae'n darparu cymhariaeth onest ar lefel isel.",
            ),
            failure_signs=(
                "Mae damweiniau a newidiadau llywio sydyn yn normal.",
                "Nid oes ganddo strategaeth adfer na dealltwriaeth o gorneli.",
            ),
        )

    return NoviceAgentGuide(
        badge="Asiant rasio'r prosiect",
        headline="Gellir arsylwi'r gyrrwr hwn drwy ei reolyddion a'i delemetreg.",
        driving_story=("Gwyliwch y dangosfwrdd i weld sut mae'n ymateb i'r ffordd.",),
        decision_steps=(
            "Darllen y efelychydd.",
            "Dehongli'r sefyllfa bresennol.",
            "Dewis gweithred.",
            "Anfon rheolyddion i'r car.",
            "Cofnodi'r canlyniad.",
        ),
        learning_story=("Nid oes esboniad Cymraeg penodol ar gyfer yr asiant hwn eto.",),
        input_signals=("Telemetreg yr efelychydd a gosodiadau'r asiant.",),
        key_takeaways=("Defnyddiwch y canllaw technegol am fwy o fanylion.",),
        strengths=("Gellir adolygu a chymharu ei rasys.",),
        failure_signs=("Gwyliwch am ganlyniadau oddi ar y trac neu'n sownd.",),
    )


def _welsh_dyna_q_guide(finalised: bool) -> NoviceAgentGuide:
    mode_note = (
        "Mae'r fersiwn derfynol yn defnyddio ei dabl sgoriau wedi'i gadw ac "
        "nid yw'n archwilio mwyach."
        if finalised
        else (
            "Mae'r fersiwn sy'n dysgu weithiau'n rhoi cynnig ar weithred wahanol "
            "i weld a yw'n gweithio'n well."
        )
    )
    return NoviceAgentGuide(
        badge="Yn dysgu gyda thabl sgoriau a chof",
        headline=(
            "Dychmygwch lyfr nodiadau sy'n rhoi sgor i bob dewis gyrru. Mae "
            "Dyna-Q yn diweddaru'r sgor ar ôl gweithred go iawn ac yna'n ymarfer "
            "sefyllfaoedd o'i gof."
        ),
        driving_story=(
            "Caiff darlleniadau parhaus eu grwpio'n sefyllfa y gellir ei rheoli.",
            "Mae'r asiant yn gwirio pa weithred sydd â'r sgor gorau.",
            mode_note,
        ),
        decision_steps=(
            "Troi darlleniadau'r synwyryddion yn label sefyllfa syml.",
            "Edrych ar sgoriau'r gweithredoedd sydd ar gael.",
            "Dewis y weithred orau, neu archwilio weithiau wrth ddysgu.",
            "Derbyn gwobr o'r diweddariad nesaf.",
            "Diweddaru'r sgor ac ymarfer enghreifftiau o'r cof.",
        ),
        learning_story=(
            "Mae gwobr yn dweud a wnaeth y dewis diwethaf helpu neu niweidio.",
            "Y tabl Q yw'r llyfr sgoriau: mae gwerth uwch yn awgrymu canlyniad "
            "hirdymor gwell.",
            "Mae ailchwarae'r cof yn gadael i un profiad ddysgu'r asiant fwy nag unwaith.",
        ),
        input_signals=(
            "Rhan o'r lap ac ystod cyflymder.",
            "Safle ar y ffordd a chyfeiriad y car.",
            "A yw'r synwyryddion yn dangos ffordd agored neu berygl.",
            "Y wobr a'r sefyllfa nesaf ar ôl pob gweithred.",
        ),
        key_takeaways=(
            "Mae'n dysgu sgoriau gweithredoedd, nid pwysau rhwydwaith niwral.",
            "Mae ailchwarae yn golygu ymarfer o'r cof heb gam efelychu arall.",
        ),
        strengths=(
            "Gellir archwilio'r tabl sgoriau'n uniongyrchol.",
            "Mae'n dysgu mwy o bob profiad drwy ei ailchwarae.",
            "Mae'r fersiwn derfynol yn gyson heb archwilio.",
        ),
        failure_signs=(
            "Mae gormod o gategorïau'n gwasgaru'r profiad yn rhy denau.",
            "Mae rhy ychydig o gategorïau'n cuddio gwahaniaethau pwysig.",
            "Gall y fersiwn sy'n dysgu wneud dewis gwael wrth archwilio.",
        ),
    )


def _welsh_td3_guide(mode: str) -> NoviceAgentGuide:
    if mode == "route":
        badge = "Gyrrwr niwral gyda rhagolwg o'r llwybr"
        headline = (
            "Mae Agent 7 yn dysgu'r rheolyddion ei hun, ond gall hefyd weld sut "
            "mae'r llwybr parod yn troi o'i flaen. Awgrym yw'r llwybr, nid "
            "gorchymyn llywio i'w gopïo."
        )
        context = (
            "Mae'n gweld synwyryddion byw ynghyd â siâp a safle'r llwybr parod.",
            "Mae'r llwybr yn rhybuddio am gorneli ond nid yw'n darparu gweithredoedd.",
        )
        inputs = (
            "Symudiad y car a'r gweithredoedd diweddar.",
            "Pob un o'r 19 synhwyrydd pellter ffordd.",
            "Safle ar y ffordd, llithro olwynion a chyfeiriad.",
            "Safle'r llwybr parod a siâp y gornel sydd o'i flaen.",
        )
        takeaway = "Dysgu atgyfnerthu gyda chyd-destun llwybr yw hwn, nid dynwared gweithredoedd."
    elif mode == "sensor":
        badge = "Gyrrwr niwral sy'n defnyddio synwyryddion yn unig"
        headline = (
            "Mae Agent 8 yn dysgu heb lwybr parod nac athro. Rhaid iddo "
            "ddarganfod sut i lywio, cyflymu, brecio ac adfer drwy wobr a synwyryddion."
        )
        context = (
            "Mae'n gweld symudiad diweddar y car a'r ffordd drwy'r synwyryddion.",
            "Mae gyrru llwyddiannus a gyrru aflwyddiannus yn helpu'r ddau feirniad "
            "hyfforddi i asesu pa ddewisiadau sy'n ddibynadwy.",
        )
        inputs = (
            "Symudiad y car a'r gweithredoedd diweddar.",
            "Pob un o'r 19 synhwyrydd pellter ffordd.",
            "Safle ar y ffordd, llithro olwynion a chyfeiriad.",
            "Dim llwybr parod, cyflymder targed na gweithred gan athro.",
        )
        takeaway = "Mae'n darganfod ei lwybr ei hun drwy wobr ac adborth synwyryddion."
    else:
        badge = "Gyrrwr niwral sy'n dysgu o wobr"
        headline = (
            "Mae Agent 6 yn dechrau heb athro. Mae'r rhwydwaith niwral yn rhoi "
            "cynnig ar reolyddion parhaus ac yn newid ei ddewisiadau drwy wobr."
        )
        context = (
            "Mae'n gweld telemetreg, synwyryddion ffordd ac ychydig o hanes diweddar.",
            "Cedwir profiadau yn y cof ailchwarae er mwyn eu hastudio eto.",
        )
        inputs = (
            "Cyflymder, cyfeiriad, safle ar y ffordd a rheolyddion diweddar.",
            "Synwyryddion pellter ffordd a llithro olwynion.",
            "Difrod, cynnydd o amgylch y lap a gwobr.",
            "Dim gweithred gan athro na llwybr rasio parod.",
        )
        takeaway = "Mae'n dysgu rheolyddion parhaus o wobr heb gopïo gyrrwr arall."

    return NoviceAgentGuide(
        badge=badge,
        headline=headline,
        driving_story=(
            *context,
            "Pan fydd y ras yn rhedeg, mae'r rhwydwaith niwral yn troi'r "
            "mewnbynnau'n llywio ac yn fwriad pedal yn uniongyrchol.",
        ),
        decision_steps=(
            "Darllen y car, y synwyryddion ffordd a'r symudiad diweddar.",
            "Graddio'r darlleniadau i rifau y gall y rhwydwaith eu defnyddio.",
            "Pasio'r rhifau drwy'r gyrrwr niwral.",
            "Troi'r ddau allbwn yn llywio, cyflymu a brecio.",
            "Dewis y gêr yn awtomatig ac anfon y rheolyddion i TORCS.",
        ),
        learning_story=(
            "Yr actor yw'r gyrrwr niwral sy'n dewis y rheolyddion.",
            "Mae dau feirniad annibynnol yn amcangyfrif pa mor ddefnyddiol fydd "
            "dewis dros amser.",
            "Mae hyfforddi'n addasu'r beirniaid yn gyntaf ac yna'n gwneud "
            "newidiadau llai ac yn llai aml i'r actor.",
            "Wrth werthuso, caiff swn archwilio ei ddiffodd fel bod y model a "
            "gadwyd yn defnyddio ei ddewis dysgedig.",
        ),
        input_signals=inputs,
        key_takeaways=(
            takeaway,
            "Mae hyfforddi'n defnyddio profion, gwobrau a chof ailchwarae; nid "
            "yw rhedeg y model wedi'i gadw yn parhau i'w newid.",
        ),
        strengths=(
            "Gall ddysgu cyfuniadau llyfn o lywio a rheoli'r pedalau.",
            "Mae ailchwarae'n defnyddio eiliadau llwyddiannus ac aflwyddiannus.",
            "Gellir gwerthuso'r rhwydwaith wedi'i gadw heb swn archwilio.",
        ),
        failure_signs=(
            "Gall un lap cyflym fod yn lwc yn hytrach na pholisi dibynadwy.",
            "Gall siglo dro ar ôl tro olygu bod y rhwydwaith yn gor-gywiro.",
            "Mae methu yn yr un gornel yn golygu nad yw wedi dysgu ymateb cadarn yno.",
        ),
    )
