from __future__ import annotations

from PySide6.QtWidgets import (
    QAbstractButton,
    QGroupBox,
    QLabel,
    QTabWidget,
    QTableWidget,
    QWidget,
)


WELSH = {
    "Enhanced AI Racing Telemetry Dashboard": "Dangosfwrdd Telemetreg Rasio AI Uwch",
    "Enhanced AI Racing": "Rasio AI Uwch",
    "TELEMETRY & LEARNING LAB": "LAB TELEMETREG A DYSGU",
    "Live Telemetry": "Telemetreg Fyw",
    "Runs": "Rasys",
    "Results": "Canlyniadau",
    "Review": "Adolygu",
    "Compare": "Cymharu",
    "Agent Lab": "Labordy Asiantau",
    "Pick a driver to see what it notices, how it decides, and whether it learns.": (
        "Dewiswch yrrwr i weld beth mae'n sylwi arno, sut mae'n penderfynu, ac a yw'n dysgu."
    ),
    "Driver style": "Dull gyrru",
    "How it improves": "Sut mae'n gwella",
    "Makes decisions with": "Yn gwneud penderfyniadau gyda",
    "Track guide": "Canllaw trac",
    "Step 1": "Cam 1",
    "Step 2": "Cam 2",
    "Step 3": "Cam 3",
    "Step 4": "Cam 4",
    "Step 5": "Cam 5",
    "Settings": "Gosodiadau",
    "Choose how the application opens and how much live information it shows.": (
        "Dewiswch sut mae'r rhaglen yn agor a faint o wybodaeth fyw mae'n ei dangos."
    ),
    "Preferences": "Dewisiadau",
    "Research Evidence": "Tystiolaeth Ymchwil",
    "Sources & Methods": "Ffynonellau a Dulliau",
    "Data & Storage": "Data a Storio",
    "Race Control": "Rheoli'r Ras",
    "Agent": "Asiant",
    "Track": "Trac",
    "Car": "Car",
    "Start": "Dechrau",
    "Stop": "Stopio",
    "Play": "Chwarae",
    "Reset": "Ailosod",
    "Open Map": "Agor Map",
    "Compare": "Cymharu",
    "Status": "Statws",
    "Idle": "Segur",
    "Live Driver Dashboard": "Dangosfwrdd Gyrrwr Byw",
    "What The Car Is Doing": "Beth mae'r Car yn ei Wneud",
    "Current State": "Cyflwr Presennol",
    "Why": "Pam",
    "Event Log": "Cofnod Digwyddiadau",
    "Road Position": "Safle ar y Ffordd",
    "Road Ahead": "Ffordd o'ch Blaen",
    "Speed": "Cyflymder",
    "Steering": "Llywio",
    "Throttle": "Cyflymydd",
    "Brake": "Brec",
    "Throttle / Brake": "Cyflymu / Brecio",
    "Run History": "Hanes Rasys",
    "Run": "Ras",
    "Started": "Dechreuwyd",
    "Best Lap": "Lap Orau",
    "Avg Speed": "Cyflymder Cyfartalog",
    "Off Track": "Oddi ar y Trac",
    "Result": "Canlyniad",
    "No run selected": "Dim ras wedi'i dewis",
    "Run Summary": "Crynodeb Ras",
    "Replay Controls": "Rheolyddion Ailchwarae",
    "Replay Road Position": "Safle Ffordd yr Ailchwarae",
    "Replay Road Ahead": "Ffordd o Flaen yr Ailchwarae",
    "At This Moment": "Ar y Foment Hon",
    "Replay State": "Cyflwr Ailchwarae",
    "Compare Saved Runs": "Cymharu Rasys wedi'u Cadw",
    "Selection": "Dewis",
    "Run A": "Ras A",
    "Run B": "Ras B",
    "Comparison Summary": "Crynodeb Cymhariaeth",
    "Synced Replay": "Ailchwarae wedi'i Gydamseru",
    "Speed Comparison": "Cymhariaeth Cyflymder",
    "Agent Guide": "Canllaw Asiant",
    "Open Technical Guide": "Agor y Canllaw Technegol",
    "See equations and code": "Gweld hafaliadau a chod",
    "Open the optional technical guide with equations, pseudocode, and source excerpts": (
        "Agorwch y canllaw technegol dewisol gyda hafaliadau, ffug-god a darnau o'r cod ffynhonnell"
    ),
    "Algorithm Guide": "Canllaw Algorithm",
    "Algorithm": "Algorithm",
    "Formulas": "Fformiwlâu",
    "Code": "Cod",
    "Interpretation": "Dehongliad",
    "Close": "Cau",
    "How The Algorithm Thinks": "Sut Mae'r Algorithm yn Meddwl",
    "What Users Should Notice": "Beth Dylai Defnyddwyr Sylwi Arno",
    "Readable Algorithm": "Algorithm Hawdd ei Ddarllen",
    "Maths / Logic In Human Terms": "Mathemateg / Rhesymeg mewn Iaith Syml",
    "Meet This Driver": "Dewch i Adnabod y Gyrrwr",
    "How It Drives": "Sut Mae'n Gyrru",
    "How It Learns": "Sut Mae'n Dysgu",
    "Driving": "Gyrru",
    "Learning": "Dysgu",
    "Inside the Brain": "Y Tu Mewn i'r Ymennydd",
    "Strengths & Limits": "Cryfderau a Therfynau",
    "The Simple Version": "Y Fersiwn Syml",
    "One Decision, Step by Step": "Un Penderfyniad, Gam wrth Gam",
    "How Learning Works": "Sut Mae Dysgu'n Gweithio",
    "What It Can See": "Beth Gall ei Weld",
    "In Plain English": "Mewn Iaith Syml",
    "What It Does Well": "Beth Mae'n ei Wneud yn Dda",
    "What Can Go Wrong": "Beth All Fynd o'i Le",
    "Where It Works": "Ble Mae'n Gweithio",
    "Technical Details": "Manylion Technegol",
    "Racing-Line Visualizer": "Delweddwr Llinell Rasio",
    "Accelerate": "Cyflymu",
    "Full throttle": "Cyflymu llawn",
    "Turn": "Troi",
    "Settle": "Sefydlogi",
    "Line Summary": "Crynodeb y Llinell",
    "Current Target": "Targed Presennol",
    "No racing line loaded": "Dim llinell rasio wedi'i llwytho",
    "No agent discovered": "Ni ddarganfuwyd asiant",
    "No project agent metadata is available.": (
        "Nid oes manylion asiant prosiect ar gael."
    ),
    "Overview": "Trosolwg",
    "Learning Journey": "Taith Ddysgu",
    "Agent Comparison": "Cymhariaeth Asiantau",
    "Refresh evidence": "Adnewyddu tystiolaeth",
    "Race Defaults": "Rhagosodiadau Ras",
    "Driver": "Gyrrwr",
    "Opening Screen": "Sgrin Agoriadol",
    "Appearance & Language": "Ymddangosiad ac Iaith",
    "Theme": "Thema",
    "Colour presentation": "Cyflwyniad lliw",
    "Language": "Iaith",
    "Display": "Dangos",
    "Live chart history": "Hanes siart fyw",
    "Reduce animated movement": "Lleihau symudiad animeiddiedig",
    "Helpful Notices": "Hysbysiadau Defnyddiol",
    "Show the reliability note before starting a TD3 race": (
        "Dangos y nodyn dibynadwyedd cyn dechrau ras TD3"
    ),
    (
        "The note explains the measured completion rate before a learned "
        "driver starts. It does not change how the agent drives."
    ): (
        "Mae'r nodyn yn esbonio'r gyfradd gwblhau fesuredig cyn i yrrwr "
        "dysgedig ddechrau. Nid yw'n newid sut mae'r asiant yn gyrru."
    ),
    "Save settings": "Cadw gosodiadau",
    "Restore defaults": "Adfer rhagosodiadau",
    "Temporary Cache": "Storfa Dros Dro",
    "Clear temporary cache": "Clirio'r storfa dros dro",
    "Run History": "Hanes Rasys",
    "Reset run history": "Ailosod hanes rasys",
    "Always Protected": "Bob Amser wedi'u Diogelu",
    (
        "Removes generated Python bytecode and refreshes cached checkpoint "
        "summaries. Models, race runs, and research evidence stay safe."
    ): (
        "Mae'n dileu is-god Python a gynhyrchwyd ac yn adnewyddu crynodebau "
        "pwynt gwirio dros dro. Mae modelau, rasys a thystiolaeth ymchwil yn "
        "aros yn ddiogel."
    ),
    (
        "Deletes older GUI race runs while preserving one useful replay for "
        "each agent. Completed laps are preferred so Review and Compare "
        "remain usable."
    ): (
        "Mae'n dileu rasys rhyngwyneb hŷn gan gadw un ailchwarae defnyddiol "
        "ar gyfer pob asiant. Rhoddir blaenoriaeth i lapiau cyflawn fel bod "
        "Adolygu a Chymharu yn parhau'n ddefnyddiol."
    ),
    (
        "Training checkpoints, replay buffers, evaluation JSON/CSV files, "
        "training logs, recorded laps, racing lines, and project source code "
        "are never removed by these controls."
    ): (
        "Nid yw'r rheolyddion hyn byth yn dileu pwyntiau gwirio hyfforddi, "
        "byfferau ailchwarae, ffeiliau gwerthuso JSON/CSV, logiau hyfforddi, "
        "lapiau wedi'u recordio, llinellau rasio na chod ffynhonnell y prosiect."
    ),
    "Reinforcement Learning": "Dysgu Atgyfnerthu",
    "Evaluation Practice": "Arfer Gwerthuso",
    "Learning Visualisation": "Delweddu Dysgu",
    "Accessibility and Language": "Hygyrchedd ac Iaith",
    "Project Documentation": "Dogfennaeth y Prosiect",
    "Follow Windows": "Dilyn Windows",
    "Light": "Golau",
    "Dark": "Tywyll",
    "Standard": "Safonol",
    "Colour-accessible": "Hygyrch o ran lliw",
    "High contrast": "Cyferbyniad uchel",
    "English": "Saesneg",
    "Cymraeg": "Cymraeg",
    "Settings saved": "Cadwyd y gosodiadau",
    "Preparing the interface...": "Paratoi'r rhyngwyneb...",
    "Loading application components...": "Llwytho cydrannau'r rhaglen...",
    "Finding drivers, tracks, and saved races...": (
        "Dod o hyd i yrwyr, traciau a rasys wedi'u cadw..."
    ),
    "Preparing the telemetry workspace...": "Paratoi'r gweithle telemetreg...",
    "Building the learning and results pages...": (
        "Adeiladu'r tudalennau dysgu a chanlyniadau..."
    ),
    "Connecting controls and live telemetry...": (
        "Cysylltu rheolyddion a thelemetreg fyw..."
    ),
    "Finishing the dashboard...": "Gorffen y dangosfwrdd...",
    "Ready": "Yn barod",
    "Before You Start": "Cyn Dechrau",
    "Learned drivers can vary between races.": "Gall gyrwyr dysgedig amrywio rhwng rasys.",
    "Do not show this note again": "Peidiwch â dangos y nodyn hwn eto",
    "Start race": "Dechrau'r ras",
    "Not now": "Nid nawr",
    "Temporary Cache Cleared": "Cliriwyd y Storfa Dros Dro",
    "Cache Could Not Be Cleared": "Ni Ellid Clirio'r Storfa",
    "Run History Already Clear": "Mae Hanes y Rasys Eisoes yn Glir",
    "There are no older race runs to remove.": "Nid oes rasys hŷn i'w dileu.",
    "Reset Run History?": "Ailosod Hanes y Rasys?",
    "Delete older GUI race runs?": "Dileu rasys rhyngwyneb hŷn?",
    "Delete older runs": "Dileu rasys hŷn",
    "Cancel": "Canslo",
    "Run History Could Not Be Reset": "Ni Ellid Ailosod Hanes y Rasys",
    "Run History Reset": "Ailosodwyd Hanes y Rasys",
}


def tr(text: str, language: str) -> str:
    if language == "cy":
        return WELSH.get(text, text)
    return text


def translate_widget_tree(root: QWidget, language: str) -> None:
    widgets = [root, *root.findChildren(QWidget)]
    for widget in widgets:
        if isinstance(widget, QGroupBox):
            _translate_property(widget, "title", widget.title, widget.setTitle, language)
        elif isinstance(widget, QAbstractButton):
            _translate_property(widget, "text", widget.text, widget.setText, language)
        elif isinstance(widget, QLabel):
            _translate_property(widget, "text", widget.text, widget.setText, language)

        if widget.toolTip():
            _translate_property(
                widget,
                "tool_tip",
                widget.toolTip,
                widget.setToolTip,
                language,
            )

        if isinstance(widget, QTabWidget):
            sources = getattr(widget, "_i18n_tab_sources", None)
            if sources is None:
                sources = tuple(widget.tabText(index) for index in range(widget.count()))
                if not any(source in WELSH for source in sources):
                    continue
                widget._i18n_tab_sources = sources
            for index, source in enumerate(sources):
                widget.setTabText(index, tr(source, language))

        if isinstance(widget, QTableWidget):
            sources = getattr(widget, "_i18n_header_sources", None)
            if sources is None:
                sources = tuple(
                    widget.horizontalHeaderItem(index).text()
                    if widget.horizontalHeaderItem(index) is not None
                    else ""
                    for index in range(widget.columnCount())
                )
                if not any(source in WELSH for source in sources):
                    continue
                widget._i18n_header_sources = sources
            for index, source in enumerate(sources):
                item = widget.horizontalHeaderItem(index)
                if item is not None:
                    item.setText(tr(source, language))


def _translate_property(widget, name, getter, setter, language: str) -> None:
    attribute = f"_i18n_source_{name}"
    source = getattr(widget, attribute, None)
    if source is None:
        source = getter()
        if source not in WELSH:
            return
        setattr(widget, attribute, source)
    setter(tr(source, language))
