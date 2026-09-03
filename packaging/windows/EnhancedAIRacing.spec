from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
GYM_TORCS_ROOT = PROJECT_ROOT / "torcs-wrapper" / "gym_torcs"


def add_file(relative_path: str) -> tuple[str, str]:
    source = PROJECT_ROOT / relative_path
    if not source.is_file():
        raise FileNotFoundError(f"Required runtime file is missing: {source}")
    return str(source), str(Path(relative_path).parent)


datas = [
    (str(PROJECT_ROOT / "torcs"), "torcs"),
    (str(PROJECT_ROOT / "data" / "racing_lines"), "data/racing_lines"),
    (str(PROJECT_ROOT / "data" / "policies"), "data/policies"),
    (str(PROJECT_ROOT / "data" / "evaluation"), "data/evaluation"),
    (str(SRC_ROOT / "storage" / "migrations"), "storage/migrations"),
    add_file("data/generated/race_results.db"),
    add_file("models/agent7_n_step_td3_v3/best_evaluation.pt"),
    add_file(
        "models/agent8_sensor_n_step_td3_self_imitation_stability/"
        "champion_83_038s_36of40_clean.pt"
    ),
]

# The Results page needs episode summaries, not multi-gigabyte replay buffers.
for pattern in ("episodes.csv", "config.json"):
    for source in (PROJECT_ROOT / "models" / "training_runs").rglob(pattern):
        relative_parent = source.relative_to(PROJECT_ROOT).parent
        datas.append((str(source), str(relative_parent)))


hidden_imports = [
    "agents.dyna_q_agent",
    "agents.map_aware_agent",
    "agents.n_step_td3_agent",
    "agents.random_agent",
    "agents.rule_based_agent",
    "agents.sensor_n_step_td3_agent",
    "gym_torcs",
    "gui.main_window",
    "snakeoil3_gym",
]


a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "windows" / "windows_entry.py")],
    pathex=[str(SRC_ROOT), str(GYM_TORCS_ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(PROJECT_ROOT / "packaging" / "windows" / "runtime_hook.py")],
    excludes=["tensorflow", "torchvision"],
    noarchive=False,
    optimize=1,
)

# Qt 6 uses the unversioned ICU forwarding API supplied by Windows. A build
# launched from an augmented PATH can otherwise collect an unrelated, versioned
# ICU distribution whose exports are incompatible with Qt6Core.dll.
windows_icu_forwarders = {"icuuc.dll", "icudt78.dll"}
a.binaries = [
    entry
    for entry in a.binaries
    if Path(entry[0]).name.lower() not in windows_icu_forwarders
]

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="EnhancedAIRacing",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=str(PROJECT_ROOT / "torcs" / "Ticon.ico"),
    contents_directory="_internal",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="Enhanced AI Racing",
)
