from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont
from PySide6.QtWidgets import QApplication


@dataclass(frozen=True)
class ThemePalette:
    canvas: str
    surface: str
    surface_alt: str
    text: str
    muted: str
    border: str
    border_strong: str
    accent: str
    accent_hover: str
    positive: str
    warning: str
    danger: str
    selection: str
    plot_grid: str
    on_accent: str
    is_dark: bool = False

    def plot_background(self) -> QColor:
        return QColor(self.surface)


@dataclass(frozen=True)
class ChartPalette:
    speed: str
    steering: str
    throttle: str
    brake: str
    comparison_a: str
    comparison_b: str
    agent6: str
    agent7: str
    agent8: str
    neutral: str
    target: str
    complete: str
    failure: str
    marker_outline: str


LIGHT_THEME = ThemePalette(
    canvas="#F3F5F7",
    surface="#FFFFFF",
    surface_alt="#F8FAFB",
    text="#182026",
    muted="#5D6973",
    border="#DCE2E6",
    border_strong="#B8C2C9",
    accent="#087F73",
    accent_hover="#066A61",
    positive="#18864B",
    warning="#A86008",
    danger="#B42318",
    selection="#CDEDE8",
    plot_grid="#CBD4D9",
    on_accent="#FFFFFF",
)


DARK_THEME = ThemePalette(
    canvas="#15191C",
    surface="#20262A",
    surface_alt="#252C31",
    text="#EFF3F5",
    muted="#B2BDC4",
    border="#394249",
    border_strong="#536069",
    accent="#22A699",
    accent_hover="#188D82",
    positive="#55B978",
    warning="#E5A13A",
    danger="#FF7B72",
    selection="#285E59",
    plot_grid="#4B565E",
    on_accent="#061B18",
    is_dark=True,
)


ACCESSIBLE_LIGHT_THEME = ThemePalette(
    canvas="#F3F5F7",
    surface="#FFFFFF",
    surface_alt="#F8FAFB",
    text="#111827",
    muted="#4B5563",
    border="#D4D9DE",
    border_strong="#7B8792",
    accent="#006FA6",
    accent_hover="#005A88",
    positive="#007A5E",
    warning="#8A6100",
    danger="#B24700",
    selection="#D7EEF8",
    plot_grid="#B7C1CA",
    on_accent="#FFFFFF",
)


ACCESSIBLE_DARK_THEME = ThemePalette(
    canvas="#15191C",
    surface="#20262A",
    surface_alt="#252C31",
    text="#F5F7F8",
    muted="#C0C8CD",
    border="#45515A",
    border_strong="#78858E",
    accent="#56B4E9",
    accent_hover="#79C6ED",
    positive="#62D6B2",
    warning="#F0C44F",
    danger="#F28E62",
    selection="#203F50",
    plot_grid="#59656D",
    on_accent="#07161E",
    is_dark=True,
)


HIGH_CONTRAST_LIGHT_THEME = ThemePalette(
    canvas="#FFFFFF",
    surface="#FFFFFF",
    surface_alt="#F4F4F4",
    text="#000000",
    muted="#303030",
    border="#767676",
    border_strong="#303030",
    accent="#0037A6",
    accent_hover="#002B82",
    positive="#005A30",
    warning="#6E4B00",
    danger="#8B0000",
    selection="#DCE8FF",
    plot_grid="#8A8A8A",
    on_accent="#FFFFFF",
)


HIGH_CONTRAST_DARK_THEME = ThemePalette(
    canvas="#000000",
    surface="#080808",
    surface_alt="#141414",
    text="#FFFFFF",
    muted="#E0E0E0",
    border="#A8A8A8",
    border_strong="#FFFFFF",
    accent="#5CE1E6",
    accent_hover="#8CECEF",
    positive="#72E6A5",
    warning="#FFD166",
    danger="#FF8C82",
    selection="#143D40",
    plot_grid="#8A8A8A",
    on_accent="#000000",
    is_dark=True,
)


def palette_for_preferences(
    appearance_mode: str,
    colour_mode: str,
    application: QApplication | None = None,
) -> ThemePalette:
    dark = appearance_mode == "dark"
    if appearance_mode == "system":
        app = application or QApplication.instance()
        try:
            dark = bool(
                app is not None
                and app.styleHints().colorScheme() == Qt.ColorScheme.Dark
            )
        except (AttributeError, RuntimeError):
            dark = False

    if colour_mode == "high_contrast":
        return HIGH_CONTRAST_DARK_THEME if dark else HIGH_CONTRAST_LIGHT_THEME
    if colour_mode == "accessible":
        return ACCESSIBLE_DARK_THEME if dark else ACCESSIBLE_LIGHT_THEME
    return DARK_THEME if dark else LIGHT_THEME


def chart_palette_for_preferences(
    colour_mode: str,
    *,
    dark: bool,
) -> ChartPalette:
    if colour_mode == "high_contrast":
        return ChartPalette(
            speed="#00D5FF" if dark else "#0037A6",
            steering="#FFD166" if dark else "#6E4B00",
            throttle="#72E6A5" if dark else "#005A30",
            brake="#FF8C82" if dark else "#8B0000",
            comparison_a="#00D5FF" if dark else "#0037A6",
            comparison_b="#FFD166" if dark else "#6E4B00",
            agent6="#FFD166" if dark else "#6E4B00",
            agent7="#00D5FF" if dark else "#0037A6",
            agent8="#72E6A5" if dark else "#005A30",
            neutral="#FFFFFF" if dark else "#202020",
            target="#FFD166" if dark else "#6E4B00",
            complete="#72E6A5" if dark else "#005A30",
            failure="#FF8C82" if dark else "#8B0000",
            marker_outline="#000000" if dark else "#FFFFFF",
        )
    if colour_mode == "accessible":
        return ChartPalette(
            speed="#56B4E9" if dark else "#0072B2",
            steering="#CC79A7",
            throttle="#62D6B2" if dark else "#007A5E",
            brake="#F28E62" if dark else "#B24700",
            comparison_a="#56B4E9" if dark else "#0072B2",
            comparison_b="#F0C44F" if dark else "#8A6100",
            agent6="#F0C44F" if dark else "#8A6100",
            agent7="#56B4E9" if dark else "#0072B2",
            agent8="#62D6B2" if dark else "#007A5E",
            neutral="#C0C8CD" if dark else "#4B5563",
            target="#F0C44F" if dark else "#8A6100",
            complete="#62D6B2" if dark else "#007A5E",
            failure="#F28E62" if dark else "#B24700",
            marker_outline="#20262A" if dark else "#FFFFFF",
        )
    return ChartPalette(
        speed="#5AA9F0" if dark else "#2F80ED",
        steering="#B995FF" if dark else "#8E5CF7",
        throttle="#55B978" if dark else "#27AE60",
        brake="#FF7B72" if dark else "#C0392B",
        comparison_a="#5AA9F0" if dark else "#2F80ED",
        comparison_b="#E5A13A" if dark else "#D97706",
        agent6="#E5A13A" if dark else "#A86008",
        agent7="#5AA9F0" if dark else "#2F80ED",
        agent8="#22A699" if dark else "#087F73",
        neutral="#B2BDC4" if dark else "#6B7280",
        target="#E5A13A" if dark else "#A86008",
        complete="#55B978" if dark else "#18864B",
        failure="#FF7B72" if dark else "#B42318",
        marker_outline="#20262A" if dark else "#FFFFFF",
    )


def apply_application_theme(
    application: QApplication,
    palette: ThemePalette = LIGHT_THEME,
) -> None:
    application.setStyle("Fusion")
    application.setFont(QFont("Segoe UI", 10))
    application.setStyleSheet(build_stylesheet(palette))


def build_stylesheet(palette: ThemePalette = LIGHT_THEME) -> str:
    dark = palette.is_dark
    header = "#11171A" if dark else "#182126"
    header_hover = "#263137" if dark else "#253138"
    header_selected = "#202A2F" if dark else "#222D32"
    header_text = "#F4F7F8"
    header_muted = "#A9B4BA"
    header_divider = "#354147"
    hover = "#2B3439" if dark else "#EDF2F3"
    pressed = "#354046" if dark else "#E0E8EA"
    disabled = "#272D31" if dark else "#EEF1F3"
    disabled_text = "#748089" if dark else "#8B969E"

    return f"""
    QMainWindow, QDialog {{
        background-color: {palette.canvas};
        color: {palette.text};
    }}
    QWidget#pageCanvas, QWidget#scrollCanvas {{
        background-color: {palette.canvas};
    }}
    QWidget {{
        color: {palette.text};
        letter-spacing: 0px;
    }}
    QLabel#pageTitle, QLabel#dialogTitle {{
        color: {palette.text};
        font-size: 18px;
        font-weight: 700;
    }}
    QLabel#pageSubtitle, QLabel#mutedText, QLabel#metricLabel {{
        color: {palette.muted};
    }}
    QLabel#metricLabel {{
        font-size: 9px;
        font-weight: 600;
    }}
    QLabel#metricValue {{
        color: {palette.text};
        font-size: 20px;
        font-weight: 700;
    }}
    QLabel#factValue {{
        color: {palette.text};
        font-size: 11px;
        font-weight: 600;
    }}
    QLabel#overviewResultTitle {{
        color: {palette.text};
        font-size: 17px;
        font-weight: 700;
    }}
    QLabel#overviewAgentName {{
        color: {palette.text};
        font-weight: 600;
    }}
    QLabel#resultsInsight {{
        color: {palette.text};
        background-color: {palette.surface_alt};
        border-left: 3px solid {palette.accent};
        padding: 9px 11px;
    }}
    QLabel#resultsWarning {{
        color: {palette.warning};
    }}
    QLabel#settingsSaveStatus[saveState="saved"] {{
        color: {palette.positive};
        font-weight: 600;
    }}
    QLabel#settingsSaveStatus[saveState="error"] {{
        color: {palette.danger};
        font-weight: 600;
    }}
    QLabel#sourceLink {{
        color: {palette.accent};
        font-weight: 600;
    }}
    QLabel#statusLabel {{
        color: {palette.muted};
        font-weight: 600;
    }}
    QLabel#learningModeLabel {{
        color: {palette.accent};
        background-color: {palette.selection};
        border-radius: 4px;
        padding: 5px 8px;
        font-weight: 600;
    }}
    QLabel#learningStepLabel {{
        color: {palette.text};
        font-weight: 600;
    }}
    QLabel#learningEquation {{
        color: {palette.accent};
        font-family: "Consolas";
        font-weight: 600;
    }}
    QLabel#checkpointEvidence {{
        color: {palette.muted};
        background-color: {palette.surface_alt};
        border-left: 3px solid {palette.accent};
        padding: 7px 9px;
    }}
    QFrame[metricCard="true"], QFrame[surface="true"] {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
    }}
    QFrame[pipelineStep="true"] {{
        background-color: {palette.surface_alt};
        border: 0;
        border-left: 3px solid {palette.accent};
        border-radius: 3px;
    }}
    QFrame[learningExplanation="true"] {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 5px;
    }}
    QGroupBox {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
        margin-top: 13px;
        padding: 13px 8px 8px 8px;
        font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin;
        left: 10px;
        padding: 0 5px;
        color: {palette.muted};
    }}
    QFrame#primaryNavigationHeader {{
        min-height: 56px;
        max-height: 56px;
        background-color: {header};
        border: 0;
        border-bottom: 1px solid {header_divider};
    }}
    QWidget#productLockup {{ background-color: transparent; }}
    QLabel#productName {{
        color: {header_text};
        font-size: 14px;
        font-weight: 700;
    }}
    QLabel#productContext {{
        color: {header_muted};
        font-size: 8px;
        font-weight: 600;
    }}
    QFrame#navigationDivider {{
        background-color: {header_divider};
        border: 0;
    }}
    QPushButton[navigationItem="true"] {{
        min-height: 54px;
        padding: 0 15px;
        color: {header_muted};
        background-color: transparent;
        border: 0;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        font-weight: 600;
    }}
    QPushButton[navigationItem="true"]:hover {{
        color: {header_text};
        background-color: {header_hover};
    }}
    QPushButton[navigationItem="true"]:checked {{
        color: {header_text};
        background-color: {header_selected};
        border-bottom-color: {palette.accent};
    }}
    QStackedWidget#primaryPageStack {{
        background-color: {palette.canvas};
        border: 0;
    }}
    QTabWidget#contentTabs::pane,
    QTabWidget#agentLabTabs::pane {{
        background-color: {palette.surface};
        border: 1px solid {palette.border};
        border-radius: 6px;
        top: -1px;
    }}
    QTabWidget#contentTabs QTabBar::tab,
    QTabWidget#agentLabTabs QTabBar::tab {{
        padding: 8px 14px;
        color: {palette.muted};
        background-color: transparent;
        border: 0;
        border-bottom: 2px solid transparent;
        font-weight: 600;
    }}
    QTabWidget#contentTabs QTabBar::tab:selected,
    QTabWidget#agentLabTabs QTabBar::tab:selected {{
        color: {palette.accent};
        border-bottom-color: {palette.accent};
    }}
    QPushButton {{
        min-height: 32px;
        padding: 5px 11px;
        color: {palette.text};
        background-color: {palette.surface};
        border: 1px solid {palette.border_strong};
        border-radius: 5px;
        font-weight: 600;
    }}
    QPushButton:hover {{ background-color: {hover}; }}
    QPushButton:pressed {{ background-color: {pressed}; }}
    QPushButton:disabled {{
        color: {disabled_text};
        background-color: {disabled};
        border-color: {palette.border};
    }}
    QPushButton[primary="true"] {{
        color: {palette.on_accent};
        background-color: {palette.accent};
        border-color: {palette.accent};
    }}
    QPushButton[primary="true"]:hover {{
        background-color: {palette.accent_hover};
    }}
    QPushButton[segmented="true"] {{
        min-height: 34px;
        border-radius: 0;
        border-right-width: 0;
    }}
    QPushButton[segmented="true"][segmentPosition="first"] {{
        border-top-left-radius: 5px;
        border-bottom-left-radius: 5px;
    }}
    QPushButton[segmented="true"][segmentPosition="last"] {{
        border-top-right-radius: 5px;
        border-bottom-right-radius: 5px;
        border-right-width: 1px;
    }}
    QPushButton[segmented="true"]:checked {{
        color: {palette.accent};
        background-color: {palette.selection};
        border-color: {palette.accent};
    }}
    QPushButton[danger="true"] {{ color: {palette.danger}; }}
    QComboBox, QSpinBox, QDoubleSpinBox {{
        min-height: 32px;
        padding: 0 9px;
        color: {palette.text};
        background-color: {palette.surface};
        border: 1px solid {palette.border_strong};
        border-radius: 5px;
        selection-background-color: {palette.selection};
    }}
    QComboBox:hover {{ border-color: {palette.accent}; }}
    QComboBox::drop-down {{ border: 0; width: 24px; }}
    QComboBox QAbstractItemView {{
        color: {palette.text};
        background-color: {palette.surface};
        border: 1px solid {palette.border_strong};
        border-radius: 4px;
        padding: 4px;
        outline: 0;
        selection-color: {palette.text};
        selection-background-color: {palette.selection};
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 3px 8px;
        border: 0;
        border-radius: 3px;
    }}
    QComboBox QAbstractItemView::item:hover,
    QComboBox QAbstractItemView::item:selected {{
        color: {palette.text};
        background-color: {palette.selection};
    }}
    QTextEdit, QPlainTextEdit {{
        color: {palette.text};
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: 5px;
        padding: 6px;
        selection-background-color: {palette.selection};
    }}
    QTableWidget {{
        color: {palette.text};
        background-color: {palette.surface};
        alternate-background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: 6px;
        gridline-color: {palette.border};
        selection-background-color: {palette.selection};
        selection-color: {palette.text};
    }}
    QHeaderView::section {{
        color: {palette.muted};
        background-color: {palette.surface_alt};
        border: 0;
        border-bottom: 1px solid {palette.border};
        padding: 8px;
        font-weight: 600;
    }}
    QScrollArea {{
        border: 0;
        background-color: {palette.canvas};
    }}
    QScrollArea > QWidget > QWidget {{
        background-color: {palette.canvas};
    }}
    QSplitter::handle {{
        background-color: {palette.border};
        width: 1px;
        height: 1px;
    }}
    QFrame#settingsDivider {{
        color: {palette.border};
        background-color: {palette.border};
        border: 0;
        max-height: 1px;
    }}
    QCheckBox {{
        spacing: 8px;
    }}
    QCheckBox::indicator {{
        width: 17px;
        height: 17px;
    }}
    QSlider::groove:horizontal {{
        height: 5px;
        background-color: {palette.border};
        border-radius: 2px;
    }}
    QSlider::handle:horizontal {{
        width: 15px;
        margin: -5px 0;
        background-color: {palette.accent};
        border-radius: 7px;
    }}
    QProgressBar {{
        color: {palette.text};
        background-color: {palette.surface_alt};
        border: 1px solid {palette.border};
        border-radius: 4px;
        text-align: center;
    }}
    QProgressBar::chunk {{
        background-color: {palette.accent};
        border-radius: 3px;
    }}
    QStatusBar {{
        color: {palette.muted};
        background-color: {palette.surface};
        border-top: 1px solid {palette.border};
    }}
    QToolTip {{
        color: {palette.text};
        background-color: {palette.surface};
        border: 1px solid {palette.border_strong};
        padding: 5px;
    }}
    """
