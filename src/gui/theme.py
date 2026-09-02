from __future__ import annotations

from dataclasses import dataclass

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

    def plot_background(self) -> QColor:
        return QColor(self.surface)


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
)


def apply_application_theme(
    application: QApplication,
    palette: ThemePalette = LIGHT_THEME,
) -> None:
    application.setStyle("Fusion")
    application.setFont(QFont("Segoe UI", 10))
    application.setStyleSheet(build_stylesheet(palette))


def build_stylesheet(palette: ThemePalette = LIGHT_THEME) -> str:
    dark = palette is DARK_THEME
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
    QLabel#statusLabel {{
        color: {palette.muted};
        font-weight: 600;
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
        color: #FFFFFF;
        background-color: {palette.accent};
        border-color: {palette.accent};
    }}
    QPushButton[primary="true"]:hover {{
        background-color: {palette.accent_hover};
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
