"""
Download Tab - Download models from HuggingFace via `tune download`.
"""

import os
import sys
import subprocess

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QCheckBox, QProgressBar,
    QMessageBox, QGridLayout, QFrame, QPlainTextEdit, QSplitter,
    QTableWidget, QTableWidgetItem, QHeaderView, QScrollArea,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ui.styles import (
    BTN_PRIMARY_LARGE, BTN_SECONDARY, BTN_SUCCESS, BTN_INFO,
    COMBO_STYLE, EDIT_STYLE, LABEL_HEADING, LABEL_SECONDARY,
    LOG_STYLE, CHECK_STYLE,
)
from ui.helpers import (
    get_project_root, list_hf_models_for_download, CHECKPOINTS_DIR,
)


class DownloadWorker(QThread):
    log_signal = pyqtSignal(str)
    finished = pyqtSignal(int)

    def __init__(self, cmd, cwd):
        super().__init__()
        self.cmd = cmd
        self.cwd = cwd
        self.process = None
        self._running = True

    def run(self):
        try:
            self.process = subprocess.Popen(
                self.cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                cwd=self.cwd,
                bufsize=1,
                env={**os.environ, 'PYTHONUNBUFFERED': '1'}
            )
            while self._running:
                line = self.process.stdout.readline()
                if not line:
                    if self.process.poll() is not None:
                        break
                    continue
                self.log_signal.emit(line.rstrip())
            if self.process:
                self.process.wait(timeout=10)
                self.finished.emit(self.process.returncode or 0)
            else:
                self.finished.emit(-1)
        except Exception as e:
            self.log_signal.emit(f"ERROR: {str(e)}")
            self.finished.emit(-1)

    def stop(self):
        self._running = False
        if self.process:
            try:
                self.process.terminate()
                self.process.wait(timeout=5)
            except (subprocess.TimeoutExpired, OSError):
                try:
                    self.process.kill()
                except OSError:
                    pass


class DownloadTab(QWidget):
    """Download models from HuggingFace using tune download."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.project_root = get_project_root()
        self._setup_ui()

    def _setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        config_w = QWidget()
        cl = QVBoxLayout(config_w)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(8)

        cl.addWidget(self._build_info_banner())

        row1 = QHBoxLayout()
        row1.addWidget(self._build_model_select_group(), 5)
        row1.addWidget(self._build_options_group(), 3)
        cl.addLayout(row1)

        cl.addWidget(self._build_custom_group())
        cl.addLayout(self._build_buttons())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_log())
        splitter.setSizes([400, 300])
        main_layout.addWidget(splitter)

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#e8f5e9;border:1px solid #a5d6a7;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)
        title = QLabel("Model Download")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#2e7d32;")
        lay.addWidget(title)
        desc = QLabel(
            "Download pre-trained models from HuggingFace Hub using torchtune. "
            "Select a model from the list or enter a custom HuggingFace repo ID. "
            "Models are saved to the checkpoints/ directory.\n\n"
            "Note: Some models (e.g. Llama) require a HuggingFace token with "
            "accepted license terms.")
        desc.setStyleSheet("font-size:12px;color:#1b5e20;")
        desc.setWordWrap(True)
        lay.addWidget(desc)
        return frame

    def _build_model_select_group(self):
        g = QGroupBox("Select Model")
        lay = QVBoxLayout(g)

        family_lay = QHBoxLayout()
        family_lay.addWidget(QLabel("Model Family:"))
        self.family_combo = QComboBox()
        self.family_combo.setStyleSheet(COMBO_STYLE)
        families = sorted(set(m["family"] for m in list_hf_models_for_download()))
        self.family_combo.addItem("All Families")
        for f in families:
            self.family_combo.addItem(f)
        self.family_combo.currentTextChanged.connect(self._filter_models)
        family_lay.addWidget(self.family_combo)
        family_lay.addStretch()
        lay.addLayout(family_lay)

        model_lay = QHBoxLayout()
        model_lay.addWidget(QLabel("Model:"))
        self.model_combo = QComboBox()
        self.model_combo.setStyleSheet(COMBO_STYLE)
        self.model_combo.setMinimumWidth(350)
        self.model_combo.currentTextChanged.connect(self._on_model_selected)
        model_lay.addWidget(self.model_combo)
        lay.addLayout(model_lay)

        self.model_info_label = QLabel("")
        self.model_info_label.setStyleSheet(LABEL_SECONDARY)
        self.model_info_label.setWordWrap(True)
        lay.addWidget(self.model_info_label)

        self._all_models = list_hf_models_for_download()
        self._filter_models("All Families")

        return g

    def _build_options_group(self):
        g = QGroupBox("Download Options")
        lay = QVBoxLayout(g)

        out_lay = QHBoxLayout()
        out_lay.addWidget(QLabel("Output Dir:"))
        self.output_edit = QLineEdit(CHECKPOINTS_DIR)
        self.output_edit.setStyleSheet(EDIT_STYLE)
        out_lay.addWidget(self.output_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet(BTN_SECONDARY)
        browse_btn.clicked.connect(self._browse_output)
        out_lay.addWidget(browse_btn)
        lay.addLayout(out_lay)

        token_lay = QHBoxLayout()
        token_lay.addWidget(QLabel("HF Token:"))
        self.token_edit = QLineEdit()
        self.token_edit.setPlaceholderText("Optional: hf_... (or set HF_TOKEN env var)")
        self.token_edit.setEchoMode(QLineEdit.Password)
        self.token_edit.setStyleSheet(EDIT_STYLE)
        token_lay.addWidget(self.token_edit)
        lay.addLayout(token_lay)

        self.ignore_patterns_check = QCheckBox("Skip original safetensors (save disk space)")
        self.ignore_patterns_check.setChecked(True)
        self.ignore_patterns_check.setStyleSheet(CHECK_STYLE)
        self.ignore_patterns_check.setToolTip(
            "Adds --ignore-patterns \"original/consolidated*\" to skip "
            "large original format weights")
        lay.addWidget(self.ignore_patterns_check)

        return g

    def _build_custom_group(self):
        g = QGroupBox("Custom Repository (optional)")
        lay = QHBoxLayout(g)
        lay.addWidget(QLabel("HF Repo ID:"))
        self.custom_repo_edit = QLineEdit()
        self.custom_repo_edit.setPlaceholderText("e.g. meta-llama/Llama-3.1-8B-Instruct")
        self.custom_repo_edit.setStyleSheet(EDIT_STYLE)
        lay.addWidget(self.custom_repo_edit)
        lay.addStretch()
        return g

    def _build_buttons(self):
        lay = QHBoxLayout()
        self.download_btn = QPushButton("DOWNLOAD MODEL")
        self.download_btn.setMinimumHeight(48)
        self.download_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#2e7d32")
            .replace("#212d40", "#1b5e20")
            .replace("#14213d", "#0d3b13"))
        self.download_btn.clicked.connect(self.start_download)
        lay.addWidget(self.download_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#c0392b")
            .replace("#212d40", "#96281b")
            .replace("#14213d", "#7a1512"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_download)
        lay.addWidget(self.stop_btn)

        list_btn = QPushButton("List Available Recipes")
        list_btn.setMinimumHeight(48)
        list_btn.setStyleSheet(BTN_PRIMARY_LARGE)
        list_btn.clicked.connect(self._list_recipes)
        lay.addWidget(list_btn)

        return lay

    def _build_progress(self):
        lay = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        lay.addWidget(self.progress_bar)
        self.status_label = QLabel("Ready")
        self.status_label.setMinimumWidth(200)
        lay.addWidget(self.status_label)
        return lay

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)
        hdr = QHBoxLayout()
        t = QLabel("Download Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)
        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(BTN_SECONDARY)
        cb.clicked.connect(lambda: self.log_edit.clear())
        hdr.addWidget(cb)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_edit = QPlainTextEdit()
        self.log_edit.setReadOnly(True)
        self.log_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#1a1a2e; color:#a3d977;
                font-family:'Consolas',monospace; font-size:12px;
                border-radius:4px; padding:10px; }
        """)
        self.log_edit.setMaximumBlockCount(500)
        lay.addWidget(self.log_edit)
        return w

    def _filter_models(self, family):
        self.model_combo.clear()
        for m in self._all_models:
            if family == "All Families" or m["family"] == family:
                self.model_combo.addItem(
                    f"{m['repo']}  ({m['size']})", m["repo"])

    def _on_model_selected(self, text):
        if not text:
            return
        repo = self.model_combo.currentData()
        if repo:
            self.model_info_label.setText(f"Repository: {repo}")

    def _browse_output(self):
        from PyQt5.QtWidgets import QFileDialog
        path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _list_recipes(self):
        self.log_edit.clear()
        self.log_edit.appendPlainText("Listing available recipes and configs...\n")
        cmd = [sys.executable, "-m", "torchtune._cli.tune", "ls"]
        try:
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                cwd=self.project_root, timeout=15)
            self.log_edit.appendPlainText(result.stdout)
            if result.stderr:
                self.log_edit.appendPlainText(result.stderr)
        except Exception as e:
            self.log_edit.appendPlainText(f"Error: {e}")

    def start_download(self):
        custom = self.custom_repo_edit.text().strip()
        if custom:
            repo = custom
        else:
            repo = self.model_combo.currentData()

        if not repo:
            QMessageBox.warning(self, "Error",
                "Please select a model or enter a custom HF repo ID.")
            return

        output_dir = self.output_edit.text().strip()
        if not output_dir:
            output_dir = CHECKPOINTS_DIR
        os.makedirs(output_dir, exist_ok=True)

        cmd = [
            sys.executable, "-m", "torchtune._cli.tune",
            "download", repo,
            "--output-dir", os.path.join(output_dir, repo.split("/")[-1]),
        ]

        token = self.token_edit.text().strip()
        if token:
            cmd.extend(["--hf-token", token])

        if self.ignore_patterns_check.isChecked():
            cmd.extend(["--ignore-patterns", "original/consolidated*"])

        self.log_edit.clear()
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"  Downloading: {repo}")
        self.log_edit.appendPlainText(f"  Output: {output_dir}")
        self.log_edit.appendPlainText("=" * 60)
        self.log_edit.appendPlainText(f"Command: {' '.join(cmd)}")
        self.log_edit.appendPlainText("=" * 60 + "\n")

        self.download_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Downloading...")
        self.progress_bar.setRange(0, 0)

        self.worker = DownloadWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def stop_download(self):
        if self.worker:
            self.worker.stop()
            self.log_edit.appendPlainText("\nDownload stopped by user")

    def _on_log(self, text):
        self.log_edit.appendPlainText(text)

    def _on_finished(self, code):
        self.download_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.progress_bar.setRange(0, 100)
        if code == 0:
            self.progress_bar.setValue(100)
            self.status_label.setText("Download complete!")
            self.log_edit.appendPlainText("\nDownload completed successfully!")
            QMessageBox.information(self, "Success",
                "Model downloaded successfully!\n\n"
                "You can now use it in the Fine-tuning tabs.")
        else:
            self.progress_bar.setValue(0)
            self.status_label.setText("Download failed")
            self.log_edit.appendPlainText(f"\nDownload failed (code: {code})")
