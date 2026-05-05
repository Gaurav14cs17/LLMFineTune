"""
Generate Tab - Run text generation using torchtune's generate recipe.

Supports interactive text generation with fine-tuned LLMs.
"""

import os
import sys
import subprocess

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QLineEdit, QPushButton, QComboBox, QSpinBox, QDoubleSpinBox,
    QCheckBox, QProgressBar, QFileDialog, QMessageBox,
    QGridLayout, QFrame, QScrollArea, QPlainTextEdit, QSplitter,
    QTextEdit,
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal

from ui.styles import (
    BTN_PRIMARY_LARGE, BTN_SECONDARY, BTN_SUCCESS,
    COMBO_STYLE, EDIT_STYLE, LABEL_HEADING, LABEL_SECONDARY,
    SPIN_STYLE, CHECK_STYLE,
)
from ui.helpers import get_project_root, list_recipes, list_configs_for_recipe


class GenerateWorker(QThread):
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
                self.process.wait(timeout=30)
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


class InferenceTab(QWidget):
    """Text generation using torchtune's generate recipe."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.project_root = get_project_root()
        self._all_recipes = list_recipes()
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
        row1.addWidget(self._build_model_group(), 5)
        row1.addWidget(self._build_generation_params_group(), 3)
        cl.addLayout(row1)

        cl.addWidget(self._build_prompt_group())
        cl.addLayout(self._build_buttons())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_output_panel())
        splitter.setSizes([380, 350])
        main_layout.addWidget(splitter)

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#f3e5f5;border:1px solid #ce93d8;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)
        title = QLabel("Text Generation")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#6a1b9a;")
        lay.addWidget(title)
        desc = QLabel(
            "Generate text using a fine-tuned model. Uses torchtune's generate "
            "recipe under the hood. Select a generation config or provide a "
            "checkpoint path, enter your prompt, and click Generate.\n\n"
            "Tip: Use models you've fine-tuned in the Training or LoRA tabs.")
        desc.setStyleSheet("font-size:12px;color:#4a148c;")
        desc.setWordWrap(True)
        lay.addWidget(desc)
        return frame

    def _build_model_group(self):
        g = QGroupBox("Model Configuration")
        lay = QVBoxLayout(g)

        recipe_lay = QHBoxLayout()
        recipe_lay.addWidget(QLabel("Recipe:"))
        self.recipe_combo = QComboBox()
        self.recipe_combo.setStyleSheet(COMBO_STYLE)
        self.recipe_combo.setMinimumWidth(250)
        for r in self._all_recipes:
            if "generate" in r["name"]:
                self.recipe_combo.addItem(r["name"], r["name"])
        if self.recipe_combo.count() == 0:
            self.recipe_combo.addItem("generate", "generate")
        self.recipe_combo.currentIndexChanged.connect(self._on_recipe_changed)
        recipe_lay.addWidget(self.recipe_combo)
        recipe_lay.addStretch()
        lay.addLayout(recipe_lay)

        config_lay = QHBoxLayout()
        config_lay.addWidget(QLabel("Config:"))
        self.config_combo = QComboBox()
        self.config_combo.setStyleSheet(COMBO_STYLE)
        self.config_combo.setMinimumWidth(300)
        config_lay.addWidget(self.config_combo)
        config_lay.addStretch()
        lay.addLayout(config_lay)

        self._on_recipe_changed()

        ckpt_lay = QHBoxLayout()
        ckpt_lay.addWidget(QLabel("Checkpoint Dir:"))
        self.ckpt_dir_edit = QLineEdit()
        self.ckpt_dir_edit.setPlaceholderText("Path to model checkpoints (overrides config)")
        self.ckpt_dir_edit.setStyleSheet(EDIT_STYLE)
        ckpt_lay.addWidget(self.ckpt_dir_edit)
        browse_btn = QPushButton("Browse")
        browse_btn.setFixedWidth(80)
        browse_btn.setStyleSheet(BTN_SECONDARY)
        browse_btn.clicked.connect(self._browse_ckpt)
        ckpt_lay.addWidget(browse_btn)
        lay.addLayout(ckpt_lay)

        return g

    def _build_generation_params_group(self):
        g = QGroupBox("Generation Parameters")
        lay = QGridLayout(g)

        lay.addWidget(QLabel("Max Tokens:"), 0, 0)
        self.max_tokens_spin = QSpinBox()
        self.max_tokens_spin.setRange(1, 4096)
        self.max_tokens_spin.setValue(300)
        self.max_tokens_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.max_tokens_spin, 0, 1)

        lay.addWidget(QLabel("Temperature:"), 1, 0)
        self.temperature_spin = QDoubleSpinBox()
        self.temperature_spin.setRange(0.0, 2.0)
        self.temperature_spin.setValue(0.6)
        self.temperature_spin.setSingleStep(0.1)
        self.temperature_spin.setDecimals(2)
        self.temperature_spin.setStyleSheet(SPIN_STYLE)
        self.temperature_spin.setToolTip("Higher = more creative, Lower = more focused")
        lay.addWidget(self.temperature_spin, 1, 1)

        lay.addWidget(QLabel("Top-k:"), 2, 0)
        self.top_k_spin = QSpinBox()
        self.top_k_spin.setRange(1, 500)
        self.top_k_spin.setValue(300)
        self.top_k_spin.setStyleSheet(SPIN_STYLE)
        lay.addWidget(self.top_k_spin, 2, 1)

        return g

    def _build_prompt_group(self):
        g = QGroupBox("Prompt")
        lay = QVBoxLayout(g)

        self.prompt_edit = QTextEdit()
        self.prompt_edit.setPlaceholderText(
            "Enter your prompt here...\n\n"
            "Examples:\n"
            "- What is the capital of France?\n"
            "- Write a Python function that sorts a list.\n"
            "- Explain quantum computing in simple terms.")
        self.prompt_edit.setMinimumHeight(80)
        self.prompt_edit.setMaximumHeight(150)
        self.prompt_edit.setStyleSheet(
            "QTextEdit { background: #ffffff; border: 1px solid #c9cdd3; "
            "border-radius: 4px; padding: 8px; font-size: 13px; color: #1a1a2e; }")
        lay.addWidget(self.prompt_edit)

        return g

    def _build_buttons(self):
        lay = QHBoxLayout()

        self.generate_btn = QPushButton("GENERATE")
        self.generate_btn.setMinimumHeight(48)
        self.generate_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#6a1b9a")
            .replace("#212d40", "#4a148c")
            .replace("#14213d", "#311b92"))
        self.generate_btn.clicked.connect(self.start_generation)
        lay.addWidget(self.generate_btn)

        self.stop_btn = QPushButton("STOP")
        self.stop_btn.setMinimumHeight(48)
        self.stop_btn.setStyleSheet(
            BTN_PRIMARY_LARGE
            .replace("#394867", "#c0392b")
            .replace("#212d40", "#96281b")
            .replace("#14213d", "#7a1512"))
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_generation)
        lay.addWidget(self.stop_btn)

        clear_btn = QPushButton("Clear Output")
        clear_btn.setMinimumHeight(48)
        clear_btn.setStyleSheet(BTN_PRIMARY_LARGE)
        clear_btn.clicked.connect(lambda: self.output_edit.clear())
        lay.addWidget(clear_btn)

        return lay

    def _build_output_panel(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)

        hdr = QHBoxLayout()
        t = QLabel("Generation Output")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)

        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(LABEL_SECONDARY)
        hdr.addWidget(self.status_label)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.output_edit = QPlainTextEdit()
        self.output_edit.setReadOnly(True)
        self.output_edit.setStyleSheet("""
            QPlainTextEdit { background-color:#1a1a2e; color:#e1bee7;
                font-family:'Consolas',monospace; font-size:13px;
                border-radius:4px; padding:12px; line-height: 1.5; }
        """)
        self.output_edit.setMaximumBlockCount(2000)
        lay.addWidget(self.output_edit)
        return w

    def _on_recipe_changed(self, _index=None):
        recipe_name = self.recipe_combo.currentData()
        self.config_combo.clear()
        if recipe_name:
            configs = list_configs_for_recipe(recipe_name)
            for c in configs:
                self.config_combo.addItem(c["name"], c["file_path"])
        if self.config_combo.count() == 0:
            self.config_combo.addItem("generation", "generation")

    def _browse_ckpt(self):
        path = QFileDialog.getExistingDirectory(
            self, "Select Checkpoint Directory", self.project_root)
        if path:
            self.ckpt_dir_edit.setText(path)

    def start_generation(self):
        recipe_name = self.recipe_combo.currentData()
        config_name = self.config_combo.currentText()
        prompt = self.prompt_edit.toPlainText().strip()

        if not prompt:
            QMessageBox.warning(self, "Error", "Please enter a prompt.")
            return

        if not recipe_name:
            recipe_name = "generate"
        if not config_name:
            config_name = "generation"

        cmd = [
            sys.executable, "-m", "torchtune._cli.tune",
            "run", recipe_name,
            "--config", config_name,
        ]

        safe_prompt = prompt.replace("'", "'\\''")
        cmd.append(f"prompt.user='{safe_prompt}'")
        cmd.append(f"max_new_tokens={self.max_tokens_spin.value()}")
        cmd.append(f"temperature={self.temperature_spin.value()}")
        cmd.append(f"top_k={self.top_k_spin.value()}")

        ckpt_dir = self.ckpt_dir_edit.text().strip()
        if ckpt_dir:
            cmd.append(f"checkpointer.checkpoint_dir={ckpt_dir}")

        self.output_edit.clear()
        self.output_edit.appendPlainText(f"Prompt: {prompt}\n")
        self.output_edit.appendPlainText("-" * 40 + "\n")

        self.generate_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_label.setText("Generating...")

        self.worker = GenerateWorker(cmd, self.project_root)
        self.worker.log_signal.connect(self._on_log)
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def stop_generation(self):
        if self.worker:
            self.worker.stop()
            self.output_edit.appendPlainText("\n[Generation stopped]")

    def _on_log(self, text):
        self.output_edit.appendPlainText(text)

    def _on_finished(self, code):
        self.generate_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        if code == 0:
            self.status_label.setText("Complete")
        else:
            self.status_label.setText("Failed")
            self.output_edit.appendPlainText(f"\nGeneration failed (code: {code})")
