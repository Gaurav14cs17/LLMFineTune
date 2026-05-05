"""
Full Fine-tuning Tab style dashboard — same font, color, size, spacing, QGroupBox
pattern as training_tab.py and lora_tab.py.
"""

import gc
import re
import time
from pathlib import Path

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGroupBox, QLabel,
    QComboBox, QPushButton, QTableWidget, QTableWidgetItem,
    QHeaderView, QFrame, QCheckBox, QSizePolicy, QMessageBox,
    QPlainTextEdit, QSplitter, QScrollArea, QGridLayout, QProgressBar,
)
from PyQt5.QtCore import Qt, QTimer

from ui.styles import (
    BTN_DANGER, BTN_SUCCESS, BTN_WARNING, BTN_SECONDARY,
    CHECK_STYLE, COMBO_STYLE, LABEL_HEADING, LABEL_SECONDARY,
    PROGRESS_STYLE,
)

import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

_RE_STEP_LOSS = re.compile(
    r"Step\s+(\d+)\s*\|\s*loss:\s*([0-9.eE+-]+)", re.IGNORECASE)
_RE_LR = re.compile(r"lr:\s*([0-9.eE+-]+)", re.IGNORECASE)
_RE_TPS = re.compile(
    r"tokens_per_second(?:_per_gpu)?:\s*([0-9.eE+-]+)", re.IGNORECASE)
_RE_GPU_MEM = re.compile(
    r"gpu_memory:\s*([0-9.eE+-]+)\s*(?:GB)?", re.IGNORECASE)
_RE_PIPE = re.compile(
    r"(\d+)\|(\d+)\|loss:\s*([0-9.eE+-]+)\|lr:\s*([0-9.eE+-]+)")


def _project_root():
    return Path(__file__).resolve().parent.parent.parent


def _collect_log_paths(exp_dir: Path):
    found = []
    for pat in ("*.log", "log_*.txt"):
        found.extend(exp_dir.glob(pat))
    for p in exp_dir.glob("*.txt"):
        if p not in found:
            found.append(p)
    uniq = {}
    for p in found:
        try:
            uniq[p.resolve()] = p
        except OSError:
            continue
    return sorted(uniq.values(), key=lambda p: p.stat().st_mtime, reverse=True)


class _Chart(FigureCanvas):
    def __init__(self):
        self.fig = Figure(dpi=100, facecolor='#ffffff')
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumHeight(180)

    def plot(self, x, y, color='#394867', title='', xlabel='Step'):
        a = self.ax
        a.clear()
        if x and y and len(x) == len(y) and len(x) > 0:
            kw = dict(color=color, lw=1.8, alpha=0.9)
            if len(x) < 100:
                kw.update(marker='o', ms=2.5)
            a.plot(x, y, **kw)
            a.fill_between(x, y, alpha=0.06, color=color)
        else:
            a.text(.5, .5, 'No data yet', transform=a.transAxes,
                   fontsize=11, ha='center', va='center', color='#9da3ac')
        if title:
            a.set_title(title, fontsize=10, fontweight='600',
                        color='#1a1a2e', pad=8, loc='left')
        if xlabel:
            a.set_xlabel(xlabel, fontsize=8, color='#9da3ac')
        a.grid(True, alpha=0.15, color='#e4e7ec', linestyle='--')
        a.tick_params(labelsize=7, colors='#9da3ac')
        for s in a.spines.values():
            s.set_visible(False)
        self.fig.tight_layout(pad=1.2)
        self.draw()


def _gpu_stats():
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        i = torch.cuda.current_device()
        t = torch.cuda.get_device_properties(i).total_mem / 1e6
        a = torch.cuda.memory_allocated(i) / 1e6
        return {"u": a, "t": t}
    except Exception:
        return None


class DashboardTab(QWidget):

    def __init__(self):
        super().__init__()
        self.iter_m = {"step": [], "loss": [], "lr": [], "tps": [], "gmem": []}
        self.exp_path = None
        self.cur_step = 0
        self._last_pipe_epoch = None
        self._lfp = 0
        self._lfn = None
        self.setup_ui()
        self.rt = QTimer()
        self.rt.timeout.connect(self.auto_refresh)

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        splitter = QSplitter(Qt.Vertical)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        config_w = QWidget()
        cl = QVBoxLayout(config_w)
        cl.setContentsMargins(4, 4, 4, 4)
        cl.setSpacing(4)

        cl.addWidget(self._build_info_banner())

        row1 = QHBoxLayout()
        row1.addWidget(self._build_experiment_group(), 5)
        row1.addWidget(self._build_status_group(), 3)
        cl.addLayout(row1)

        cl.addWidget(self._build_charts_group())
        cl.addWidget(self._build_checkpoints_group())
        cl.addLayout(self._build_progress())

        scroll.setWidget(config_w)
        splitter.addWidget(scroll)
        splitter.addWidget(self._build_log())
        splitter.setSizes([440, 280])
        main_layout.addWidget(splitter)

        self.load_experiments()
        self._init_charts()

    def _build_info_banner(self):
        frame = QFrame()
        frame.setStyleSheet(
            "QFrame{background:#e3f2fd;border:1px solid #90caf9;"
            "border-radius:6px;padding:10px 14px;}")
        lay = QVBoxLayout(frame)
        lay.setSpacing(4)
        title = QLabel("Training Dashboard")
        title.setStyleSheet("font-size:14px;font-weight:700;color:#1565c0;")
        lay.addWidget(title)
        desc = QLabel(
            "Monitor training progress in real-time. Select an experiment "
            "directory and log file, then enable auto-refresh to track "
            "loss, learning rate, throughput, and GPU memory.\n\n"
            "Parses torchtune log format: "
            "Step N | loss:... | lr:... | tokens_per_second:...")
        desc.setStyleSheet("font-size:12px;color:#0d47a1;")
        desc.setWordWrap(True)
        lay.addWidget(desc)
        return frame

    def _build_experiment_group(self):
        g = QGroupBox("Experiment & Log Selection")
        lay = QVBoxLayout(g)

        recipe_lay = QHBoxLayout()
        recipe_lay.addWidget(QLabel("Experiment:"))
        self.exp_combo = QComboBox()
        self.exp_combo.setStyleSheet(COMBO_STYLE)
        self.exp_combo.setMinimumWidth(280)
        self.exp_combo.currentTextChanged.connect(self._on_exp)
        recipe_lay.addWidget(self.exp_combo)
        recipe_lay.addStretch()
        lay.addLayout(recipe_lay)

        config_lay = QHBoxLayout()
        config_lay.addWidget(QLabel("Log File:"))
        self.log_combo = QComboBox()
        self.log_combo.setStyleSheet(COMBO_STYLE)
        self.log_combo.setMinimumWidth(350)
        self.log_combo.currentTextChanged.connect(self._on_log)
        config_lay.addWidget(self.log_combo)
        config_lay.addStretch()
        lay.addLayout(config_lay)

        btn_lay = QHBoxLayout()
        for text, style, slot in [
            ("Refresh", BTN_SUCCESS, self.manual_refresh),
            ("Clear Log", BTN_DANGER, self.clear_log),
            ("Clear All", BTN_WARNING, self.clear_all),
        ]:
            b = QPushButton(text)
            b.setFixedHeight(32)
            b.setCursor(Qt.PointingHandCursor)
            b.setStyleSheet(style)
            b.clicked.connect(slot)
            btn_lay.addWidget(b)

        self.auto_chk = QCheckBox("Auto-refresh (2s)")
        self.auto_chk.setStyleSheet(CHECK_STYLE)
        self.auto_chk.toggled.connect(self._toggle_auto)
        btn_lay.addWidget(self.auto_chk)
        btn_lay.addStretch()
        lay.addLayout(btn_lay)

        return g

    def _build_status_group(self):
        g = QGroupBox("Live Metrics")
        lay = QVBoxLayout(g)

        self.gpu_info_frame = QFrame()
        self.gpu_info_frame.setStyleSheet(
            "QFrame{background:#e8edf3;border:1px solid #c9cdd3;"
            "border-radius:4px;padding:8px}")
        gif_lay = QGridLayout(self.gpu_info_frame)
        gif_lay.setSpacing(4)

        gif_lay.addWidget(QLabel("Step:"), 0, 0)
        self.lbl_step = QLabel("\u2013")
        self.lbl_step.setStyleSheet("font-weight:600;color:#394867;font-size:13px;")
        gif_lay.addWidget(self.lbl_step, 0, 1)

        gif_lay.addWidget(QLabel("Loss:"), 1, 0)
        self.lbl_loss = QLabel("\u2013")
        self.lbl_loss.setStyleSheet("font-weight:600;color:#c0392b;font-size:13px;")
        gif_lay.addWidget(self.lbl_loss, 1, 1)

        gif_lay.addWidget(QLabel("Best Loss:"), 2, 0)
        self.lbl_best = QLabel("\u2013")
        self.lbl_best.setStyleSheet("font-weight:600;color:#3a7d44;font-size:13px;")
        gif_lay.addWidget(self.lbl_best, 2, 1)

        gif_lay.addWidget(QLabel("LR:"), 3, 0)
        self.lbl_lr = QLabel("\u2013")
        self.lbl_lr.setStyleSheet("font-weight:600;color:#b45309;font-size:13px;")
        gif_lay.addWidget(self.lbl_lr, 3, 1)

        gif_lay.addWidget(QLabel("Tokens/s:"), 4, 0)
        self.lbl_tps = QLabel("\u2013")
        self.lbl_tps.setStyleSheet("font-weight:600;color:#2e6f8e;font-size:13px;")
        gif_lay.addWidget(self.lbl_tps, 4, 1)

        gif_lay.addWidget(QLabel("GPU Mem:"), 5, 0)
        self.lbl_gpu = QLabel("\u2013")
        self.lbl_gpu.setStyleSheet("font-weight:600;color:#6a1b9a;font-size:13px;")
        gif_lay.addWidget(self.lbl_gpu, 5, 1)

        lay.addWidget(self.gpu_info_frame)

        self.status_badge = QLabel("Idle")
        self.status_badge.setStyleSheet(
            "background:#e8edf3;color:#697586;font-weight:600;font-size:12px;"
            "border:1px solid #c9cdd3;border-radius:4px;padding:6px;")
        self.status_badge.setAlignment(Qt.AlignCenter)
        lay.addWidget(self.status_badge)

        return g

    def _build_charts_group(self):
        g = QGroupBox("Training Charts")
        lay = QGridLayout(g)
        lay.setSpacing(8)

        self.loss_chart = _Chart()
        self.lr_chart = _Chart()
        self.tps_chart = _Chart()
        self.gpu_chart = _Chart()

        lay.addWidget(self.loss_chart, 0, 0)
        lay.addWidget(self.lr_chart, 0, 1)
        lay.addWidget(self.tps_chart, 1, 0)
        lay.addWidget(self.gpu_chart, 1, 1)
        return g

    def _build_checkpoints_group(self):
        g = QGroupBox("Saved Checkpoints")
        lay = QVBoxLayout(g)

        self.ckpt_table = QTableWidget()
        self.ckpt_table.setColumnCount(3)
        self.ckpt_table.setHorizontalHeaderLabels(["Checkpoint", "Size", "Modified"])
        self.ckpt_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.ckpt_table.verticalHeader().setVisible(False)
        self.ckpt_table.verticalHeader().setDefaultSectionSize(26)
        self.ckpt_table.setAlternatingRowColors(True)
        self.ckpt_table.setMaximumHeight(150)
        lay.addWidget(self.ckpt_table)
        return g

    def _build_progress(self):
        lay = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setStyleSheet(PROGRESS_STYLE)
        lay.addWidget(self.progress_bar)
        self.progress_label = QLabel("Ready")
        self.progress_label.setMinimumWidth(200)
        lay.addWidget(self.progress_label)
        return lay

    def _build_log(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(5, 5, 5, 5)
        hdr = QHBoxLayout()
        t = QLabel("Live Training Log")
        t.setStyleSheet(LABEL_HEADING)
        hdr.addWidget(t)
        cb = QPushButton("Clear")
        cb.setFixedWidth(80)
        cb.setStyleSheet(BTN_SECONDARY)
        cb.clicked.connect(lambda: self.log_view.clear())
        hdr.addWidget(cb)
        self.clear_on_start_check = QCheckBox("Clear on start")
        self.clear_on_start_check.setChecked(True)
        self.clear_on_start_check.setStyleSheet(LABEL_SECONDARY)
        hdr.addWidget(self.clear_on_start_check)
        hdr.addStretch()
        lay.addLayout(hdr)

        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setStyleSheet("""
            QPlainTextEdit { background-color:#1a1a2e; color:#a3d977;
                font-family:'Consolas',monospace; font-size:12px;
                border-radius:4px; padding:10px; }
        """)
        self.log_view.setMaximumBlockCount(2000)
        lay.addWidget(self.log_view)
        return w

    def _init_charts(self):
        self.loss_chart.plot([], [], '#c0392b', 'Training Loss', 'Step')
        self.lr_chart.plot([], [], '#394867', 'Learning Rate', 'Step')
        self.tps_chart.plot([], [], '#3a7d44', 'Tokens / Second', 'Step')
        self.gpu_chart.plot([], [], '#6a1b9a', 'GPU Memory (GB)', 'Step')

    # ------------------------------------------------------------------ #
    #  Experiment / Log discovery
    # ------------------------------------------------------------------ #

    def load_experiments(self):
        self.exp_combo.blockSignals(True)
        cur = self.exp_combo.currentText()
        self.exp_combo.clear()
        pr = _project_root()

        ds = []
        for base in ("output", "workspace"):
            d = pr / base
            if d.is_dir():
                for sub in d.iterdir():
                    if sub.is_dir():
                        try:
                            ds.append((sub, sub.stat().st_mtime))
                        except OSError:
                            ds.append((sub, 0))

        seen = set()
        for d, _ in sorted(ds, key=lambda x: -x[1]):
            try:
                key = str(d.resolve())
            except OSError:
                continue
            if key not in seen:
                seen.add(key)
                self.exp_combo.addItem(d.name, str(d))

        i = self.exp_combo.findText(cur)
        self.exp_combo.setCurrentIndex(max(i, 0) if self.exp_combo.count() else -1)
        self.exp_combo.blockSignals(False)
        if self.exp_combo.currentText():
            self.exp_path = Path(self.exp_combo.currentData() or "")
            self._load_logs()

    def _load_logs(self):
        self.log_combo.blockSignals(True)
        cur = self.log_combo.currentText()
        self.log_combo.clear()
        if self.exp_path and self.exp_path.exists():
            logs = _collect_log_paths(self.exp_path)
            for l in logs:
                self.log_combo.addItem(l.name)
            if self.log_combo.count():
                f = self.log_combo.itemText(0)
                self.log_combo.setItemText(0, f"{f} (Latest)")
        if cur:
            cl = cur.replace(" (Latest)", "")
            for i in range(self.log_combo.count()):
                if self.log_combo.itemText(i).replace(" (Latest)", "") == cl:
                    self.log_combo.setCurrentIndex(i)
                    break
        if self.log_combo.currentIndex() < 0 and self.log_combo.count():
            self.log_combo.setCurrentIndex(0)
        self.log_combo.blockSignals(False)

    def _on_exp(self, name):
        if name:
            stored = self.exp_combo.itemData(self.exp_combo.currentIndex())
            self.exp_path = Path(stored) if stored else _project_root() / "output" / name
            self._reset()
            self._load_logs()
            self._refresh()

    def _on_log(self, _):
        self._reset()
        self._refresh()

    # ------------------------------------------------------------------ #
    #  Monitoring
    # ------------------------------------------------------------------ #

    def _toggle_auto(self, on):
        if on:
            self.rt.start(2000)
            self._set_badge("Monitoring", "#e8edf3", "#2e6f8e")
        else:
            self.rt.stop()
            self._set_badge("Idle", "#e8edf3", "#697586")

    def start_monitoring(self):
        self._reset()
        if self.clear_on_start_check.isChecked():
            self.log_view.clear()
        self.auto_chk.setChecked(True)
        self.load_experiments()
        QTimer.singleShot(1000, self._refresh)
        self._set_badge("Training", "#e9f5ec", "#2d6235")
        self.progress_bar.setRange(0, 0)
        self.progress_label.setText("Training...")

    def stop_monitoring(self):
        self.auto_chk.setChecked(False)
        self._set_badge("Stopped", "#fbe9e7", "#96281b")
        self.progress_bar.setRange(0, 100)
        self.progress_label.setText("Stopped")

    def manual_refresh(self):
        self.load_experiments()
        self._refresh()

    def auto_refresh(self):
        self._refresh()

    def _set_badge(self, text, bg, fg):
        self.status_badge.setText(text)
        self.status_badge.setStyleSheet(
            f"background:{bg};color:{fg};font-weight:600;font-size:12px;"
            f"border:1px solid #c9cdd3;border-radius:4px;padding:6px;")

    # ------------------------------------------------------------------ #
    #  Clear
    # ------------------------------------------------------------------ #

    def clear_log(self):
        s = self.log_combo.currentText()
        if not s or not self.exp_path:
            return
        c = s.replace(" (Latest)", "")
        lp = self.exp_path / c
        if QMessageBox.question(
                self, "Delete", f"Delete {c}?",
                QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes:
            try:
                if lp.exists():
                    lp.unlink()
                self._reset()
                self._init_charts()
                self._load_logs()
                self._refresh()
            except OSError as e:
                QMessageBox.critical(self, "Error", str(e))

    def clear_all(self):
        if not self.exp_path or not self.exp_path.exists():
            return
        logs = _collect_log_paths(self.exp_path)
        if not logs:
            return
        if QMessageBox.question(
                self, "Delete All", f"Delete ALL {len(logs)} logs?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        for l in logs:
            try:
                l.unlink()
            except OSError:
                pass
        self._reset()
        self._init_charts()
        self._load_logs()

    # ------------------------------------------------------------------ #
    #  Data
    # ------------------------------------------------------------------ #

    def _reset(self):
        self.iter_m = {"step": [], "loss": [], "lr": [], "tps": [], "gmem": []}
        self._lfp = 0
        self._lfn = None
        self.cur_step = 0
        self._last_pipe_epoch = None
        self.log_view.clear()

    def _parse(self, lf):
        lp = str(lf)
        if lp != self._lfn:
            self._reset()
            self._lfn = lp
        try:
            with open(lf, encoding="utf-8", errors="replace") as f:
                f.seek(self._lfp)
                lines = f.readlines()
                self._lfp = f.tell()
        except OSError:
            return
        if not lines:
            return

        for line in lines:
            self.log_view.appendPlainText(line.rstrip())
            step, loss_v, lr_v, tps_v, gmem_v = None, None, None, None, None

            pm = _RE_PIPE.search(line)
            if pm:
                try:
                    self._last_pipe_epoch = int(pm.group(1))
                    step = int(pm.group(2))
                    loss_v = float(pm.group(3))
                    lr_v = float(pm.group(4))
                except ValueError:
                    continue
                self.cur_step = step
            else:
                sm = _RE_STEP_LOSS.search(line)
                if sm:
                    try:
                        step = int(sm.group(1))
                        loss_v = float(sm.group(2))
                    except ValueError:
                        continue
                    self.cur_step = step

            if step is None:
                continue

            lm = _RE_LR.search(line)
            if lm and lr_v is None:
                try:
                    lr_v = float(lm.group(1))
                except ValueError:
                    pass
            tm = _RE_TPS.search(line)
            if tm:
                try:
                    tps_v = float(tm.group(1))
                except ValueError:
                    pass
            gm = _RE_GPU_MEM.search(line)
            if gm:
                try:
                    gmem_v = float(gm.group(1))
                except ValueError:
                    pass

            im = self.iter_m
            st = im["step"]
            if st and step == st[-1]:
                if loss_v is not None:
                    im["loss"][-1] = loss_v
                if lr_v is not None:
                    im["lr"][-1] = lr_v
                if tps_v is not None:
                    im["tps"][-1] = tps_v
                if gmem_v is not None:
                    im["gmem"][-1] = gmem_v
            else:
                _prev = lambda k: im[k][-1] if im[k] else 0.0
                st.append(step)
                im["loss"].append(loss_v if loss_v is not None else _prev("loss"))
                im["lr"].append(lr_v if lr_v is not None else _prev("lr"))
                im["tps"].append(tps_v if tps_v is not None else _prev("tps"))
                im["gmem"].append(gmem_v if gmem_v is not None else _prev("gmem"))

    # ------------------------------------------------------------------ #
    #  Refresh
    # ------------------------------------------------------------------ #

    def _refresh(self):
        if not self.exp_path or not self.exp_path.exists():
            self.progress_label.setText("No experiment selected")
            return

        s = self.log_combo.currentText()
        if s:
            c = s.replace(" (Latest)", "")
            lp = self.exp_path / c
            try:
                if lp.exists():
                    self._parse(lp)
                    self._update_ui()
            except Exception as e:
                self.progress_label.setText(f"Error: {e}")
        else:
            self.progress_label.setText("No log file")

        self._load_ckpts()
        gc.collect()

    def _update_ui(self):
        im = self.iter_m
        n = len(im["step"])

        losses = [x for x in im["loss"] if x and x == x]
        if im["step"]:
            self.lbl_step.setText(str(im["step"][-1]))
        if losses:
            self.lbl_loss.setText(f"{im['loss'][-1]:.4f}")
            self.lbl_best.setText(f"{min(losses):.4f}")

        last_lr = next((x for x in reversed(im["lr"]) if x and x == x), None)
        if last_lr is not None:
            self.lbl_lr.setText(f"{last_lr:.2e}")

        last_tps = next((x for x in reversed(im["tps"]) if x and x == x), None)
        if last_tps is not None:
            self.lbl_tps.setText(f"{last_tps:.0f}")

        last_gpu = next((x for x in reversed(im["gmem"]) if x and x == x), None)
        if last_gpu is not None:
            self.lbl_gpu.setText(f"{last_gpu:.1f} GB")

        g = _gpu_stats()
        if g:
            if last_gpu is None:
                self.lbl_gpu.setText(f"{g['u']:.0f} / {g['t']:.0f} MB")

        ep = f"ep {self._last_pipe_epoch} | " if self._last_pipe_epoch else ""
        self.progress_label.setText(
            f"{ep}Step {self.cur_step} | {n} data points" if n else "Waiting for data...")

        st = im["step"]
        if st:
            self.loss_chart.plot(st, im["loss"], '#c0392b', 'Training Loss', 'Step')
            self.lr_chart.plot(st, im["lr"], '#394867', 'Learning Rate', 'Step')
            self.tps_chart.plot(st, im["tps"], '#3a7d44', 'Tokens / Second', 'Step')
            self.gpu_chart.plot(st, im["gmem"], '#6a1b9a', 'GPU Memory (GB)', 'Step')

    def _load_ckpts(self):
        if not self.exp_path:
            return
        try:
            cs = []
            for ext in ("*.pt", "*.pth", "*.safetensors", "*.bin"):
                cs.extend(self.exp_path.rglob(ext))
            cs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        except OSError:
            cs = []
        self.ckpt_table.setRowCount(len(cs))
        for i, c in enumerate(cs):
            try:
                s = c.stat()
                self.ckpt_table.setItem(i, 0, QTableWidgetItem(c.name))
                self.ckpt_table.setItem(i, 1, QTableWidgetItem(f"{s.st_size / 1e6:.1f} MB"))
                self.ckpt_table.setItem(i, 2, QTableWidgetItem(time.ctime(s.st_mtime)))
            except OSError:
                self.ckpt_table.setItem(i, 0, QTableWidgetItem(c.name))
