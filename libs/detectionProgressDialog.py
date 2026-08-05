try:
    from PyQt5.QtGui import *
    from PyQt5.QtCore import *
    from PyQt5.QtWidgets import *
except ImportError:
    from PyQt4.QtGui import *
    from PyQt4.QtCore import *

TICK_MS = 100


class DetectionProgressDialog(QDialog):
    """Modal progress feedback for YOLO autodetection.

    Batch runs report a real percentage plus an ETA. A single image has no
    progress to report from ultralytics, so the bar goes indeterminate during
    inference rather than faking a percentage; the elapsed clock is what tells
    the user the run is still alive.
    """

    cancelled = pyqtSignal()

    def __init__(self, total, parent=None):
        super(DetectionProgressDialog, self).__init__(parent)
        self.total = total
        self.batch = total > 1
        self.done_count = 0
        self._image_times = []
        self._image_started_at = None
        self._finished = False

        self.setWindowTitle("Autodetección YOLO")
        self.setModal(True)
        self.setWindowFlags((self.windowFlags() | Qt.CustomizeWindowHint)
                            & ~Qt.WindowCloseButtonHint
                            & ~Qt.WindowContextHelpButtonHint)
        self.setMinimumWidth(380)

        self.phase_label = QLabel("Preparando…")
        self.phase_label.setWordWrap(True)

        self.progress = QProgressBar()
        self.progress.setTextVisible(self.batch)
        if self.batch:
            self.progress.setRange(0, total)
            self.progress.setValue(0)
            self.progress.setFormat("%v / %m (%p %)")
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(0)

        self.stats_label = QLabel("Tiempo: 00:00.0")
        stats_font = self.stats_label.font()
        stats_font.setPointSize(max(8, stats_font.pointSize() - 1))
        self.stats_label.setFont(stats_font)

        self.cancel_button = QPushButton("Cancelar")
        self.cancel_button.clicked.connect(self.on_cancel)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        buttons.addWidget(self.cancel_button)

        layout = QVBoxLayout(self)
        layout.addWidget(self.phase_label)
        layout.addWidget(self.progress)
        layout.addWidget(self.stats_label)
        layout.addLayout(buttons)

        self.elapsed = QElapsedTimer()
        self.elapsed.start()
        self.ticker = QTimer(self)
        self.ticker.setInterval(TICK_MS)
        self.ticker.timeout.connect(self.update_stats)
        self.ticker.start()

    # -- worker slots ----------------------------------------------------

    def on_model_loading(self, model_name):
        self.phase_label.setText("Cargando modelo %s…" % model_name)
        if not self.batch:
            self.progress.setRange(0, 100)
            self.progress.setValue(15)

    def on_model_loaded(self):
        if not self.batch:
            self.progress.setValue(40)

    def on_image_started(self, idx, total, name):
        self._image_started_at = self.elapsed.elapsed()
        if self.batch:
            self.phase_label.setText("Analizando %d/%d: %s" % (idx + 1, total, name))
        else:
            self.phase_label.setText("Analizando %s…" % name)
            # No progress signal available from predict(): go indeterminate.
            self.progress.setRange(0, 0)

    def on_image_finished(self, idx, path, shapes):
        if self._image_started_at is not None:
            self._image_times.append(self.elapsed.elapsed() - self._image_started_at)
            self._image_started_at = None
        self.done_count += 1
        if self.batch:
            self.progress.setValue(self.done_count)
        else:
            self.progress.setRange(0, 100)
            self.progress.setValue(100)
        self.update_stats()

    def on_finished(self, summary):
        """Freeze the dialog on its result so an empty run is unambiguous."""
        self._finished = True
        self.ticker.stop()
        self.update_stats()
        self.phase_label.setText(summary)
        self.progress.setRange(0, 100)
        self.progress.setValue(100)
        self.cancel_button.setText("Cerrar")
        QTimer.singleShot(1000, self.accept)

    # -- internals -------------------------------------------------------

    def update_stats(self):
        parts = ["Tiempo: %s" % self.format_ms(self.elapsed.elapsed())]
        eta = self.estimate_eta()
        if eta is not None:
            parts.append("Restante: ~%s" % self.format_ms(eta))
        self.stats_label.setText("   ".join(parts))

    def estimate_eta(self):
        if not self.batch or self._finished or not self._image_times:
            return None
        remaining = self.total - self.done_count
        if remaining <= 0:
            return None
        average = sum(self._image_times) / float(len(self._image_times))
        return average * remaining

    @staticmethod
    def format_ms(ms):
        total_seconds = ms / 1000.0
        minutes = int(total_seconds // 60)
        seconds = total_seconds - minutes * 60
        return "%02d:%04.1f" % (minutes, seconds)

    def on_cancel(self):
        if self._finished:
            self.accept()
            return
        self.phase_label.setText("Cancelando…")
        self.cancel_button.setEnabled(False)
        self.cancelled.emit()

    def reject(self):
        # Esc must not leave the worker running behind an invisible dialog.
        if not self._finished:
            self.on_cancel()
            return
        super(DetectionProgressDialog, self).reject()
