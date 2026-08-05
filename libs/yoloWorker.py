try:
    from PyQt5.QtCore import *
except ImportError:
    from PyQt4.QtCore import *


class YoloWorker(QObject):
    """Runs YOLO model loading and inference off the GUI thread.

    The worker never touches widgets: it emits raw shape tuples and the main
    thread is responsible for load_labels/save_file and everything else Qt.
    """

    modelLoading = pyqtSignal(str)          # basename of the .pt file
    modelLoaded = pyqtSignal()
    imageStarted = pyqtSignal(int, int, str)  # index, total, basename
    imageFinished = pyqtSignal(int, str, object)  # index, path, shapes list
    failed = pyqtSignal(str)
    finished = pyqtSignal(int)              # number of images actually processed

    def __init__(self, model_path, image_paths, conf, imgsz,
                 contour_extractor=None, parent=None):
        super(YoloWorker, self).__init__(parent)
        self.model_path = model_path
        self.image_paths = list(image_paths)
        self.conf = conf
        self.imgsz = imgsz
        # Callback used to split segmentation masks into polygons. It is pure
        # cv2/numpy work, so it is safe to call from this thread.
        self.contour_extractor = contour_extractor
        self.model = None
        self._abort = False
        # The GUI thread consumes results one at a time (loading and saving each
        # image); without this handshake the worker would race ahead and queue
        # an unbounded number of results.
        self._ack = QSemaphore(0)

    def abort(self):
        self._abort = True
        self._ack.release()

    def is_aborted(self):
        return self._abort

    def ack(self):
        """Called from the GUI thread once a result has been consumed."""
        self._ack.release()

    def run(self):
        try:
            if self.model is None:
                import os
                self.modelLoading.emit(os.path.basename(self.model_path))
                from ultralytics import YOLO
                self.model = YOLO(self.model_path)
                self.modelLoaded.emit()
        except Exception as e:
            self.failed.emit("Error cargando YOLO: %s" % str(e))
            self.finished.emit(0)
            return

        if self._abort:
            self.finished.emit(0)
            return

        import os
        total = len(self.image_paths)
        processed = 0

        for idx, img_path in enumerate(self.image_paths):
            if self._abort:
                break

            self.imageStarted.emit(idx, total, os.path.basename(img_path))
            try:
                shapes = self._predict(img_path)
            except Exception as e:
                self.failed.emit("Error procesando %s: %s"
                                 % (os.path.basename(img_path), str(e)))
                continue

            if self._abort:
                break

            self.imageFinished.emit(idx, img_path, shapes)
            self._ack.acquire()
            processed += 1

        self.finished.emit(processed)

    def _predict(self, img_path):
        results = self.model.predict(img_path, conf=self.conf, imgsz=self.imgsz,
                                     rect=True, verbose=False)
        if not results:
            return []

        result = results[0]
        shapes = []
        is_segment = hasattr(result, 'masks') and result.masks is not None

        if is_segment:
            img_shape = result.orig_shape
            for mask, box in zip(result.masks.xy, result.boxes):
                class_idx = int(box.cls[0].item())
                class_name = self.model.names[class_idx]
                if self.contour_extractor is None:
                    continue
                for poly in self.contour_extractor(mask, img_shape, class_name):
                    shapes.append((class_name, poly, None, None, False, True))
        else:
            for box in result.boxes:
                class_idx = int(box.cls[0].item())
                class_name = self.model.names[class_idx]
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                points = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
                shapes.append((class_name, points, None, None, False, True))

        return shapes
