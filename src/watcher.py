import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from src.model_loader import load_model

class ModelFileChangeHandler(FileSystemEventHandler):
    def __init__(self, model_path: str, reload_callback):
        self.model_path = model_path
        self.reload_callback = reload_callback

    def on_modified(self, event):
        if event.src_path == self.model_path:
            self.reload_callback()

def watch_model_file(model_path: str, reload_callback):
    event_handler = ModelFileChangeHandler(model_path, reload_callback)
    observer = Observer()
    observer.schedule(event_handler, os.path.dirname(model_path), recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()