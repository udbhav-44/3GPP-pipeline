"""
WebSocket server that watches files and streams updates to clients.

- ProcessLogs.md → plain appended logs (NO diff format)
- Results.csv     → structured table for sidebar
"""

import time
import json
import threading
import asyncio
import websockets
import logging
from logging_config import setup_logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import queue
from pathlib import Path
import csv

BASE_DIR = Path(__file__).resolve().parent
setup_logging()
logger = logging.getLogger("watcher")

# --------------------------------------------------
# File System Event Handler
# --------------------------------------------------

class MyHandler(FileSystemEventHandler):
    def __init__(self, message_queue):
        super().__init__()
        self.message_queue = message_queue

        # Track how much of the log file we've already read
        self.log_offset = 0

        self.watch_files = {
            BASE_DIR / "ProcessLogs.md": "logs",
            BASE_DIR / "Results.csv": "results"
        }

    def on_modified(self, event):
        self._handle_event(event.src_path)

    def on_created(self, event):
        self._handle_event(event.src_path)

    def on_moved(self, event):
        self._handle_event(event.dest_path)

    def _handle_event(self, src_path):
        resolved_path = Path(src_path).resolve()

        if resolved_path not in self.watch_files:
            return

        file_type = self.watch_files[resolved_path]

        if file_type == "logs":
            self.handle_logs(resolved_path)

        elif file_type == "results":
            self.handle_results(resolved_path)

    # --------------------------------------------------
    # ProcessLogs.md → plain tail-style logs
    # --------------------------------------------------
    def handle_logs(self, path: Path):
        try:
            # Handle file truncation / rewrite
            if path.stat().st_size < self.log_offset:
                self.log_offset = 0

            with open(path, "r") as f:
                f.seek(self.log_offset)
                new_content = f.read()
                self.log_offset = f.tell()

            if new_content.strip():
                self.message_queue.put({
                    "type": "logs",
                    "response": new_content
                })

        except FileNotFoundError:
            pass

    # --------------------------------------------------
    # Results.csv → sidebar table
    # --------------------------------------------------
    def handle_results(self, path: Path):
        try:
            with open(path, "r") as f:
                reader = csv.DictReader(f)
                rows = list(reader)

            self.message_queue.put({
                "type": "results",
                "format": "table",
                "columns": reader.fieldnames,
                "rows": rows
            })

        except FileNotFoundError:
            pass


# --------------------------------------------------
# Observer Thread
# --------------------------------------------------

def start_observer(message_queue):
    event_handler = MyHandler(message_queue)
    observer = Observer()
    observer.schedule(event_handler, path=str(BASE_DIR), recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(0.3)
    except KeyboardInterrupt:
        observer.stop()

    observer.join()


# --------------------------------------------------
# WebSocket Handler
# --------------------------------------------------

async def handle_connection(websocket):
    logger.info("Client connected")

    message_queue = queue.Queue()

    observer_thread = threading.Thread(
        target=start_observer,
        args=(message_queue,),
        daemon=True
    )
    observer_thread.start()

    try:
        while True:
            try:
                event_data = message_queue.get_nowait()
                await websocket.send(json.dumps(event_data))
            except queue.Empty:
                await asyncio.sleep(0.2)

    except websockets.exceptions.ConnectionClosed:
        logger.info("Client disconnected")


# --------------------------------------------------
# Server Entrypoint
# --------------------------------------------------

async def main():
    logger.info("WebSocket server running on ws://0.0.0.0:8090")
    async with websockets.serve(handle_connection, "0.0.0.0", 8090):
        await asyncio.Future()  # run forever


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Server shutdown")
