from data_downloader import CARDDataDownloader
from pathlib import Path
import json

class DataInitializer:
    def __init__(self, socketio=None):
        self.status_file = Path('data/download_status.json')
        self.downloader = CARDDataDownloader(socketio=socketio)
        self.socketio = socketio
        
    def get_status(self):
        if not self.status_file.exists():
            return {'status': 'not_started', 'message': 'Download not started'}
        with open(self.status_file) as f:
            return json.load(f)
            
    def update_status(self, status, message):
        self.status_file.parent.mkdir(exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump({'status': status, 'message': message}, f)
            
    def initialize_data(self):
        try:
            self.update_status('downloading', 'Downloading CARD data...')
            self.downloader.run()
            self.update_status('completed', 'CARD data downloaded successfully')
        except Exception as e:
            self.update_status('error', f'Error downloading data: {str(e)}')
            raise
