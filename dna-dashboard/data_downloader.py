import requests
import tarfile
import gzip
import os
import logging
from pathlib import Path

class CARDDataDownloader:
    def __init__(self, socketio=None, base_url="https://card.mcmaster.ca/download/7/", 
                 filename="baits-v4.0.0.tar.bz2",
                 output_dir="data"):
        self.socketio = socketio
        self.base_url = base_url
        self.filename = filename
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def emit_progress(self, message, progress):
        """Emit download progress through socketio"""
        if self.socketio:
            self.socketio.emit('download_progress', {
                'message': message,
                'progress': progress
            })

    def download_file(self):
        url = f"{self.base_url}{self.filename}"
        target_path = self.output_dir / self.filename
        
        if target_path.exists():
            self.emit_progress("File already downloaded", 100)
            return target_path
            
        self.logger.info(f"Downloading from {url}")
        self.emit_progress("Starting download...", 0)
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        file_size = int(response.headers.get('content-length', 0))
        block_size = 8192
        progress = 0
        
        with open(target_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=block_size):
                f.write(chunk)
                progress += len(chunk)
                percentage = int((progress / file_size) * 50)  # 50% of total progress
                self.emit_progress(f"Downloading... {percentage}%", percentage)
        
        return target_path

    def extract_archive(self, archive_path):
        self.emit_progress("Extracting files...", 50)
        self.logger.info(f"Extracting {archive_path}")
        with tarfile.open(archive_path, 'r:bz2') as tar:
            tar.extractall(path=self.output_dir)
        self.emit_progress("Extraction completed", 75)

    def process_gzipped_fasta(self, filepath):
        self.emit_progress(f"Processing {filepath.name}...", 80)
        output_path = filepath.with_suffix('')
        if output_path.exists():
            self.logger.info(f"Already processed: {output_path}")
            return output_path
            
        self.logger.info(f"Processing gzipped file: {filepath}")
        with gzip.open(filepath, 'rt') as f_in:
            with open(output_path, 'w') as f_out:
                f_out.write(f_in.read())
        return output_path

    def run(self):
        try:
            # Download and extract archive
            archive_path = self.download_file()
            self.extract_archive(archive_path)
            
            # Process all FASTA files
            processed_files = []
            for pattern in ['*.fasta.gz', '*.fasta']:
                for file in self.output_dir.glob(pattern):
                    if file.suffix == '.gz':
                        processed_file = self.process_gzipped_fasta(file)
                    else:
                        processed_file = file
                    processed_files.append(processed_file)
                    
            self.logger.info(f"Processed {len(processed_files)} FASTA files: {[f.name for f in processed_files]}")
            self.emit_progress("Processing completed", 100)
            
        except Exception as e:
            self.emit_progress(f"Error: {str(e)}", -1)
            self.logger.error(f"Error processing CARD data: {str(e)}")
            raise

if __name__ == "__main__":
    downloader = CARDDataDownloader()
    downloader.run()
