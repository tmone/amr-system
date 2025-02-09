import os
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class PredictionCache:
    def __init__(self, cache_dir):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def get_cache_path(self, filename):
        return os.path.join(self.cache_dir, f"{filename}.json")
    
    def save_prediction(self, filename, predictions, metadata=None):
        try:
            cache_data = {
                'timestamp': datetime.now().isoformat(),
                'predictions': predictions,
                'metadata': metadata or {}
            }
            
            with open(self.get_cache_path(filename), 'w') as f:
                json.dump(cache_data, f)
            logger.info(f"Cached predictions for {filename}")
            return True
        except Exception as e:
            logger.error(f"Error caching predictions for {filename}: {e}")
            return False
    
    def get_prediction(self, filename):
        try:
            cache_path = self.get_cache_path(filename)
            if not os.path.exists(cache_path):
                return None
                
            with open(cache_path, 'r') as f:
                cache_data = json.load(f)
            logger.info(f"Retrieved cached predictions for {filename}")
            return cache_data
        except Exception as e:
            logger.error(f"Error reading cache for {filename}: {e}")
            return None
            
    def clear_cache(self, filename=None):
        try:
            if filename:
                cache_path = self.get_cache_path(filename)
                if os.path.exists(cache_path):
                    os.remove(cache_path)
            else:
                for file in os.listdir(self.cache_dir):
                    os.remove(os.path.join(self.cache_dir, file))
            return True
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
            return False
