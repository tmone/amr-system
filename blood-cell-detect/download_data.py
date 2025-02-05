import kagglehub
import os

def download_dataset():
    try:
        print("Downloading blood cell detection dataset...")
        path = kagglehub.dataset_download("quangnguynvnnn/bloodcell-yolo-format")
        print(f"Dataset downloaded successfully to: {path}")
        return path
    except Exception as e:
        print(f"Error downloading dataset: {str(e)}")
        return None

if __name__ == "__main__":
    download_path = download_dataset()
    if download_path:
        print("Ready to use the dataset!")
