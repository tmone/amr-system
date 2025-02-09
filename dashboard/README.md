# yolo-dashboard README.md

# YOLO Dashboard

This project is a web application that allows users to upload images, videos, or compressed folders containing images to perform object detection using a trained YOLO model. The application provides a user-friendly interface for visualizing predictions and results.

## Features

- **Image Upload**: Upload a single image to predict and display the image with bounding boxes and labels for detected objects.
- **Video Upload**: Upload a video to play and detect objects in real-time, with different colors for each class.
- **Batch Processing**: Upload a compressed folder containing multiple images to predict on all images and summarize the results.

## Project Structure

```
dashboard
├── src
│   ├── app.py                # Main entry point of the application
│   ├── templates             # HTML templates for the dashboard
│   │   ├── index.html        # Main dashboard interface
│   │   ├── image.html        # Results page for image predictions
│   │   ├── video.html        # Results page for video predictions
│   │   └── batch.html        # Results page for batch processing
│   ├── static                # Static files (CSS, JS, uploads)
│   │   ├── css
│   │   │   └── style.css     # Styles for the dashboard
│   │   ├── js
│   │   │   └── main.js       # JavaScript functions for UI interactions
│   │   └── uploads           # Directory for temporary uploads
│   ├── utils                 # Utility functions for processing
│   │   ├── detector.py       # Functions for image predictions
│   │   ├── video.py          # Functions for video processing
│   │   └── batch_processor.py # Functions for batch image processing
│   └── config
│       └── model_config.yaml # Configuration settings for the YOLO model
├── requirements.txt          # Python dependencies
├── README.md                 # Project documentation
└── .gitignore                # Files to ignore by Git
```

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd dashboard
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the application:
   ```
   python src/app.py
   ```

4. Open your web browser and navigate to `http://localhost:5000` to access the dashboard.

## Usage Guidelines

- Use the dashboard to upload images, videos, or folders for object detection.
- Follow the on-screen instructions for each upload type.
- Review the results displayed on the respective pages for images, videos, and batch processing.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.