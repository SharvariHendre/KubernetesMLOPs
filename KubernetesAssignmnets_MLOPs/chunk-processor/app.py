from flask import Flask, jsonify
import pandas as pd
import threading
import os
import numpy as np

# Initialize the Flask app
app = Flask(__name__)

# Lock to handle thread synchronization for shared resource (statistics)
statistics_lock = threading.Lock()

class ChunkIterator:
    def __init__(self, file_path, chunk_size=100):
        """
        Initialize the ChunkIterator.
        :param file_path: Path to the dataset CSV file.
        :param chunk_size: Number of rows to process in each chunk.
        """
        self.file_path = file_path
        self.chunk_size = chunk_size
        self.iterator = pd.read_csv(self.file_path, chunksize=self.chunk_size)

    def __iter__(self):
        """
        Make the ChunkIterator class iterable.
        """
        return self

    def __next__(self):
        """
        Return the next chunk of data.
        """
        try:
            chunk = next(self.iterator)
        except StopIteration:
            raise StopIteration
        return chunk

    def calculate_statistics(self, chunk):
        """
        Calculate basic statistics (mean, median) for numerical columns in the chunk.
        :param chunk: A chunk of the dataset.
        :return: A dictionary containing statistics.
        """
        stats = {}
        for column in chunk.select_dtypes(include=['float64', 'int64']):
            stats[column] = {
                'mean': chunk[column].mean(),
                'median': chunk[column].median(),
                'min': chunk[column].min(),
                'max': chunk[column].max(),
            }
        return stats

# Global variable to store statistics
statistics = {}

def process_dataset(file_path, chunk_size):
    """
    Process the dataset in chunks and store statistics for each chunk.
    :param file_path: Path to the dataset CSV file.
    :param chunk_size: Number of rows to process in each chunk.
    """
    global statistics
    chunk_iterator = ChunkIterator(file_path, chunk_size)
    
    for chunk in chunk_iterator:
        print("Processing a new chunk...")
        
        # Calculate statistics for the current chunk
        stats = chunk_iterator.calculate_statistics(chunk)
        
        # Store the statistics for each chunk in a thread-safe manner
        with statistics_lock:
            statistics.update(stats)
        print(f"Statistics updated for chunk!")

@app.route('/')
def index():
    return "Welcome to the chunk processing service!"

@app.route('/process')
def process():
    # Check if the file exists
    file_path = "Mall_Customers.csv"
    if not os.path.exists(file_path):
        return f"Error: The file {file_path} does not exist!"

    # Run the dataset processing in a separate thread to avoid blocking Flask
    threading.Thread(target=process_dataset, args=(file_path, 50)).start()
    return "Dataset processing started in the background!"

@app.route('/results')
def get_results():
    # Check if statistics are available
    if not statistics:
        return "No results available yet. Please process the dataset first."
    
    # Convert the statistics to a JSON serializable format
    serializable_stats = {}
    for column, stats in statistics.items():
        serializable_stats[column] = {
            stat: (float(value) if isinstance(value, (np.int64, np.float64)) else value)
            for stat, value in stats.items()
        }
    
    return jsonify(serializable_stats)

if __name__ == '__main__':
    # Start Flask application
    app.run(host='0.0.0.0', port=5000)