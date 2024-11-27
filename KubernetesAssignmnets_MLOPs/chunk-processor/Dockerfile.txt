# Use a smaller Python image
FROM python:3.9-slim

# Set the working directory to /app
WORKDIR /app

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install the dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Set the environment variable for Flask to run in production mode (optional but recommended)
ENV FLASK_ENV=production

# Expose the port the app runs on (use 5000 since that's what your app is using)
EXPOSE 5000

# Start the Flask application
CMD ["python", "app.py"]