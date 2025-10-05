# 1. Start from an official Python base image
# This provides a clean Linux environment with a specific Python version installed.
FROM python:3.9-slim

# 2. Set the working directory inside the container
# This is where our application files will live.
WORKDIR /app

# 3. Copy the necessary files into the container
# First, copy the requirements file.
COPY requirements.txt .

# 4. Install the Python dependencies
# This step is done before copying the rest of the code to leverage Docker's layer caching.
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application source code and artifacts
# This includes app.py, the models/, processors/, and templates/ folders.
COPY . .

# 6. Expose the port the app runs on
# This tells Docker that the container listens on port 5000.
EXPOSE 5000

# 7. Define the command to run when the container starts
# This is the command that launches our Flask application.
CMD ["python", "app.py"]