FROM python:3.12-slim

WORKDIR /app

# Copy the dependencies file to the working directory
COPY requirements.txt .

# Install any dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the content of the local src directory to the working directory
COPY . .

ENV PYTHONPATH=/app

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Run the chat_engine application
CMD ["python", "src/chat_engine.py"]
