# Use a Python base image
FROM python:3.10

# Set the working directory
WORKDIR /app

# Copy the application files
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose the correct port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "CGGTTS_Analyser.py", "--server.port=8501", "--server.address=0.0.0.0"]
