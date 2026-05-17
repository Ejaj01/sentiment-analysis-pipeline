import os
import subprocess
import sys

if __name__ == "__main__":
    print("🚀 Initializing Sentiment Analysis Dashboard Deployment Pipeline...")

    # Locate the path to the interactive Streamlit interface
    app_path = os.path.join("src", "app.py")

    # Programmatically execute the Streamlit server using the active system python path
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", app_path], check=True)
    except KeyboardInterrupt:
        print("\n🛑 Local deployment server terminated successfully.")
