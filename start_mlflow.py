"""
Скрипт для запуска MLflow UI для проекта ML классификации
"""

import subprocess
import webbrowser
import time
import os

def start_mlflow_ui():
    """Запускает MLflow UI и открывает браузер"""
    
    
    project_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_dir)
    
    print("Запуск MLflow UI...")
    print(f"Рабочая директория: {project_dir}")
    
    
    try:
        process = subprocess.Popen(
            ["mlflow", "ui", "--backend-store-uri", "sqlite:///mlflow.db"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        
        time.sleep(3)
        
        
        webbrowser.open("http://localhost:5000")
        
        print("MLflow UI успешно запущен!")
        print("Откройте браузер и перейдите по адресу: http://localhost:5000")
        print("\nНажмите Ctrl+C для остановки сервера")
        
        
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\nОстановка MLflow UI...")
            process.terminate()
            
    except Exception as e:
        print(f"Ошибка запуска MLflow UI: {e}")
        print("\nУбедитесь, что MLflow установлен:")
        print("pip install mlflow")

if __name__ == "__main__":
    start_mlflow_ui()
