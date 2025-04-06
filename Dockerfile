# Wybierz oficjalny obraz Pythona
FROM python:3.9-slim

# Ustaw katalog roboczy w kontenerze
WORKDIR /app

# Skopiuj plik requirements.txt do kontenera
COPY requirements.txt .

# Zainstaluj wszystkie zależności z requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Skopiuj resztę plików aplikacji do kontenera
COPY . .

# Zdefiniuj domyślny port, na którym będzie działać aplikacja
EXPOSE 8080

# Uruchom aplikację Streamlit
CMD ["streamlit", "run", "app.py"]

