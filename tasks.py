# tasks.py
from celery import Celery
import subprocess

celery_app = Celery(
    'tasks',
    broker='redis://redis:6379/0',
    backend='redis://redis:6379/0'
)

@celery_app.task(bind=True)
def run_simulation_task(self, args_list):
    """
    Uruchamia symulację przez wywołanie skryptu 8.py z przekazanymi argumentami.
    Zbieramy dodatkowe informacje o postępie:
      - PROGRESS: X            (procentowy postęp)
      - INFO: Completed X/Y runs
      - INFO: Elapsed Xs, estimated remain Ys

    Linie zawierające "it/s" (TQDM) oraz "Symulacje:" są odfiltrowywane,
    dzięki czemu nie pojawią się w logach ani w Streamlit.
    """
    process = subprocess.Popen(
        ["python", "8.py"] + args_list,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )

    logs = ""
    # Słownik do przechowywania bieżących informacji o postępie
    progress_info = {
        'progress': 0,   # procentowy postęp
        'completed': 0,  # liczba wykonanych symulacji
        'total': 0,      # łączna liczba symulacji
        'elapsed': 0.0,  # czas, który już upłynął
        'remain': 0.0    # szacowany czas do końca
    }

    while True:
        line = process.stdout.readline()
        if not line:
            break

        # Odfiltrowujemy niechciane linie
        if "it/s" in line or "Symulacje:" in line:
            continue

        logs += line

        # PROGRESS: X
        if "PROGRESS:" in line:
            try:
                progress = int(line.split("PROGRESS:")[1].strip())
                progress_info['progress'] = progress
            except Exception:
                pass

        # INFO: Completed X/Y runs
        elif "INFO: Completed" in line and "runs" in line:
            try:
                part = line.split("INFO: Completed")[1].strip()  # "2/27 runs"
                part = part.replace("runs", "").strip()          # "2/27"
                done, total = part.split("/")
                progress_info['completed'] = int(done.strip())
                progress_info['total'] = int(total.strip())
            except Exception:
                pass

        # INFO: Elapsed Xs, estimated remain Ys
        elif "INFO: Elapsed" in line and "estimated remain" in line:
            try:
                part = line.split("INFO: Elapsed")[1].strip()  # "5.3s, estimated remain 150.2s"
                left, right = part.split(",")
                elapsed_val = left.replace("s", "").strip()     # "5.3"
                right = right.replace("estimated remain", "").strip()  # "150.2s"
                remain_val = right.replace("s", "").strip()     # "150.2"
                progress_info['elapsed'] = float(elapsed_val)
                progress_info['remain'] = float(remain_val)
            except Exception:
                pass

        # Aktualizujemy stan zadania w Celery (stan PROGRESS)
        self.update_state(state='PROGRESS', meta=progress_info)

    process.wait()
    return {
        'result': 'Symulacja zakończona',
        'logs': logs
    }

