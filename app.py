# -*- coding: utf-8 -*-
# Required imports
import streamlit as st
import os
import json
import datetime
import time
import pandas as pd
import altair as alt
import shutil
import logging
import yaml # For config.yaml
from yaml.loader import SafeLoader # Secure YAML loading
import streamlit_authenticator as stauth # For login
from streamlit_autorefresh import st_autorefresh
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, field_validator, ValidationInfo, ValidationError # Pydantic for config
import logging

# --- Constants ---
HISTORY_DIR = "history"
SAVED_SIM_FILE = "saved_simulations.json"
PRESETS_DIR = "presets" # Directory for config presets
USER_TASKS_FILE = "user_tasks.json" # Maps {username: task_id}
SIMULATION_CODE_FILE = "8.py" # Assumed name of the simulation script
RESULTS_CSV_FILE = "results.csv" # Temporary file for results
AUTH_CONFIG_FILE = 'config.yaml' # Authentication configuration file

# Ensure directories exist
for dir_path in [HISTORY_DIR, PRESETS_DIR]:
    if not os.path.exists(dir_path):
        try:
            os.makedirs(dir_path)
            logging.info(f"Created directory: {dir_path}")
        except Exception as e:
            # Use st.error only if Streamlit context is guaranteed, otherwise log
            logging.error(f"Failed to create directory '{dir_path}': {e}")
            # Consider if the app should stop here if directories are critical


# --- Basic Configuration & Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

st.set_page_config(
    page_title="Donation Game Simulation Pro",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Authenticator Config ---
# Poprawiony kod do wklejenia:

# Zdefiniuj placeholder *przed* blokiem try, jako stałą
DEFAULT_COOKIE_KEY_PLACEHOLDER = 'f06cc54620ca938b534a09db3ff690c8eb7fcbff71a3f6385d0451384aea9242'

# --- Load Authenticator Config ---
try:
    with open(AUTH_CONFIG_FILE) as file:
        auth_config = yaml.load(file, Loader=SafeLoader)

    # --- Dodany Debug Print (możesz go zostawić na razie) ---
    print(f"DEBUG app.py: Odczytany klucz cookie: >>>{auth_config.get('cookie', {}).get('key')}<<<")
    # ---------------------------------------------------------

    # Sprawdź, czy kluczowe sekcje istnieją
    if not auth_config or 'credentials' not in auth_config or 'cookie' not in auth_config:
         raise ValueError("Config file is missing required sections (credentials, cookie).")

    # Poprawione sprawdzanie Security Warning
    # Porównujemy odczytany klucz z *prawdziwym* placeholderem
    loaded_key = auth_config.get('cookie',{}).get('key')
    if not loaded_key or loaded_key == DEFAULT_COOKIE_KEY_PLACEHOLDER:
        # Klucz jest pusty LUB jest równy domyślnemu placeholderowi -> pokaż ostrzeżenie
        try:
              st.warning(f"Security Warning: Please change the default 'cookie: key:' in {AUTH_CONFIG_FILE}!")
        except Exception:
              # Loguj, jeśli nie można wyświetlić ostrzeżenia Streamlit (np. podczas startu)
              logging.warning(f"Security Warning: Please change the default 'cookie: key:' in {AUTH_CONFIG_FILE}!")
    # Jeśli klucz istnieje i NIE jest placeholderem, ostrzeżenie się nie pokaże.

except FileNotFoundError:
    logging.error(f"FATAL ERROR: {AUTH_CONFIG_FILE} not found. Authentication will be disabled.")
    auth_config = {'credentials': {'usernames': {}}, 'cookie': {'name':'dg_cookie','key':'dummy_key','expiry_days':1}, 'preauthorized': {}}
    try: st.error(f"FATAL ERROR: {AUTH_CONFIG_FILE} not found. Authentication disabled.")
    except Exception: pass
except Exception as e:
     logging.error(f"FATAL ERROR loading {AUTH_CONFIG_FILE}: {e}. Authentication disabled.")
     auth_config = {'credentials': {'usernames': {}}, 'cookie': {'name':'dg_cookie','key':'dummy_key','expiry_days':1}, 'preauthorized': {}}
     try: st.error(f"FATAL ERROR loading {AUTH_CONFIG_FILE}: {e}. Authentication disabled.")
     except Exception: pass

# --- Initialize 
# Poprawiony kod:
authenticator = stauth.Authenticate(
    auth_config.get('credentials', {'usernames':{}}),
    auth_config.get('cookie', {}).get('name', 'some_cookie_name'),
    auth_config.get('cookie', {}).get('key', 'some_signature_key'),
    auth_config.get('cookie', {}).get('expiry_days', 30), # <--- DODAJ TEN PRZECINEK
    auth_config.get('preauthorized', {})
)
# ... dalszy kod ...
# --- Import Celery (with fallback) ---
celery_available = False
try:
    from tasks import run_simulation_task, celery_app
    if hasattr(celery_app, 'AsyncResult'): # Basic check
        celery_available = True
        logging.info("Celery components imported successfully.")
    else:
         raise ImportError("Celery app object seems invalid.")
except ImportError as e:
    logging.error(f"Could not import Celery components from tasks.py: {e}")
    # Dummy implementations
    def run_simulation_task(*args, **kwargs): raise ImportError("Celery task not available")
    class DummyCeleryApp:
        class AsyncResult:
            def __init__(self, task_id): self.task_id = task_id; self.state = 'PENDING'; self.info = {}
            @property
            def traceback(self): return "Celery not available."
        class Control:
            def revoke(self, *args, **kwargs): pass
        control = Control()
    celery_app = DummyCeleryApp()

# --- Pydantic Model for Configuration ---
# Upewnij się, że masz te importy na górz

# Definicja klasy Pydantic
class SimulationConfig(BaseModel):
    model: str = Field(default="Bazowy", description="Wybrany model symulacji")
    agents: int = Field(default=100, gt=0, description="Liczba agentow (N)")
    pairs: int = Field(default=300, gt=0, description="Liczba par na generacje (M)")
    generations: int = Field(default=10, gt=0, description="Liczba generacji")
    runs: int = Field(default=1, gt=0, description="Liczba uruchomien")
    q_values_str: str = Field(default="0.8, 0.9, 1.0", alias="q_values", description="Wartosci q (oddzielone przecinkami)")
    noise_values_ea_ep: str = Field(default="0.0, 0.05, 0.1", description="Wartosci szumu dla modelu 'Z szumem'")
    generosity_choice: str = Field(default="g1", description="Wybor parametru dla 'Generosci'")
    g1_values: str = Field(default="0.0, 0.01, 0.02", description="Wartosci g1 dla 'Generosci'")
    g2_values: str = Field(default="0.0, 0.01, 0.02", description="Wartosci g2 dla 'Generosci'")
    noise_values: str = Field(default="0.0, 0.05, 0.1", description="Wartosci szumu dla 'Generosci'")

    # Validator dla modelu
    @field_validator('model')
    @classmethod
    def model_must_be_valid(cls, v):
        valid_models = ["Bazowy", "Z szumem", "Generosci", "Wybaczania"]
        if v not in valid_models:
            raise ValueError(f"Nieznany model: {v}.")
        return v

    # Poprawiony validator dla list float dla Pydantic v2
    @field_validator('q_values_str', 'noise_values_ea_ep', 'g1_values', 'g2_values', 'noise_values', mode='before')
    @classmethod
    def parse_comma_separated_floats(cls, v: Any, info: ValidationInfo) -> str:
        field_name = info.field_name # Nazwa sprawdzanego pola

        # Upewnij się, że wejście to string
        if not isinstance(v, str):
            v = str(v)

        # Pozwól na puste stringi dla pól nie wymaganych zawsze (innych niż q_values_str)
        if not v.strip() and field_name != 'q_values_str':
            return "" # Zwróć pusty string

        try:
            # Spróbuj sparsować wartości
            values = [float(x.strip()) for x in v.split(',') if x.strip()]

            # Sprawdź, czy lista nie jest pusta (jeśli nie powinna być)
            if not values:
                if field_name == 'q_values_str':
                    raise ValueError("Lista q nie moze byc pusta")
                else:
                     # Dla innych pól pusty string jest akceptowalny po walidacji
                     return ""

            # Walidacja formatu poprawna, zwróć oryginalny string (konwersja do listy float później)
            return v
        except ValueError as e:
            # Bardziej szczegółowy błąd dla problemów z konwersją na float
            if "could not convert string to float" in str(e):
                 raise ValueError(f"Niepoprawny format liczbowy w '{field_name}'. Uzyj liczb oddzielonych przecinkami.")
            else:
                 raise ValueError(str(e)) # Przekaż inne błędy ValueError (np. z pustej listy q)
        except Exception as e:
            # Złap inne nieoczekiwane błędy
            raise ValueError(f"Blad przetwarzania '{field_name}': {e}")

    def get_relevant_config_dict(self) -> Dict[str, Any]:
        """Zwraca słownik z polami relevantnymi dla wybranego modelu."""
        # Użyj model_dump dla Pydantic v2, dict dla v1
        dump_method = getattr(self, "model_dump", getattr(self, "dict", None))
        if dump_method is None: return {} # Na wszelki wypadek

        common_fields = {'model', 'agents', 'pairs', 'generations', 'runs', 'q_values_str'}
        # Użyj exclude=None, aby upewnić się, że alias 'q_values' nie powoduje problemów
        data = dump_method(include=common_fields, exclude=None)

        # Dodaj pola specyficzne dla modelu
        if self.model == "Z szumem":
            data.update(dump_method(include={'noise_values_ea_ep'}, exclude=None))
        elif self.model == "Generosci":
            data.update(dump_method(include={'generosity_choice', 'noise_values'}, exclude=None))
            if self.generosity_choice == 'g1':
                data.update(dump_method(include={'g1_values'}, exclude=None))
            else: # g2
                data.update(dump_method(include={'g2_values'}, exclude=None))
        # Dodaj inne modele, jeśli dostaną parametry
        return data

    def get_float_list(self, field_name_str: str) -> List[float]:
        """Konwertuje string z atrybutu na listę float, podnosi ValueError w razie błędu."""
        raw_str = getattr(self, field_name_str, "")
        if not raw_str.strip(): return []
        try:
            return [float(x.strip()) for x in raw_str.split(',') if x.strip()]
        except ValueError:
            # Ten błąd nie powinien się zdarzyć, jeśli validator działa poprawnie
            logging.error(f"Inconsistency: Failed to convert validated string '{raw_str}' in {field_name_str} to float list.")
            raise ValueError(f"Niepoprawny format liczbowy w {field_name_str}: '{raw_str}'")

# Koniec definicji klasy SimulationConfig
# --- CSS Styling ---
dark_css = """
<style>
/* Globalne ustawienia - ciemny motyw */
html, body { background-color: #121212; color: #FFFFFF; font-family: 'sans serif'; scroll-behavior: smooth; }
/* Karty */
.card { background-color: #1E1E1E; padding: 20px; border-radius: 10px; margin-bottom: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.4); transition: transform 0.2s ease; }
.card:hover { transform: translateY(-5px); box-shadow: 0 10px 15px rgba(0,0,0,0.5); }
/* Efekt fade-in dla elementow */
.fade-in { opacity: 0; animation: fadeIn 0.5s forwards; } @keyframes fadeIn { to { opacity: 1; } }
/* Naglowki w kolorze akcentu */
h1, h2, h3, h4 { color: #FF6600; }
/* Styl przyciskow Streamlit */
.stButton>button { background-color: #FF6600; color: #FFFFFF; border: none; border-radius: 8px; padding: 0.6rem 1rem; font-size: 16px; cursor: pointer; transition: background-color 0.2s ease; margin-top: 10px; }
.stButton>button:hover { background-color: #FFA347; }
/* Styl dla tabel */
table { background-color: #2A2A2A; border-collapse: collapse; }
thead tr { background-color: #333333; } tbody tr:nth-child(even) { background-color: #1F1F1F; }
td, th { padding: 0.5rem; border: 1px solid #444444; }
/* Styl dla separatora <hr> */
hr, .stMarkdown hr { border: none; border-top: 1px solid #444; margin: 0.5rem 0; }
/* Ustawienia kontenera aplikacji */
.reportview-container .main .block-container { max-width: 1200px; padding: 2rem; margin: auto; }
</style>
"""
st.markdown(dark_css, unsafe_allow_html=True)


# --- State Management ---
def initialize_session_state():
    # Initialize only if 'app_initialized' flag is not set
    if 'app_initialized' not in st.session_state:
        logging.info("Initializing session state for the first time.")
        default_state = {
            "authentication_status": None, "name": None, "username": None,
            "simulation_started": False, "simulation_finished": False, "cancel_confirmation": False,
            "current_config": SimulationConfig().model_dump(), # Store config as dict
            "task_id": None, # Task ID for the *current user* in *this session*
            "df": None, "chart": None, "history_saved": False,
            "last_progress_info": None, "last_task_state": None,
            "saved_simulations": {}, "saved_simulations_loaded": False,
            "selected_preset": None,
            "compare_slot_1": None, "compare_slot_2": None,
            "app_initialized": True # Flag to prevent re-initialization on reruns
        }
        for key, default_value in default_state.items():
            if key not in st.session_state:
                st.session_state[key] = default_value

        # Load saved simulations only once per browser session start
        if not st.session_state.saved_simulations_loaded:
            loaded_sims = load_json(SAVED_SIM_FILE)
            st.session_state.saved_simulations = loaded_sims if isinstance(loaded_sims, dict) else {}
            st.session_state.saved_simulations_loaded = True
            logging.info(f"Loaded {len(st.session_state.saved_simulations)} saved simulation slots.")


# --- File I/O Functions ---
def save_json(filepath, data):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logging.info(f"Saved data to {filepath}")
        return True
    except Exception as e:
        logging.error(f"Error saving {filepath}: {e}")
        return False

def load_json(filepath):
    if os.path.exists(filepath):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()
                if not content.strip():
                    logging.warning(f"File {filepath} is empty.")
                    return None
                return json.loads(content)
        except json.JSONDecodeError:
            logging.warning(f"File {filepath} is corrupted or not valid JSON. Returning None.")
            return None
        except Exception as e:
            logging.error(f"Error loading {filepath}: {e}")
            return None
    return None # File doesn't exist

def remove_file(filepath):
     if os.path.exists(filepath):
        try:
            os.remove(filepath)
            logging.info(f"Removed file: {filepath}")
            return True
        except Exception as e:
            logging.error(f"Error removing {filepath}: {e}")
            return False
     return False # File didn't exist

# --- User-Task Mapping Functions ---
def load_user_tasks() -> Dict[str, str]:
    # Ensure file exists before loading
    if not os.path.exists(USER_TASKS_FILE):
        if save_json(USER_TASKS_FILE, {}): # Create empty file if missing
             logging.info(f"Created empty user tasks file: {USER_TASKS_FILE}")
             return {}
        else:
             logging.error(f"Failed to create user tasks file: {USER_TASKS_FILE}")
             return {} # Return empty dict on error
    data = load_json(USER_TASKS_FILE)
    return data if isinstance(data, dict) else {}

def save_user_tasks(user_tasks: Dict[str, str]):
    save_json(USER_TASKS_FILE, user_tasks)

def get_user_task_id(username: str) -> Optional[str]:
    if not username: return None
    user_tasks = load_user_tasks()
    return user_tasks.get(username)

def set_user_task_id(username: str, task_id: Optional[str]):
    if not username:
        logging.error("Cannot set task ID for empty username.")
        return
    user_tasks = load_user_tasks() # Ensure load_user_tasks handles empty/missing file
    if task_id:
        user_tasks[username] = task_id
        logging.info(f"Assigned task {task_id} to user {username}")
    else:
        if username in user_tasks:
            removed_task_id = user_tasks.pop(username, None) # Use pop with default None
            if removed_task_id:
                 logging.info(f"Removed task mapping for task {removed_task_id} for user {username}")
        # No need to log if user wasn't in mapping
    save_user_tasks(user_tasks)

# --- History Management ---
@st.cache_data(ttl=300) # Cache history list for 5 minutes
def load_history():
    records = []
    if not os.path.exists(HISTORY_DIR):
        try: os.makedirs(HISTORY_DIR)
        except Exception as e: logging.error(f"Cannot create history dir {HISTORY_DIR}: {e}"); return []

    try:
        dir_list = sorted([d for d in os.listdir(HISTORY_DIR) if os.path.isdir(os.path.join(HISTORY_DIR, d))], reverse=True)
        for folder in dir_list:
            metadata = load_json(os.path.join(HISTORY_DIR, folder, "metadata.json"))
            if metadata:
                metadata["record_id"] = folder
                records.append(metadata)
    except Exception as e:
         logging.error(f"Error loading history directory: {e}")
         # Avoid showing error in UI repeatedly due to cache?
    return records

def clear_history_cache():
    load_history.clear()
    logging.info("Cleared history cache.")

def save_history(sim_config):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    record_dir = os.path.join(HISTORY_DIR, timestamp)
    try:
        os.makedirs(record_dir, exist_ok=True)
        # Save metadata (ensure it's serializable)
        config_dict = sim_config if isinstance(sim_config, dict) else sim_config.dict()
        save_json(os.path.join(record_dir, "metadata.json"), config_dict)
        # Copy results and code if they exist
        results_copied = False
        if os.path.exists(RESULTS_CSV_FILE):
            try: shutil.copy2(RESULTS_CSV_FILE, os.path.join(record_dir, "results.csv")); results_copied = True
            except Exception as e_csv: logging.warning(f"Could not copy {RESULTS_CSV_FILE} to history: {e_csv}")
        code_copied = False
        if os.path.exists(SIMULATION_CODE_FILE):
            try: shutil.copy2(SIMULATION_CODE_FILE, os.path.join(record_dir, os.path.basename(SIMULATION_CODE_FILE))); code_copied = True
            except Exception as e_code: logging.warning(f"Could not copy {SIMULATION_CODE_FILE} to history: {e_code}")

        logging.info(f"History saved to {record_dir} (Results copied: {results_copied}, Code copied: {code_copied})")
        clear_history_cache() # Invalidate cache after adding new record
        return record_dir
    except Exception as e:
        logging.error(f"Error saving history to {record_dir}: {e}")
        st.error(f"Error saving history: {e}")
        return None

def delete_history_record(record_id):
    record_dir = os.path.join(HISTORY_DIR, record_id)
    if os.path.exists(record_dir) and os.path.isdir(record_dir):
        try:
            shutil.rmtree(record_dir)
            logging.info(f"Deleted history record {record_id}")
            clear_history_cache() # Invalidate cache
            # Also remove from saved simulations if it was saved there
            saved_simulations = st.session_state.get("saved_simulations", {})
            slot_to_remove = None
            for slot, saved_info in saved_simulations.items():
                 if saved_info.get("record_id") == record_id:
                      slot_to_remove = slot
                      break
            if slot_to_remove:
                 logging.warning(f"Removing deleted history record {record_id} from saved slot '{slot_to_remove}'")
                 st.session_state.saved_simulations.pop(slot_to_remove)
                 save_json(SAVED_SIM_FILE, st.session_state.saved_simulations) # Save updated slots
            return True
        except Exception as e:
            logging.error(f"Could not delete history record {record_id}: {e}")
            st.error(f"Could not delete history record: {e}")
            return False
    else:
        logging.warning(f"History record {record_id} not found or is not a directory.")
        return False

# --- Parameter Presets Functions ---
def list_presets():
    if not os.path.exists(PRESETS_DIR): return []
    try:
        return sorted([f.replace('.json', '') for f in os.listdir(PRESETS_DIR) if f.endswith('.json')])
    except Exception as e:
        logging.error(f"Error listing presets: {e}"); return []

def save_preset(name: str, config: Dict[str, Any]):
    if not name.strip():
        st.error("Nazwa presetu nie moze byc pusta."); return False
    safe_name = "".join(c if c.isalnum() or c in ('-', '_') else '_' for c in name.replace(' ', '_')).strip('_')
    if not safe_name:
        st.error("Niepoprawna nazwa presetu."); return False
    filepath = os.path.join(PRESETS_DIR, f"{safe_name}.json")
    try:
        config_model = SimulationConfig(**config)
        config_to_save = config_model.get_relevant_config_dict()
    except ValidationError as e:
         st.error(f"Blad walidacji konfiguracji presetu:\n{e}")
         logging.error(f"Preset config validation error for '{safe_name}': {e}")
         return False
    except Exception as e:
        st.error(f"Blad przy walidacji presetu: {e}")
        logging.error(f"Unexpected preset validation error for '{safe_name}': {e}")
        return False

    if save_json(filepath, config_to_save):
        st.success(f"Preset '{safe_name}' zapisany.")
        return True
    else:
         st.error(f"Nie udalo sie zapisac presetu '{safe_name}'.")
         return False

def load_preset(name: str) -> Optional[Dict[str, Any]]:
    if not name: return None
    filepath = os.path.join(PRESETS_DIR, f"{name}.json")
    config = load_json(filepath)
    if config:
        st.success(f"Preset '{name}' zaladowany.")
        st.session_state.selected_preset = name
    else:
        st.error(f"Nie udalo sie zaladowac presetu '{name}'.")
    return config

def delete_preset(name: str):
    if not name: return False
    filepath = os.path.join(PRESETS_DIR, f"{name}.json")
    preset_deleted = False
    if os.path.exists(filepath):
        if remove_file(filepath):
            st.success(f"Preset '{name}' usuniety.")
            preset_deleted = True
        else:
             st.error(f"Nie udalo sie usunac presetu '{name}'.")
    else:
        st.warning(f"Preset '{name}' nie znaleziony.")
    # Clear selection if the deleted preset was selected
    if preset_deleted and st.session_state.get("selected_preset") == name:
        st.session_state.selected_preset = None
    return preset_deleted


# --- Plotting Functions ---
def generate_plot_for_main(df: pd.DataFrame, model: str, extra_config: dict) -> alt.Chart:
    """Generates the main results plot."""
    try:
        # Determine plot title and encodings based on model
        if model == "Z szumem":
            if not all(col in df.columns for col in ['ea', 'avg', 'q']): raise ValueError("Missing columns for Z szumem plot")
            x_encoding = alt.X("ea:Q", scale=alt.Scale(zero=False), title="Szum (ea=ep)")
            tooltip = ['q', 'ea', 'ep', 'avg', 'std']
            plot_title = "Wykres - Model Z szumem (po symulacji)"
            color_encoding = alt.Color("q:N", title="q")
        elif model == "Generosci":
            gc = extra_config.get("generosity_choice", "g1")
            if not all(col in df.columns for col in [gc, 'ea', 'avg']): raise ValueError(f"Missing columns for Generosci ({gc}) plot")
            x_encoding = alt.X("ea:Q", scale=alt.Scale(zero=False), title="ea")
            tooltip = [gc, 'q', 'ea', 'ep', 'avg', 'std']
            plot_title = f"Wykres - Model Generosci ({gc}) (po symulacji)"
            color_encoding = alt.Color(f"{gc}:N", title=gc)
        else: # Bazowy, Wybaczania, etc.
            if 'avg' not in df.columns: raise ValueError("Missing 'avg' column for default plot")
            # Assuming index makes sense for these models, adjust if needed
            df = df.reset_index()
            x_encoding = alt.X("index:Q", title="Index")
            tooltip = ['index', 'q', 'avg', 'std'] if 'q' in df.columns else ['index', 'avg', 'std']
            plot_title = f"Wykres - Model {model} (po symulacji)"
            # No color encoding by default for base/other models unless specified
            color_encoding = alt.value('steelblue') # Default color

        # Base chart
        chart = alt.Chart(df).mark_line(point=True, strokeDash=[5, 5] if model in ["Z szumem", "Generosci"] else []).encode(
            x=x_encoding,
            y=alt.Y("avg:Q", scale=alt.Scale(zero=False), title="avg"),
            color=color_encoding,
            tooltip=tooltip
        ).properties(title=plot_title)

        # Common styling
        return chart.configure_axis(
            grid=True, gridColor='#444', gridOpacity=0.3, domainColor='#666',
            tickColor='#666', labelColor='white', titleColor='white'
        ).configure_view(strokeWidth=0).interactive()

    except Exception as e:
        st.error(f"Error generating main plot: {e}")
        return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_point().properties(title="Plot Error")

def generate_plot_history(df: pd.DataFrame, model: str, extra_config: dict) -> alt.Chart:
    """Generates the plot for the history view."""
    try:
        # Similar logic to main plot, adjust titles
        if model == "Z szumem":
            if not all(col in df.columns for col in ['ea', 'avg', 'q']): raise ValueError("Missing columns for Z szumem history plot")
            x_encoding = alt.X("ea:Q", scale=alt.Scale(zero=False), title="Szum (ea=ep)")
            tooltip = ['q', 'ea', 'ep', 'avg', 'std']
            plot_title = "Wykres - Z szumem (z historii)"
            color_encoding = alt.Color("q:N", title="q")
        elif model == "Generosci":
            gc = extra_config.get("generosity_choice", "g1")
            if not all(col in df.columns for col in [gc, 'ea', 'avg']): raise ValueError(f"Missing columns for Generosci ({gc}) history plot")
            x_encoding = alt.X("ea:Q", scale=alt.Scale(zero=False), title="ea")
            tooltip = [gc, 'q', 'ea', 'ep', 'avg', 'std']
            plot_title = f"Wykres - Generosci ({gc}) (z historii)"
            color_encoding = alt.Color(f"{gc}:N", title=gc)
        else:
            if 'avg' not in df.columns: raise ValueError("Missing 'avg' column for default history plot")
            df = df.reset_index()
            x_encoding = alt.X("index:Q", title="Index")
            tooltip = ['index', 'q', 'avg', 'std'] if 'q' in df.columns else ['index', 'avg', 'std']
            plot_title = f"Wykres - Model {model} (z historii)"
            color_encoding = alt.value('steelblue')

        chart = alt.Chart(df).mark_line(point=True, strokeDash=[5, 5] if model in ["Z szumem", "Generosci"] else []).encode(
            x=x_encoding,
            y=alt.Y("avg:Q", scale=alt.Scale(zero=False), title="avg"),
            color=color_encoding,
            tooltip=tooltip
        ).properties(title=plot_title)

        return chart.configure_axis(
            grid=True, gridColor='#444', gridOpacity=0.3, domainColor='#666',
            tickColor='#666', labelColor='white', titleColor='white'
        ).configure_view(strokeWidth=0).interactive()

    except Exception as e:
        st.error(f"Error generating history plot: {e}")
        return alt.Chart(pd.DataFrame({'x': [], 'y': []})).mark_point().properties(title="Plot Error")


# --- Simulation Logic ---
def start_simulation(config_dict: Dict[str, Any], username: str):
    """Validates config, prepares args, checks policy, and starts Celery task."""
    existing_task_id = get_user_task_id(username)
    if existing_task_id:
        st.error(f"Masz juz przypisana symulacje (ID: {existing_task_id[:8]}...). Poczekaj lub ja przerwij.")
        return False

    try: # Validate config via Pydantic
        sim_config = SimulationConfig(**config_dict)
        logging.info(f"User {username} validated config: {sim_config.model_dump_json(indent=2)}")
    except ValidationError as e:
        st.error(f"Blad konfiguracji:\n{e}"); logging.error(f"Pydantic validation failed user {username}: {e}"); return False
    except Exception as e:
        st.error(f"Blad przetwarzania konfiguracji: {e}"); logging.error(f"Config processing error user {username}: {e}", exc_info=True); return False

    try: # Prepare args list
        args_list = ["--size", str(sim_config.agents), "--pairs", str(sim_config.pairs), "--generations", str(sim_config.generations), "--runs", str(sim_config.runs), "--output", RESULTS_CSV_FILE, "--q_values"] # Add output redirection
        q_list = sim_config.get_float_list('q_values_str');
        if not q_list: raise ValueError("Lista q pusta")
        args_list.extend(map(str, q_list))

        model_name = sim_config.model
        if model_name == "Z szumem":
            args_list.extend(["--model", "1"]); noise_list = sim_config.get_float_list('noise_values_ea_ep');
            if not noise_list: raise ValueError("Lista szumu pusta")
            args_list.extend(["--ea_values"] + list(map(str, noise_list))); args_list.extend(["--ep_values"] + list(map(str, noise_list)))
        elif model_name == "Generosci":
            args_list.extend(["--model", "2", "--generosity"]); noise_g_list = sim_config.get_float_list('noise_values');
            if not noise_g_list: raise ValueError("Lista szumu dla G pusta")
            args_list.extend(["--ea_values"] + list(map(str, noise_g_list))); args_list.extend(["--ep_values"] + list(map(str, noise_g_list)))
            if sim_config.generosity_choice == "g1": g1_list = sim_config.get_float_list('g1_values'); args_list.extend(["-g1"] + list(map(str, g1_list)))
            else: g2_list = sim_config.get_float_list('g2_values'); args_list.extend(["-g2"] + list(map(str, g2_list)))
        elif model_name == "Wybaczania": args_list.extend(["--model", "3", "--forgiveness_action", "--forgiveness_reputation"])
        else: args_list.extend(["--model", "0"])

    except ValueError as e: st.error(f"Blad argumentow: {e}"); logging.error(f"Arg prep error user {username}: {e}"); return False
    except Exception as e: st.error(f"Blad argumentow: {e}"); logging.error(f"Unexpected arg prep error user {username}: {e}", exc_info=True); return False

    # Update state (resetting relevant parts)
    st.session_state.update({
        "current_config": sim_config.dict(), "simulation_started": True, "simulation_finished": False,
        "df": None, "chart": None, "history_saved": False, "task_id": None,
        "last_progress_info": None, "last_task_state": None, "cancel_confirmation": False
    })

    try: # Launch Celery task
        if not celery_available: raise RuntimeError("Celery backend jest niedostepny.")
        # Clear previous results file if it exists
        remove_file(RESULTS_CSV_FILE)

        logging.info(f"User {username} starting sim task args: {' '.join(args_list)}")
        task = run_simulation_task.delay(args_list);
        st.session_state.task_id = task.id;
        set_user_task_id(username, task.id); # Map user to task

        st.success(f"Symulacja uruchomiona, ID: {task.id}")
        logging.info(f"Sim task started user {username} ID: {task.id}")
        st.rerun(); return True # Rerun to show progress view
    except Exception as e:
        logging.error(f"Celery task launch failed user {username}: {e}", exc_info=True)
        st.error(f"Nie udalo sie uruchomic zadania Celery: {e}")
        st.session_state.simulation_started = False; set_user_task_id(username, None); # Rollback
        return False

def cancel_simulation_action(username: str):
    """Handles the cancellation process after confirmation for the logged-in user."""
    task_id = get_user_task_id(username)
    if task_id:
        try:
            logging.warning(f"User {username} cancelling task {task_id}")
            celery_app.control.revoke(task_id, terminate=True, signal='SIGTERM')
            st.info(f"Wyslano sygnal przerwania {task_id}")
        except Exception as e:
            logging.error(f"Revoke error user {username} task {task_id}: {e}")
            st.error(f"Blad sygnalu przerwania: {e}")
        finally:
            set_user_task_id(username, None) # Always remove mapping on cancel attempt
    else:
        st.warning("Brak aktywnej symulacji do przerwania.")

    # Reset session state
    st.session_state.update({
        "simulation_started": False, "simulation_finished": True, "task_id": None,
        "df": None, "chart": None, "history_saved": False,
        "last_progress_info": None, "last_task_state": None, "cancel_confirmation": False
    })
    st.rerun()

def check_simulation_status(progress_placeholder, username: str):
    """Checks Celery task status for the logged-in user and updates UI."""
    task_id = get_user_task_id(username)
    if not task_id:
        # If this session thinks sim is running but mapping is gone, reset session state
        if st.session_state.simulation_started:
            logging.warning(f"User {username} session state/task mapping mismatch. Resetting session.")
            st.session_state.update({
                "simulation_started": False, "simulation_finished": True, "task_id": None,
                "last_task_state": None, "last_progress_info": None
            })
            progress_placeholder.info("Twoja poprzednia symulacja zakonczyla sie lub zostala przerwana.")
        return # No active task for this user

    # Ensure session state knows the correct task ID
    if st.session_state.task_id != task_id:
         st.session_state.update({"task_id": task_id, "simulation_started": True, "simulation_finished": False})

    try:
        task = celery_app.AsyncResult(task_id)
        current_progress_info = task.info or {}
        current_state = task.state

        state_changed = current_state != st.session_state.last_task_state
        info_changed = current_progress_info != st.session_state.last_progress_info

        if state_changed or info_changed:
            logging.info(f"Task {task_id} (User: {username}) state/info changed: {current_state}")
            st.session_state.last_task_state = current_state
            st.session_state.last_progress_info = current_progress_info

            with progress_placeholder.container(): # Use container to clear previous content
                if current_state == 'PENDING': st.info(f"Zadanie {task_id[:8]}... oczekuje...")
                elif current_state == 'STARTED': st.info(f"Zadanie {task_id[:8]}... uruchomione.")
                elif current_state == 'PROGRESS':
                    progress = current_progress_info.get('progress', 0)
                    completed = current_progress_info.get('completed', 0)
                    total = current_progress_info.get('total', 1)
                    elapsed = current_progress_info.get('elapsed', 0.0)
                    remain = current_progress_info.get('remain', 0.0)
                    progress_text = f"**Postep:** {progress:.1f}% ({completed}/{total}) | **Czas:** {elapsed:.1f}s | **Pozostalo:** {remain:.1f}s"
                    st.markdown(progress_text)
                    st.progress(progress / 100.0)

                elif current_state == 'SUCCESS':
                    st.success(f"Symulacja ({task_id[:8]}...) zakonczona!")
                    set_user_task_id(username, None) # Clear mapping on success
                    st.session_state.simulation_finished = True

                    # Process results only once per session after finish
                    if not st.session_state.df and not st.session_state.chart:
                         logging.info(f"Processing results for task {task_id} (User: {username})")
                         if os.path.exists(RESULTS_CSV_FILE):
                             try:
                                 df = pd.read_csv(RESULTS_CSV_FILE, encoding='utf-8')
                                 if not df.empty:
                                     st.session_state.df = df
                                     extra_config = {}
                                     sim_cfg = st.session_state.get("current_config", {}) # Use current session's config
                                     if sim_cfg.get("model") == "Generosci": extra_config["generosity_choice"] = sim_cfg.get("generosity_choice")
                                     st.session_state.chart = generate_plot_for_main(df, sim_cfg.get("model", "Bazowy"), extra_config)

                                     # Save history (only if not already saved in this session)
                                     if not st.session_state.history_saved:
                                         sim_config_to_save = {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "user": username, **sim_cfg}
                                         # Ensure 'params' key exists for compatibility if needed by history view
                                         sim_config_to_save.setdefault("params", sim_cfg)
                                         sim_config_to_save["extra_config"] = extra_config # Save plot config too
                                         save_history(sim_config_to_save)
                                         st.session_state.history_saved = True
                                 else:
                                     logging.warning(f"{RESULTS_CSV_FILE} is empty for task {task_id}.")
                                     st.warning(f"{RESULTS_CSV_FILE} jest pusty.")
                             except Exception as e:
                                 logging.error(f"Error processing {RESULTS_CSV_FILE} for task {task_id}: {e}")
                                 st.error(f"Blad przetwarzania {RESULTS_CSV_FILE}: {e}")
                         else:
                             logging.warning(f"{RESULTS_CSV_FILE} not found for task {task_id}.")
                             st.warning(f"Brak pliku {RESULTS_CSV_FILE}.")

                    # Rerun outside placeholder to display results in main area
                    st.rerun()

                elif current_state == 'FAILURE':
                    st.error(f"Blad symulacji ({task_id[:8]}...).")
                    set_user_task_id(username, None) # Clear mapping on failure
                    logging.error(f"Task {task_id} (User: {username}) failed.\n{task.traceback}")
                    with st.expander("Pokaz Traceback", expanded=False):
                        st.text(task.traceback or "Brak dostepnego tracebacku.")
                    st.session_state.simulation_finished = True

                elif current_state == 'REVOKED':
                     st.warning(f"Symulacja ({task_id[:8]}...) przerwana.")
                     set_user_task_id(username, None) # Should be cleared by cancel, but ensure here too
                     st.session_state.simulation_finished = True
                else:
                    st.info(f"Status zadania {task_id[:8]}...: {current_state}")

    except Exception as e:
        logging.error(f"Error check status task {task_id} user {username}: {e}", exc_info=True)
        # Avoid overwriting placeholder if simulation already marked finished in session
        if not st.session_state.get("simulation_finished", False):
             progress_placeholder.error(f"Blad sprawdzania statusu: {e}")

def render_config_form(username: str):
    """Renderuje formularz konfiguracji symulacji (bez st.form dla dynamicznego UI)."""
    st.header("Konfiguracja Symulacji") # Simulation Configuration

    # Odczytaj bieżącą konfigurację ze stanu sesji do ustawienia wartości domyślnych widgetów
    current_config_dict = st.session_state.get("current_config", SimulationConfig().dict())

    # --- Sekcja Presetów (Nowy Układ) ---
    st.markdown("---")
    st.subheader("Presety Konfiguracji") # Configuration Presets

    # Utwórz dwie główne kolumny
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("##### Ladowanie Presetu") # Load Preset
        try:
            preset_list = ["(Wybierz preset)"] + list_presets() # (Select preset)
            # Obsługa sytuacji, gdy wybrany preset został usunięty
            current_selection = st.session_state.get("selected_preset")
            preset_index = preset_list.index(current_selection) if current_selection in preset_list else 0
            selected_preset_name = st.selectbox(
                "Zaladuj preset:", # Load preset:
                preset_list,
                key="preset_load_select",
                index=preset_index,
                label_visibility="collapsed" # Ukryj etykietę, bo jest w markdown
            )
        except Exception as e:
            logging.error(f"Error loading presets list: {e}")
            st.error("Blad ladowania listy presetow.")
            selected_preset_name = "(Wybierz preset)" # Fallback

        if st.button("Zaladuj Wybrany", key="load_preset_btn", use_container_width=True): # Load Selected
            if selected_preset_name != "(Wybierz preset)":
                loaded_cfg = load_preset(selected_preset_name)
                if loaded_cfg:
                    # Upewnij się, że wszystkie klucze istnieją, używając domyślnego modelu jako bazy
                    base_cfg = SimulationConfig().dict()
                    base_cfg.update(loaded_cfg) # Nadpisz wartościami z presetu
                    st.session_state.current_config = base_cfg
                    # Zaktualizuj stan widgetów, aby odzwierciedlić załadowane wartości
                    # UWAGA: Bezpośrednie ustawianie st.session_state[widget_key] może być
                    # zastąpione przez rerun, który odczyta nowe wartości z current_config.
                    st.rerun() # Wymuś rerun, aby formularz się zaktualizował
            else:
                st.warning("Wybierz preset do zaladowania.") # Select a preset to load.

        # Przycisk usuwania presetu - widoczny tylko gdy coś jest wybrane
        if selected_preset_name != "(Wybierz preset)":
            st.markdown("") # Dodaj trochę przestrzeni
            if st.button(f"Usun Preset '{selected_preset_name}'", key="delete_preset_btn", type="secondary", use_container_width=True): # Delete Preset '...'
                  if delete_preset(selected_preset_name):
                       st.rerun() # Odśwież, aby zaktualizować listę i formularz

    with col_right:
        st.markdown("##### Zapisywanie Presetu") # Save Preset
        preset_save_name = st.text_input(
            "Zapisz jako preset:", # Save as preset:
            key="preset_save_name",
            placeholder="Nazwa presetu...", # Preset name...
            label_visibility="collapsed"
        )
        if st.button("Zapisz Aktualna Konfiguracje", key="save_preset_btn", use_container_width=True): # Save Current Configuration
            # Zbierz aktualne wartości z widgetów, aby zapisać jako preset
            current_widget_config = {
                 "model": st.session_state.get("model_select_form", "Bazowy"),
                 "agents": st.session_state.get("agents_form", 100),
                 "pairs": st.session_state.get("pairs_form", 300),
                 "generations": st.session_state.get("generations_form", 10),
                 "runs": st.session_state.get("runs_form", 1),
                 "q_values_str": st.session_state.get("q_values_form", "0.8, 0.9, 1.0"),
                 "noise_values_ea_ep": st.session_state.get("noise_values_ea_ep_form", "0.0, 0.05, 0.1"),
                 "generosity_choice": st.session_state.get("generosity_choice_form", "g1"),
                 "g1_values": st.session_state.get("g1_values_form", "0.0, 0.01, 0.02"),
                 "g2_values": st.session_state.get("g2_values_form", "0.0, 0.01, 0.02"),
                 "noise_values": st.session_state.get("noise_values_form", "0.0, 0.05, 0.1"),
            }
            save_preset(preset_save_name, current_widget_config)
            # Wyczyszczenie pola po zapisie - trudne bez reruna, zostawiamy na razie
            # st.session_state.preset_save_name = "" # Może nie zadziałać

    st.markdown("---")
    # --- Koniec Sekcji Presetów ---


    # --- Główna Konfiguracja (bez form) ---
    # Odczytaj domyślne wartości z current_config (zaktualizowanego ewentualnie przez preset)
    current_config_dict = st.session_state.get("current_config", SimulationConfig().dict())

    # Model Selection
    model_index = ["Bazowy", "Z szumem", "Generosci", "Wybaczania"].index(current_config_dict.get("model", "Bazowy"))
    model_choice = st.selectbox(
        "Model:", ["Bazowy", "Z szumem", "Generosci", "Wybaczania"],
        index=model_index,
        key="model_select_form", # Unikalny klucz widgetu
        help="Wybierz model teoretyczny."
    )

    # General Parameters
    st.markdown("#### Parametry Ogolne")
    c1, c2 = st.columns(2)
    with c1:
        st.number_input("Agenci (N)", value=current_config_dict.get("agents", 100), step=10, min_value=10, key="agents_form", help="Liczba agentow")
        st.number_input("Pary/Gen (M)", value=current_config_dict.get("pairs", 300), step=10, min_value=1, key="pairs_form", help="Liczba interakcji")
        st.number_input("Generacje", value=current_config_dict.get("generations", 10), step=1, min_value=1, key="generations_form", help="Liczba generacji")
    with c2:
        st.number_input("Uruchomienia", value=current_config_dict.get("runs", 1), step=1, min_value=1, key="runs_form", help="Liczba przebiegow")
        st.text_input("Wartosci q (przecinki)", value=current_config_dict.get("q_values_str", "0.8, 0.9, 1.0"), key="q_values_form", help="Prawdopodobienstwo q")

    # --- Model-Specific Parameters ---
    st.markdown("---"); st.markdown("#### Parametry Specyficzne")

    # Logika warunkowa teraz zadziała poprawnie po zmianie `model_choice` i rerunie Streamlit
    if model_choice == "Z szumem":
        st.text_input(
            "Szum ea/ep (przecinki)",
            value=current_config_dict.get("noise_values_ea_ep", "0.0, 0.05, 0.1"),
            key="noise_values_ea_ep_form", # Unikalny klucz
            help="ea=ep"
        )
    elif model_choice == "Generosci":
        # Użyj wartości z widgetu radio, jeśli istnieje w stanie sesji, inaczej z configu
        gen_choice_index = ["g1", "g2"].index(st.session_state.get("generosity_choice_form", current_config_dict.get("generosity_choice", "g1")))
        st.radio(
            "Parametr G:", ["g1", "g2"],
            index=gen_choice_index,
            key="generosity_choice_form", # Unikalny klucz
            horizontal=True
        )
        # Dynamicznie pokazuj odpowiednie pole g1/g2 na podstawie *aktualnej* wartości radio
        current_gen_choice = st.session_state.generosity_choice_form # Odczytaj aktualny stan radio
        if current_gen_choice == "g1":
            st.text_input(
                "Wartosci g1 (przecinki)",
                value=current_config_dict.get("g1_values", "0.0, 0.01, 0.02"),
                key="g1_values_form", # Unikalny klucz
                help="g1"
            )
        else: # g2
            st.text_input(
                "Wartosci g2 (przecinki)",
                value=current_config_dict.get("g2_values", "0.0, 0.01, 0.02"),
                key="g2_values_form", # Unikalny klucz
                help="g2"
            )
        # Pole na szum dla modelu Generosnosc jest zawsze widoczne, gdy ten model jest wybrany
        st.text_input(
            "Szum ea/ep dla G (przecinki)",
            value=current_config_dict.get("noise_values", "0.0, 0.05, 0.1"),
            key="noise_values_form", # Unikalny klucz
            help="ea/ep dla G"
        )
    elif model_choice == "Wybaczania":
        st.caption("Model Wybaczania: Brak parametrow UI.")
    elif model_choice == "Bazowy":
        st.caption("Model Bazowy: Brak dodatkowych parametrow.")

    # --- Przycisk Uruchomienia (bez form) ---
    st.markdown("---")
    submitted = st.button(
        "Uruchom Symulacje",
        type="primary",
        disabled=not celery_available,
        key="start_button" # Użyj unikalnego klucza
    )
    if submitted:
        if not celery_available:
            st.error("Backend Celery jest niedostepny."); # Zatrzymaj, jeśli Celery nie działa

        else:
            logging.info(f"Start button clicked by user {username}.")
            # Zbierz aktualne wartości z widgetów (stanu sesji)
            config_to_run = {
                "model": st.session_state.model_select_form,
                "agents": st.session_state.agents_form,
                "pairs": st.session_state.pairs_form,
                "generations": st.session_state.generations_form,
                "runs": st.session_state.runs_form,
                "q_values_str": st.session_state.q_values_form,
                # Bezpiecznie pobierz wartości specyficzne dla modelu, używając .get
                "noise_values_ea_ep": st.session_state.get("noise_values_ea_ep_form", ""),
                "generosity_choice": st.session_state.get("generosity_choice_form", "g1"),
                "g1_values": st.session_state.get("g1_values_form", ""),
                "g2_values": st.session_state.get("g2_values_form", ""),
                "noise_values": st.session_state.get("noise_values_form", ""),
            }
            # Zaktualizuj główny config w stanie sesji przed uruchomieniem
            st.session_state.current_config = config_to_run
            # Wywołaj start_simulation z zebraną konfiguracją i nazwą użytkownika
            start_simulation(config_to_run, username)
            # Rerun jest wywoływany wewnątrz start_simulation w razie sukcesu

def render_simulation_status_view(username: str):
    """Renders the view for monitoring the logged-in user's simulation."""
    st.header("Aktualna Symulacja")
    config = st.session_state.get("current_config", {})
    with st.expander("Parametry Uruchomionej Symulacji", expanded=False):
         if config:
             try:
                 display_config = SimulationConfig(**config)
                 st.table(pd.DataFrame(list(display_config.get_relevant_config_dict().items()), columns=['Parametr', 'Wartosc']))
             except Exception: # Fallback to raw dict display
                 st.table(pd.DataFrame(list(config.items()), columns=['Parametr', 'Wartosc']))
         else: st.write("Brak parametrow.")

    progress_placeholder = st.empty() # Placeholder for status updates

    if not st.session_state.simulation_finished:
        with progress_placeholder.container(): st.info("Ladowanie statusu...") # Initial message
        # Auto-refresh to check status
        st_autorefresh(interval=15000, limit=None, key="simulation_refresh") # Check every 5s
        check_simulation_status(progress_placeholder, username) # Pass username

        # Cancel Button & Confirmation Logic
        st.markdown("---")
        if st.button("Przerwij Symulacje", key="cancel_sim_btn", disabled=not celery_available):
            st.session_state.cancel_confirmation = True; st.rerun()

        if st.session_state.get("cancel_confirmation", False):
            st.warning("Czy na pewno chcesz przerwac?")
            col1, col2 = st.columns(2)
            if col1.button("Tak, przerwij", key="confirm_cancel_btn"): cancel_simulation_action(username) # Pass username
            if col2.button("Nie, kontynuuj", key="reject_cancel_btn"): st.session_state.cancel_confirmation = False; st.rerun()

    else:
        # Simulation Finished - display results area
        st.markdown("---")
        st.header("Wyniki Symulacji")
        # Display results DataFrame and Chart (if available in state)
        results_available = False
        if st.session_state.get("df") is not None:
            st.subheader("Tabela Wynikow")
            st.dataframe(st.session_state.df, use_container_width=True)
            results_available = True
        if st.session_state.get("chart") is not None:
            st.subheader("Wykres Wynikow")
            st.altair_chart(st.session_state.chart, use_container_width=True)
            results_available = True

        # Show placeholder message if finished but no results loaded
        if not results_available:
             progress_placeholder.warning("Brak wynikow do wyswietlenia dla tej zakonczonej symulacji.")

        st.markdown("---")
        if st.button("Nowa Symulacja", key="new_sim_btn"):
            # Preserve login state, reset everything else
            auth_status=st.session_state.authentication_status
            name=st.session_state.name
            uname=st.session_state.username
            initialize_session_state() # Re-init state to defaults
            # Restore login state
            st.session_state.authentication_status=auth_status
            st.session_state.name=name
            st.session_state.username=uname
            # Clear potentially leftover files from previous run for THIS session
            remove_file(RESULTS_CSV_FILE)
            logging.info(f"User {username} starting New Simulation - State Reset (preserving login).")
            st.rerun()

def render_history_tab():
        """Renders the History tab (global history)."""
        st.title("Historia Symulacji (Globalna)") # Simulation History (Global)
        try:
            records = load_history() # Uses cache
            if not records:
                st.info("Brak zapisanej historii.") # No saved history.
                # Add refresh button even when empty
                if st.button("Odswiez Historie", key="refresh_history_empty"):
                    clear_history_cache(); st.rerun()
                return # Stop rendering if no records

            # Filtering and Refresh UI
            col_filter, col_refresh = st.columns([3, 1])
            with col_filter:
                all_models = sorted(list(set(r.get("model", "Nieznany") for r in records))) # Unknown
                filter_model = st.selectbox(
                    "Filtruj wg modelu", # Filter by model
                    ["Wszystkie"] + all_models, # All
                    index=0,
                    key="history_filter_model"
                )
            with col_refresh:
                 st.write("") # Vertical alignment spacer
                 if st.button("Odswiez Historie", key="refresh_history_list"): # Refresh History
                     clear_history_cache(); st.rerun()

            # Filter records based on selection
            filtered_records = [r for r in records if filter_model == "Wszystkie" or r.get("model") == filter_model] # All

            if not filtered_records:
                 st.info(f"Brak historii dla modelu: {filter_model}") # No history for model...
                 return

            st.write(f"Znaleziono {len(filtered_records)} zapisow.") # Found ... records.
            st.markdown("---")

            # --- Display Records ---
            for rec in filtered_records:
                 record_id = rec.get('record_id', 'N/A')
                 # Use record_id in the key for the card div if needed, otherwise just class
                 st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
                 timestamp = rec.get('timestamp', '?'); model = rec.get('model', '?'); user = rec.get('user', 'N/A'); # Get user if available
                 st.markdown(f"#### ID: {record_id} (User: {user}, {timestamp}) - Model: {model}")

                 col1, col2 = st.columns([1, 1])
                 record_dir = os.path.join(HISTORY_DIR, record_id)
                 results_path = os.path.join(record_dir, "results.csv")
                 code_filename = os.path.basename(SIMULATION_CODE_FILE) # Get base name
                 code_path = os.path.join(record_dir, code_filename)

                 with col1:
                     # Parameters Expander
                     with st.expander("Parametry", expanded=False): # Parameters
                         params_dict = rec.get("params", {});
                         if params_dict:
                             try:
                                 # Try to display using Pydantic model structure
                                 disp_params = SimulationConfig(**params_dict).get_relevant_config_dict()
                                 params_df = pd.DataFrame(list(disp_params.items()), columns=['Parametr', 'Wartosc']) # Parameter, Value
                                 st.table(params_df)
                             except Exception:
                                 # Fallback to simple JSON display if params don't match model
                                 st.json(params_dict, expanded=False)
                         else:
                             st.write("Brak parametrow.") # No parameters.

                     # Results Table & Download
                     st.markdown("##### Wyniki (tabela)") # Results (table)
                     if os.path.exists(results_path):
                         try:
                             df_data = pd.read_csv(results_path, encoding='utf-8')
                             st.dataframe(df_data, height=200, use_container_width=True)
                             # Download Button (grouped with table)
                             with open(results_path, "rb") as fp:
                                 st.download_button(
                                     label="Pobierz CSV", # Download CSV
                                     data=fp,
                                     file_name=f"{record_id}_results.csv",
                                     mime="text/csv",
                                     key=f"dl_csv_{record_id}" # Unique key
                                 )
                         except pd.errors.EmptyDataError:
                             st.warning("Plik wyników jest pusty.") # Results file is empty.
                         except Exception as e:
                             st.error(f"Blad odczytu results.csv: {e}") # Error reading results.csv
                     else:
                         st.warning("Brak pliku results.csv.") # Missing results.csv file.

                 with col2:
                     # Chart
                     st.markdown("##### Wykres") # Chart
                     if os.path.exists(results_path):
                         try:
                             df_hist = pd.read_csv(results_path, encoding='utf-8')
                             if not df_hist.empty:
                                 model_hist = rec.get("model", "")
                                 extra_conf = rec.get("extra_config", {})
                                 chart = generate_plot_history(df_hist, model_hist, extra_conf)
                                 st.altair_chart(chart, use_container_width=True)
                             else:
                                 st.warning("Plik pusty, brak wykresu.") # File empty, no chart.
                         except Exception as e:
                             st.error(f"Blad generowania wykresu: {e}") # Error generating chart
                     else:
                         st.warning("Brak pliku - brak wykresu.") # Missing file - no chart.

                     # Code Expander & Download (Corrected Block)
                     with st.expander(f"Kod ({code_filename})", expanded=False): # Code (...)
                         if os.path.exists(code_path):
                             try:
                                 # Read the code content first
                                 with open(code_path, "r", encoding="utf-8") as f:
                                      code_content = f.read()
                                 # Display the code content
                                 st.code(code_content, language='python')
                                 # Then provide the download button
                                 with open(code_path, "rb") as fp:
                                     st.download_button(
                                         label="Pobierz Kod", # Download Code
                                         data=fp,
                                         file_name=f"{record_id}_{code_filename}",
                                         mime="text/x-python",
                                         key=f"dl_code_{record_id}" # Unique key
                                     )
                             except Exception as e:
                                 st.error(f"Blad odczytu lub wyswietlania kodu: {e}") # Error reading or displaying code
                         else:
                             st.warning("Brak pliku kodu.") # Missing code file.

                 # Delete Button Logic (outside col2, but within the card loop)
                 st.markdown("<br>", unsafe_allow_html=True) # Add space before delete button
                 delete_key = f"delete_{record_id}"; confirm_delete_key = f"confirm_delete_{record_id}";
                 # Initialize confirmation state if not present
                 if confirm_delete_key not in st.session_state: st.session_state[confirm_delete_key] = False

                 # Use a column just for the delete button if needed for layout, or place directly
                 if st.button(f"Usun Historie {record_id}", key=f"btn_{delete_key}", help=f"Usun rekord historii {record_id}"): # Delete History
                     st.session_state[confirm_delete_key] = True; st.rerun() # Rerun to show confirmation

                 # Confirmation dialog logic
                 if st.session_state.get(confirm_delete_key, False):
                     st.warning(f"Czy na pewno usunac **{record_id}**?") # Are you sure...?
                     col_confirm, col_cancel = st.columns(2)
                     if col_confirm.button("Tak, usun", key=f"yes_{delete_key}"): # Yes, delete
                         if delete_history_record(record_id):
                             st.success(f"Usunieto {record_id}") # Deleted...
                         else:
                             st.error("Nie udalo sie usunac.") # Failed to delete.
                         st.session_state[confirm_delete_key] = False; st.rerun() # Reset flag and refresh list
                     if col_cancel.button("Anuluj", key=f"no_{delete_key}"): # Cancel
                         st.session_state[confirm_delete_key] = False; st.rerun() # Reset flag and refresh UI

                 # --- End of Card ---
                 st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            logging.error(f"Error rendering history tab: {e}", exc_info=True)
            st.error(f"Wystapil blad podczas wyswietlania historii: {e}") # An error occurred while displaying history

def render_saved_sims_tab():
    """Renders the Saved Simulations tab with comparison (currently global)."""
    st.title("Zapisane Symulacje i Porownanie (Globalne)")
    st.write("Zapisz symulacje z historii i porownaj wybrane.")
    st.markdown("---")
    try:
        saved_sims = st.session_state.get("saved_simulations", {})
        history_records = load_history() # Use cached history
        record_options_dict = { r.get('record_id'): f"{r.get('record_id','N/A')} ({r.get('model', '?')}, {r.get('timestamp', '')})" for r in history_records if r.get('record_id')}
        record_options_list = ["(brak)"] + sorted(list(record_options_dict.values()), reverse=True) # Sort dropdown by timestamp implicitly
        display_to_id_map = {v: k for k, v in record_options_dict.items()}
        slot_names = [f"Slot {i+1}" for i in range(6)] + ["Reference 1", "Reference 2"]

        st.header("Zarzadzanie Slotami")
        num_columns_manage = 3; cols_manage = st.columns(num_columns_manage)
        for i, slot_name in enumerate(slot_names):
             with cols_manage[i % num_columns_manage]:
                 st.markdown(f"**{slot_name}**"); slot_key_base = f"slot_manage_{slot_name.replace(' ', '_')}"
                 if slot_name in saved_sims:
                     sim_info = saved_sims[slot_name]; record_id = sim_info.get('record_id', 'N/A'); model_ = sim_info.get('model', '?'); st.caption(f"Zawiera: {record_id[:8]}... ({model_})")
                     del_key = f"del_{slot_key_base}";
                     if st.button(f"Oproznij Slot", key=del_key):
                         st.session_state.saved_simulations.pop(slot_name, None); save_json(SAVED_SIM_FILE, st.session_state.saved_simulations);
                         if st.session_state.get("compare_slot_1") == slot_name: st.session_state.compare_slot_1 = None
                         if st.session_state.get("compare_slot_2") == slot_name: st.session_state.compare_slot_2 = None;
                         st.rerun();
                 else:
                     select_key = f"select_{slot_key_base}"; chosen_display = st.selectbox("Wybierz z historii:", record_options_list, key=select_key, index=0, label_visibility="collapsed")
                     if chosen_display != "(brak)":
                         chosen_id = display_to_id_map.get(chosen_display)
                         if chosen_id:
                             assign_key = f"assign_{slot_key_base}";
                             if st.button(f"Przypisz '{chosen_id[:8]}...' do Slotu", key=assign_key):
                                 rec_to_save = next((r for r in history_records if r.get("record_id") == chosen_id), None)
                                 if rec_to_save: st.session_state.saved_simulations[slot_name] = rec_to_save; save_json(SAVED_SIM_FILE, st.session_state.saved_simulations); st.rerun()
                                 else: st.error(f"Nie znaleziono {chosen_id}")
                         else: st.error("Blad mapowania ID")

        st.markdown("---"); st.header("Porownanie Zapisanych Symulacji")
        available_slots = sorted(list(saved_sims.keys()))
        if len(available_slots) < 2: st.info("Zapisz przynajmniej dwie symulacje.")
        else:
            col_comp1, col_comp2 = st.columns(2)
            # Set indices trying to avoid selecting the same slot
            index1 = available_slots.index(st.session_state.compare_slot_1) if st.session_state.get("compare_slot_1") in available_slots else 0
            index2 = available_slots.index(st.session_state.compare_slot_2) if st.session_state.get("compare_slot_2") in available_slots else (1 if len(available_slots) > 1 else 0)
            if index1 == index2 and len(available_slots) > 1: index2 = (index1 + 1) % len(available_slots)

            slot1_name = col_comp1.selectbox("Wybierz Slot 1:", available_slots, key="compare_slot_1", index=index1)
            slot2_name = col_comp2.selectbox("Wybierz Slot 2:", available_slots, key="compare_slot_2", index=index2)

            if slot1_name and slot2_name and slot1_name != slot2_name:
                st.markdown("---"); sim_info1 = saved_sims.get(slot1_name); sim_info2 = saved_sims.get(slot2_name);
                if sim_info1 and sim_info2:
                     c1, c2 = st.columns(2);
                     with c1: st.subheader(f"Slot 1: {slot1_name}"); display_comparison_details(sim_info1)
                     with c2: st.subheader(f"Slot 2: {slot2_name}"); display_comparison_details(sim_info2)
                else: st.error("Nie mozna zaladowac danych.")
            elif slot1_name == slot2_name and len(available_slots)>1:
                 st.warning("Wybierz dwa rozne sloty.") # Show warning only if selection is possible

    except Exception as e:
        logging.error(f"Error rendering saved simulations tab: {e}", exc_info=True)
        st.error(f"Blad w zakladce Zapisane Symulacje: {e}")

def display_comparison_details(sim_info):
    """Helper to display details of a saved sim for comparison."""
    record_id = sim_info.get('record_id', 'N/A'); model_ = sim_info.get('model', '?');
    st.markdown(f"**ID:** {record_id} | **Model:** {model_}")
    with st.expander("Parametry", expanded=False):
        params = sim_info.get("params", {});
        if params: st.json(params, expanded=False)
        else: st.write("Brak parametrow.")
    st.markdown("**Wykres:**")
    record_dir = os.path.join(HISTORY_DIR, record_id) if record_id != 'N/A' else None;
    results_path = os.path.join(record_dir, "results.csv") if record_dir else None;
    if results_path and os.path.exists(results_path):
         try:
             df_hist = pd.read_csv(results_path, encoding='utf-8');
             if not df_hist.empty:
                 extra_conf = sim_info.get("extra_config", {});
                 chart = generate_plot_history(df_hist, model_, extra_conf);
                 st.altair_chart(chart, use_container_width=True)
             else: st.caption("Brak danych.")
         except Exception as e: st.warning(f"Blad wykresu: {e}")
    else: st.caption(f"Brak pliku.")


# --- Sidebar ---
def render_sidebar():
    """Renders the sidebar content."""
    st.sidebar.title("Status i Opcje")

    # User info and Logout Button
    if st.session_state["authentication_status"]:
        try:
             st.sidebar.write(f"Witaj, **{st.session_state['name']}**!")
             authenticator.logout('Wyloguj', 'sidebar', key='logout_btn')
        except Exception as e:
             st.sidebar.error("Blad wylogowania.")
             logging.error(f"Logout button error: {e}")
    else:
         st.sidebar.info("Zaloguj sie, aby uzyc aplikacji.")

    # About Section
    with st.sidebar.expander("O Aplikacji / Modelach", expanded=False):
        st.markdown("""
        **Symulator Gry Donacyjnej v2.1**

        Aplikacja do symulacji ewolucji strategii w Grze Donacyjnej.

        **Funkcje:**
        * Logowanie użytkownika
        * Śledzenie symulacji per użytkownik
        * Konfiguracja parametrów i modeli
        * Zapisywanie/ładowanie presetów konfiguracyjnych
        * Historia symulacji (globalna)
        * Zapisywanie i porównywanie wybranych przebiegów

        **Modele:** Bazowy, Z szumem (ea/ep), Generosnosc (g1/g2), Wybaczania.
        """)

    # User-specific Simulation Status
    st.sidebar.markdown("---")
    st.sidebar.subheader("Twoj Status Symulacji")
    if st.session_state.get("username"): # Check if user is logged in
        user_task_id = get_user_task_id(st.session_state.username)
        if user_task_id:
            # Display status based on session state, which should reflect the mapped task
            last_state = st.session_state.get("last_task_state")
            last_info = st.session_state.get("last_progress_info")
            current_session_task_id = st.session_state.get("task_id")

            if current_session_task_id == user_task_id and not st.session_state.simulation_finished:
                st.sidebar.warning(f"W toku... (ID: {user_task_id[:8]}...)")
                if last_state == 'PROGRESS' and last_info:
                    progress = last_info.get('progress', 0)
                    st.sidebar.progress(progress / 100.0)
                    completed = last_info.get('completed', 0); total = last_info.get('total', 1);
                    st.sidebar.caption(f"{completed}/{total} ({progress:.0f}%)")
                elif last_state:
                     st.sidebar.caption(f"Status: {last_state}")
                else: # State not yet checked in this session
                     st.sidebar.caption("Sprawdzanie statusu...")
            elif st.session_state.simulation_finished and current_session_task_id == user_task_id:
                 # Task finished within this session
                 st.sidebar.success("Zakonczona / Przerwana")
            else:
                # Mapping exists, but session state doesn't show it running (likely started elsewhere or session restarted)
                st.sidebar.warning(f"Aktywna w tle? (ID: {user_task_id[:8]}...)")
                st.sidebar.caption("Odswiez zakladke Symulacja, aby zaktualizowac.")
        else:
            st.sidebar.info("Brak aktywnych symulacji.")
    else:
        st.sidebar.info("Zaloguj sie, aby zobaczyc status.")


# --- Main App Logic ---
def main():
    """Główna funkcja aplikacji Streamlit."""
    initialize_session_state() # Inicjalizacja stanu sesji na początku

    # --- Ekran Logowania / Obsługa statusu autentykacji ---
    # Wywołanie login() - używamy poprawionej składni dla v0.3.x
    # Renderuje formularz logowania i zwraca status
    name, authentication_status, username = authenticator.login(
        fields={'Form name':'Logowanie', 'Username':'Nazwa użytkownika', 'Password':'Hasło', 'Login':'Zaloguj'},
        location='main' # Lokalizacja formularza logowania
    )

    # Aktualizacja stanu sesji na podstawie wyniku logowania
    st.session_state['authentication_status'] = authentication_status
    st.session_state['name'] = name
    st.session_state['username'] = username

    # --- Główna logika aplikacji (po sprawdzeniu logowania) ---
    if st.session_state["authentication_status"]:
        # --- UŻYTKOWNIK ZALOGOWANY ---
        render_sidebar() # Wyrenderuj pasek boczny (zawiera m.in. przycisk wylogowania)

        st.title(f"Donation Game Dashboard") # Tytuł główny (powitanie jest w sidebarze)

        # Sprawdź, czy zalogowany użytkownik ma zadanie działające w tle
        # (np. uruchomione w innej sesji przeglądarki lub przed restartem appki)
        # i zaktualizuj stan bieżącej sesji, jeśli to konieczne.
        if username: # Upewnij się, że username nie jest None
            user_task_id = get_user_task_id(username)
            # Aktualizuj stan sesji tylko jeśli mapowanie istnieje ORAZ
            # albo symulacja nie jest oznaczona jako uruchomiona w tej sesji,
            # albo ID zadania w sesji jest inne niż zmapowane ID.
            if user_task_id and (not st.session_state.simulation_started or st.session_state.task_id != user_task_id):
                logging.info(f"Znaleziono zmapowane zadanie {user_task_id} dla użytkownika {username}. Aktualizacja stanu sesji.")
                # Szybkie sprawdzenie, czy zadanie faktycznie nadal działa w Celery
                try:
                    task_still_active = celery_app.AsyncResult(user_task_id).state not in ['SUCCESS', 'FAILURE', 'REVOKED']
                except Exception:
                    task_still_active = False # Załóż, że nieaktywne, jeśli sprawdzenie zawiedzie
                    logging.error(f"Nie udało się sprawdzić statusu zmapowanego zadania {user_task_id} przy logowaniu.")

                if task_still_active:
                    st.session_state.update({
                        "task_id": user_task_id,
                        "simulation_started": True,
                        "simulation_finished": False,
                        "last_task_state": None, # Wymuś odświeżenie statusu
                        "last_progress_info": None
                        # Można by spróbować załadować config z historii, jeśli potrzebny
                    })
                    # Użyj toast dla dyskretnego powiadomienia
                    st.toast("Wykryto aktywną symulację w tle.", icon="⏳")
                else:
                    # Mapowanie istnieje, ale zadanie wygląda na zakończone - wyczyść mapowanie
                    logging.warning(f"Czyszczenie potencjalnie przestarzałego mapowania zadania dla użytkownika {username}, zadanie {user_task_id}")
                    set_user_task_id(username, None)
                    # Upewnij się, że stan sesji odzwierciedla brak działającego zadania
                    st.session_state.update({ "simulation_started": False, "simulation_finished": True, "task_id": None })

        # --- Definicja i renderowanie zakładek dla zalogowanego użytkownika ---
        tab_names = ["Symulacja", "Historia", "Zapisane / Porownaj"]
        tab1, tab2, tab3 = st.tabs(tab_names)

        # Renderowanie zawartości zakładek
        with tab1:
            # Użycie karty dla grupowania wizualnego
            st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
            if not st.session_state.simulation_started:
                # Jeśli żadna symulacja nie jest aktywna dla tego usera w tej sesji, pokaż formularz
                render_config_form(st.session_state.username)
            else:
                # Jeśli symulacja jest aktywna dla tego usera w tej sesji, pokaż widok statusu
                render_simulation_status_view(st.session_state.username)
            st.markdown('</div>', unsafe_allow_html=True)

        with tab2:
            # Wyświetl historię (obecnie globalną)
            render_history_tab()

        with tab3:
            # Wyświetl zapisane symulacje (obecnie globalne)
            render_saved_sims_tab()

    elif st.session_state["authentication_status"] == False:
        # --- BŁĘDNE LOGOWANIE ---
        st.error('Nazwa uzytkownika lub haslo jest nieprawidlowe')

    elif st.session_state["authentication_status"] == None:
        # --- FORMULARZ LOGOWANIA JEST WYŚWIETLANY ---
        # (authenticator.login sam go wyświetla w miejscu 'main')
        st.warning('Prosze podac nazwe uzytkownika i haslo')

# --- Entry Point --- (Pozostaje bez zmian)
if __name__ == "__main__":
    # Wykonaj krytyczne sprawdzenia przed uruchomieniem logiki Streamlit
    if not os.path.exists(AUTH_CONFIG_FILE):
         st.error(f"KRYTYCZNY BLAD: Brak pliku konfiguracyjnego '{AUTH_CONFIG_FILE}'. Aplikacja nie moze dzialac poprawnie.")
         st.stop() # Zatrzymaj wykonywanie, jeśli brakuje pliku konfiguracyjnego
    if not celery_available:
         # Wyświetl błąd, ale pozwól aplikacji działać w ograniczonym trybie
         st.error("KRYTYCZNY BLAD: Backend Celery jest niedostepny. Uruchamianie symulacji bedzie niemozliwe, ale mozesz przegladac historie/zapisane.")

    # Uruchom główną funkcję aplikacji
    main()
