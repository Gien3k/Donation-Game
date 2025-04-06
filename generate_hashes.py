# generate_hashes.py
import streamlit_authenticator as stauth
import sys
import bcrypt # Upewnij się, że bcrypt jest zainstalowany (z requirements.txt)

if len(sys.argv) > 1:
    passwords_to_hash = sys.argv[1:]
    # Używamy bezpośrednio bcrypt, bo stauth może mieć problemy z generowaniem poza aplikacją
    hashed_passwords = []
    for password in passwords_to_hash:
        hashed = bcrypt.hashpw(password.encode(), bcrypt.gensalt())
        hashed_passwords.append(hashed.decode()) # Dekoduj do stringa
    print("Wygenerowane Hashe (skopiuj odpowiedni):")
    for i, h in enumerate(hashed_passwords):
        print(f"  Haslo '{passwords_to_hash[i]}': {h}")

else:
    print("Usage: python generate_hashes.py <password_1> <password_2> ...")
    print("Przyklad: python generate_hashes.py admin")
