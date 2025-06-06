import cv2
import sqlite3
import numpy as np
import tkinter as tk
from tkinter import messagebox, simpledialog
import time
import io
import os

# Caminhos para banco de dados e modelo treinado
DB_PATH = "faces.db"
TRAINER_PATH = "trainer.yml"
MAX_IMAGES = 30
CAPTURE_INTERVAL = 0.5  # intervalo entre capturas

# Inicializa o banco de dados SQLite e cria as tabelas, se não existirem
def init_db():
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            image BLOB NOT NULL,
            label_id INTEGER NOT NULL
        )""")
        c.execute("""
        CREATE TABLE IF NOT EXISTS user_map (
            username TEXT PRIMARY KEY,
            label_id INTEGER NOT NULL
        )""")


# Retorna o ID do usuário ou cria um novo se ele não existir
def get_or_create_user_id(username):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT label_id FROM user_map WHERE username=?", (username,))
        row = c.fetchone()
        if row:
            return row[0]
        c.execute("SELECT MAX(label_id) FROM user_map")
        max_id = c.fetchone()[0]
        new_id = 1 if max_id is None else max_id + 1
        c.execute("INSERT INTO user_map (username, label_id) VALUES (?, ?)", (username, new_id))
        conn.commit()
        return new_id


# Salva a imagem do rosto no banco de dados
def save_face_to_db(username, label_id, image):
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        # Converte a imagem para bytes
        is_success, buffer = cv2.imencode(".jpg", image)
        if is_success:
            blob = buffer.tobytes()
            c.execute("INSERT INTO faces (username, image, label_id) VALUES (?, ?, ?)",
                      (username, blob, label_id))
            conn.commit()


# Carrega os rostos e os rótulos do banco de dados
def load_faces_from_db():
    faces = []
    labels = []
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        for row in c.execute("SELECT image, label_id FROM faces"):
            image_data, label_id = row
            image_array = np.frombuffer(image_data, dtype=np.uint8)
            img = cv2.imdecode(image_array, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                faces.append(img)
                labels.append(label_id)
    return faces, labels


# Captura os rostos do usuário pela webcam
def collect_faces(username):
    label_id = get_or_create_user_id(username)
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
        return

    detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    count = 0
    last_capture_time = time.time()

    messagebox.showinfo("Informação", f"Capturando {MAX_IMAGES} imagens para '{username}'. Mantenha-se na frente da câmera.")

    while count < MAX_IMAGES:
        ret, frame = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        now = time.time()

        if len(faces) > 0 and (now - last_capture_time) >= CAPTURE_INTERVAL:
            (x, y, w, h) = faces[0]
            face = gray[y:y + h, x:x + w]
            save_face_to_db(username, label_id, face)
            count += 1
            last_capture_time = now

            # Janela de visualização (opcional)
            preview = cv2.resize(frame, (320, 240))
            cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.imshow("Registrando Rosto", preview)

        if cv2.waitKey(1) & 0xFF == 27:  # Tecla ESC
            break

    cam.release()
    cv2.destroyAllWindows()
    train_model()
    messagebox.showinfo("Concluído", f"Registro de rosto finalizado para '{username}'.")


# Treina o modelo com os rostos armazenados no banco
def train_model():
    faces, labels = load_faces_from_db()
    if not faces:
        messagebox.showwarning("Aviso", "Nenhum rosto encontrado para treinar.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))
    recognizer.save(TRAINER_PATH)


# Reconhece o rosto do usuário para fazer login
def recognize_face(threshold=60.0):
    if not os.path.exists(TRAINER_PATH):
        messagebox.showerror("Erro", "Modelo não treinado. Registre um usuário primeiro.")
        return

    # Mapeia os IDs de volta para nomes de usuário
    with sqlite3.connect(DB_PATH) as conn:
        c = conn.cursor()
        c.execute("SELECT username, label_id FROM user_map")
        id_to_user = {row[1]: row[0] for row in c.fetchall()}

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(TRAINER_PATH)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        messagebox.showerror("Erro", "Não foi possível acessar a webcam.")
        return

    messagebox.showinfo("Login", "Mostre seu rosto para fazer login. Pressione ESC para cancelar.")
    recognized_user = None

    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            roi = gray[y:y + h, x:x + w]
            label_id, confidence = recognizer.predict(roi)
            if confidence < threshold:
                recognized_user = id_to_user.get(label_id, "Desconhecido")
                break

        preview = cv2.resize(frame, (320, 240))
        cv2.imshow("Login com Rosto", preview)
        if recognized_user or cv2.waitKey(1) & 0xFF == 27:
            break

    cam.release()
    cv2.destroyAllWindows()

    if recognized_user:
        messagebox.showinfo("Login", f"✅ Bem-vindo(a) de volta, {recognized_user}!")
    else:
        messagebox.showwarning("Falha no Login", "❌ Rosto não reconhecido.")


# Interface gráfica com Tkinter
def main_gui():
    init_db()
    root = tk.Tk()
    root.title("Login com Reconhecimento Facial")
    root.geometry("300x240")

    tk.Label(root, text="Login com Rosto", font=("Arial", 16)).pack(pady=10)

    def on_register():
        username = simpledialog.askstring("Registrar", "Digite o nome de usuário:")
        if username:
            collect_faces(username.strip())

    def on_login():
        recognize_face()

    tk.Button(root, text="📷 Registrar Usuário", command=on_register, width=25).pack(pady=8)
    tk.Button(root, text="🔓 Login com Rosto", command=on_login, width=25).pack(pady=8)
    root.mainloop()


# Inicia a aplicação
if __name__ == "__main__":
    main_gui()
