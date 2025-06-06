# 🔐 Login com Reconhecimento Facial usando OpenCV, SQLite e Tkinter

Este projeto implementa um sistema de **login facial** com múltiplos usuários usando:
- OpenCV (visão computacional)
- SQLite (banco de dados local)
- Tkinter (interface gráfica)

As imagens dos rostos são armazenadas em um banco SQLite como BLOBs e um modelo de reconhecimento facial é treinado automaticamente após o cadastro de um novo usuário.

---

## 🖥️ Funcionalidades

- Registro de usuário com webcam
- Armazenamento de rostos no banco de dados SQLite
- Treinamento automático do modelo facial (LBPH)
- Login com reconhecimento facial
- Suporte a múltiplos usuários
- Interface gráfica simples via Tkinter

---

## 📦 Requisitos
 - Necessário ter o pipenv instalado
   
Instale os pacotes necessários com:

```bash
pipenv install
