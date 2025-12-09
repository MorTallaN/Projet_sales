import tkinter as tk
from tkinter import messagebox
import joblib
import numpy as np

# Charger ton modèle
model = joblib.load("regression_model.pkl")

def predict_sales():
    try:
        tv = float(entry_tv.get())
        radio = float(entry_radio.get())

        X_new = np.array([[tv, radio]])
        prediction = model.predict(X_new)[0]

        messagebox.showinfo("Résultat", f"Prévision des ventes : {prediction:.2f} $")
    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer des valeurs numériques.")

# Fenêtre
root = tk.Tk()
root.title("Prédiction des ventes")

# Inputs
tk.Label(root, text="Budget TV ($)").pack()
entry_tv = tk.Entry(root)
entry_tv.pack()

tk.Label(root, text="Budget Radio ($)").pack()
entry_radio = tk.Entry(root)
entry_radio.pack()

# Bouton
tk.Button(root, text="Prédire", command=predict_sales).pack()

root.mainloop()
