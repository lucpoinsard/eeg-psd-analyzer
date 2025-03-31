'''
analyse_psd_gui.py

Ce script propose une interface graphique (Tkinter) pour analyser des fichiers EEG au format .set (EEGLAB),
en utilisant la bibliothèque MNE-Python.

Fonctionnalités :
- Sélection du fichier EEG .set
- Paramétrage des bandes de fréquences (delta, theta, alpha, beta)
- Découpage en epochs de durée fixe (par défaut : 2s)
- Calcul de la densité spectrale de puissance (PSD) via Welch (fenêtre de Hann)
- Exclusion automatique des événements 'boundary' (EEGLAB)
- Étiquetage des epochs avec le nom des événements
- Calcul :
  - De la puissance moyenne par bande pour chaque électrode (µV²)
  - De la puissance moyenne totale pour chaque epoch (toutes électrodes et bandes confondues)
- Export au format CSV avec suggestion automatique du nom de sortie
- Affichage d’un résumé des epochs traités

Auteur : Luc Poinsard
Date : Février 2025
Version : 1.0
'''

# ---------------------------------------------
# Installation automatique des packages requis
# ---------------------------------------------
import subprocess
import sys

required_packages = ['mne', 'numpy', 'pandas']

# Si un package est manquant, il est installé automatiquement
for package in required_packages:
    try:
        __import__(package)
    except ImportError:
        print(f"{package} n'est pas installé. Installation en cours...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# ---------------------------------------------
# Importations après vérification
# ---------------------------------------------
import tkinter as tk
from tkinter import filedialog, messagebox
import mne
import numpy as np
import pandas as pd
from mne.time_frequency import psd_array_welch
import os

# ---------------------------------------------
# Fonctions de l'interface
# ---------------------------------------------

def browse_file():
    """Permet de sélectionner un fichier .set"""
    filepath.set(filedialog.askopenfilename(filetypes=[("EEGLAB files", "*.set")]))
    suggest_output_filename()  # Propose automatiquement un nom de fichier CSV

def save_file():
    """Permet de choisir le nom et l’emplacement du fichier CSV"""
    savepath.set(filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")]))

def suggest_output_filename():
    """Propose un nom de sortie par défaut basé sur le fichier .set"""
    if filepath.get():
        base = os.path.basename(filepath.get()).replace(".set", "_psd.csv")
        savepath.set(os.path.join(os.path.dirname(filepath.get()), base))

# ---------------------------------------------
# Fonction principale d’analyse EEG
# ---------------------------------------------

def run_analysis():
    try:
        # --- Chargement du fichier EEG ---
        raw = mne.io.read_raw_eeglab(filepath.get(), preload=True)

        # --- Extraction des événements et étiquettes ---
        events, event_id = mne.events_from_annotations(raw)

        # --- Suppression des événements "boundary" ajoutés par EEGLAB ---
        if 'boundary' in event_id:
            boundary_code = event_id['boundary']
            events = events[events[:, 2] != boundary_code]
            event_id = {k: v for k, v in event_id.items() if k != 'boundary'}

        # --- Dictionnaire inverse pour récupérer les noms à partir des IDs ---
        event_code_to_label = {v: k for k, v in event_id.items()}

        # --- Création des epochs de durée fixe (par défaut : 2s) ---
        duration = float(epoch_duration.get())
        epochs = mne.make_fixed_length_epochs(raw, duration=duration, preload=True)

        # --- Récupération des bornes de fréquences pour chaque bande ---
        fmins = {b: float(entries[b+"_min"].get()) for b in bands}
        fmaxs = {b: float(entries[b+"_max"].get()) for b in bands}

        # --- Calcul de la PSD via Welch avec fenêtre de Hann ---
        psds, freqs = psd_array_welch(
            epochs.get_data(),
            sfreq=raw.info['sfreq'],
            fmin=min(fmins.values()),
            fmax=max(fmaxs.values()),
            n_fft=int(duration * raw.info['sfreq']),
            average='mean',
            window='hann'  # Utilisation d'une fenêtre de Hann (standard EEG)
        )

        # --- Calcul de la puissance absolue (µV²) par bande et par électrode ---
        band_power = {b: [] for b in bands}
        for psd in psds:  # Pour chaque epoch
            for b in bands:
                idx_band = np.logical_and(freqs >= fmins[b], freqs < fmaxs[b])
                band_width = fmaxs[b] - fmins[b]
                power = psd[:, idx_band].mean(axis=1) * band_width
                band_power[b].append(power)

        # --- Association des événements par nom à chaque epoch ---
        event_times = events[:, 0] / raw.info['sfreq']
        epoch_times = epochs.events[:, 0] / raw.info['sfreq']
        event_labels = []
        for t in epoch_times:
            label = ''
            for e_time, e_id in zip(event_times, events[:, 2]):
                if abs(t - e_time) < duration / 2:
                    label = event_code_to_label.get(e_id, str(e_id))
                    break
            event_labels.append(label)

        # --- Construction des lignes de résultats ---
        rows = []
        for i in range(len(epochs)):
            row = {'epoch_start': epoch_times[i], 'event': event_labels[i]}
            all_values = []

            # Stockage des valeurs pour chaque électrode et bande
            for b in bands:
                values = []
                for ch_idx, ch_name in enumerate(raw.info['ch_names']):
                    value = band_power[b][i][ch_idx]
                    row[f'{ch_name}_{b}'] = value
                    values.append(value)
                    all_values.append(value)
                # Moyenne par bande (toutes électrodes)
                row[f'{b}_avg'] = np.mean(values)

            # Puissance totale moyenne de l’epoch (toutes électrodes, toutes bandes)
            row['total_power'] = np.mean(all_values)
            rows.append(row)

        # --- Export CSV avec mention de l’unité ---
        df = pd.DataFrame(rows)
        try:
            with open(savepath.get(), "w", newline="") as f:
                f.write("# Unité des puissances : µV²\n")
                df.to_csv(f, index=False)
        except PermissionError:
            messagebox.showerror("Erreur", "Impossible d'écrire le fichier CSV. Fermez-le dans Excel si nécessaire.")
            return

        # --- Résumé pour l’utilisateur ---
        n_epochs = len(epochs)
        n_labeled = sum(1 for e in event_labels if e)
        messagebox.showinfo("Analyse terminée", f"{n_epochs} epochs traités\n{n_labeled} epochs étiquetés.")

    except Exception as e:
        messagebox.showerror("Erreur inattendue", str(e))

# ---------------------------------------------
# Interface graphique (Tkinter)
# ---------------------------------------------

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Analyse EEG - PSD par bandes")

    # Définition des bandes par défaut
    bands = ['delta', 'theta', 'alpha', 'beta']
    filepath = tk.StringVar()
    savepath = tk.StringVar()
    epoch_duration = tk.StringVar(value="2.0")
    entries = {}

    # --- Sélection du fichier .set ---
    tk.Label(root, text="Fichier .set :").grid(row=0, column=0, sticky="e")
    tk.Entry(root, textvariable=filepath, width=50).grid(row=0, column=1)
    tk.Button(root, text="Parcourir", command=browse_file).grid(row=0, column=2)

    # --- Paramètre : durée des epochs ---
    tk.Label(root, text="Durée des epochs (s) :").grid(row=1, column=0, sticky="e")
    tk.Entry(root, textvariable=epoch_duration).grid(row=1, column=1, sticky="w")

    # --- Paramètres des bandes de fréquences ---
    row = 2
    for b in bands:
        entries[b+"_min"] = tk.StringVar(value=str({'delta':1, 'theta':4, 'alpha':8, 'beta':13}[b]))
        entries[b+"_max"] = tk.StringVar(value=str({'delta':4, 'theta':8, 'alpha':13, 'beta':30}[b]))

        tk.Label(root, text=f"{b.capitalize()} min :").grid(row=row, column=0, sticky="e")
        tk.Entry(root, textvariable=entries[b+"_min"]).grid(row=row, column=1, sticky="w")
        tk.Label(root, text=f"{b.capitalize()} max :").grid(row=row, column=2, sticky="e")
        tk.Entry(root, textvariable=entries[b+"_max"]).grid(row=row, column=3, sticky="w")
        row += 1

    # --- Sélection du fichier de sortie ---
    tk.Label(root, text="Sauvegarder le fichier CSV :").grid(row=row, column=0, sticky="e")
    tk.Entry(root, textvariable=savepath, width=50).grid(row=row, column=1)
    tk.Button(root, text="Choisir", command=save_file).grid(row=row, column=2)
    row += 1

    # --- Lancement du traitement ---
    tk.Button(root, text="Démarrer l'analyse", command=run_analysis, bg="green", fg="white").grid(row=row, column=1, pady=10)

    # --- Exécution de l'interface ---
    root.mainloop()
