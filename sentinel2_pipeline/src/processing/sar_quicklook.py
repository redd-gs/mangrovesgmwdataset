"""Génération de quicklooks RGB à partir des réponses Sentinel-1 (VV, VH) float32.

Lecture des dossiers data/output/sentinel1_time_series/*/response.tiff
Transformation: lin -> dB, mise à l'échelle par percentiles, pseudo-RGB (VV, moyenne, VH),
amélioration via `enhancements.enhance_image`, sauvegarde en quicklook.png.
"""
from pathlib import Path
import numpy as np
import rasterio
try:
    from .enhancements import enhance_image  # exécution comme module
except ImportError:
    from enhancements import enhance_image  # exécution directe
from PIL import Image


def _scale_db(db_arr: np.ndarray, p_low=2, p_high=98) -> np.ndarray:
    lo, hi = np.percentile(db_arr, [p_low, p_high])  # Calcule les valeurs des percentiles p_low et p_high du tableau (par défaut 2% et 98%)
    if hi - lo < 1e-6:                              # Vérifie si l'écart entre les deux percentiles est presque nul (pour éviter une division par zéro)
        return np.zeros_like(db_arr)                # Si oui, retourne un tableau de zéros de la même forme que db_arr
    scaled = (db_arr - lo) / (hi - lo)              # Normalise les valeurs du tableau entre 0 et 1 selon les percentiles calculés
    return np.clip(scaled, 0, 1)                    # Coupe les valeurs pour qu'elles restent entre 0 et 1 (tout ce qui est en dehors est ramené à


def sar_response_to_rgb(tiff_path: Path) -> np.ndarray:
    with rasterio.open(tiff_path) as ds:  # Ouvre le fichier TIFF contenant les images radar Sentinel-1
        arr = ds.read()  # (2,H,W) attendu: VV,VH  # Lit les deux bandes (VV et VH) sous forme de tableau numpy
    if arr.shape[0] < 2:  # Vérifie qu'il y a bien deux bandes dans le fichier
        raise ValueError(f"Le fichier {tiff_path} ne contient pas 2 bandes (VV,VH)")  # Erreur si ce n'est pas le cas
    vv_lin = arr[0].astype("float32")  # Récupère la première bande (VV) et la convertit en float32
    vh_lin = arr[1].astype("float32")  # Récupère la seconde bande (VH) et la convertit en float32
    # Valeurs anormales (très grandes) -> clamp simple pour éviter les infinities
    vv_lin = np.nan_to_num(np.clip(vv_lin, 0, np.percentile(vv_lin, 99.9)))  # Coupe les valeurs extrêmes de VV pour éviter les valeurs aberrantes
    vh_lin = np.nan_to_num(np.clip(vh_lin, 0, np.percentile(vh_lin, 99.9)))  # Coupe les valeurs extrêmes de VH pour éviter les valeurs aberrantes
    # Passage en dB (sigma0 dB approx.)
    vv_db = 10 * np.log10(vv_lin + 1e-6)  # Convertit VV de l'échelle linéaire à l'échelle décibel (dB)
    vh_db = 10 * np.log10(vh_lin + 1e-6)  # Convertit VH de l'échelle linéaire à l'échelle décibel (dB)
    # Mise à l'échelle par percentiles
    vv_s = _scale_db(vv_db)  # Normalise VV en dB entre 0 et 1 selon les percentiles (pour l'affichage)
    vh_s = _scale_db(vh_db)  # Normalise VH en dB entre 0 et 1 selon les percentiles (pour l'affichage)
    mean_s = _scale_db((vv_db + vh_db) / 2)  # Calcule la moyenne VV/VH, puis normalise entre 0 et 1
    rgb = np.stack([vv_s, mean_s, vh_s], axis=-1)  # Assemble les trois canaux (VV, moyenne, VH) en une image pseudo-RGB
    # Amélioration légère (contraste/brightness + gamma <1 pour éclaircir)
    enhanced = enhance_image(rgb, brightness_factor=1.15, contrast_factor=1.25, gamma=0.9)  # Améliore l'image pour la rendre plus lisible (contraste, luminosité, gamma)
    return enhanced  # Retourne l'image RGB améliorée

def process_all(root: Path = Path("data/output/sentinel1_time_series")):
    if not root.exists():
        print(f"Dossier {root} introuvable.")
        return
    count = 0
    for sub in sorted(root.iterdir()):
        if not sub.is_dir():
            continue
        tiff_path = sub / "response.tiff"
        if not tiff_path.exists():
            continue
        try:
            rgb = sar_response_to_rgb(tiff_path)
            out_path = sub / "quicklook.png"
            Image.fromarray((np.clip(rgb,0,1)*255).astype("uint8")).save(out_path)
            count += 1
            print(f"✔ Quicklook généré: {out_path}")
        except Exception as e:
            print(f"⚠ Erreur pour {tiff_path}: {e}")
    print(f"Terminé. Quicklooks générés: {count}")


if __name__ == "__main__":
    process_all()
