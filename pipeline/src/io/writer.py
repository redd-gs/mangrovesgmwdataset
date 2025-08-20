import json
from pathlib import Path
import numpy as np
from PIL import Image


def write_image(image, output_path):
    """Écrit une image numpy (float 0..1 ou uint8) en PNG.

    Ajoute des vérifications pour détecter les tableaux constants (tout noir/blanc) et
    applique une normalisation simple si la dynamique est très faible afin d'éviter des
    sorties uniformes.
    """
    try:
        arr = np.asarray(image)
        if arr.ndim == 2:
            pass  # grayscale
        elif arr.ndim == 3 and arr.shape[2] in (3, 4):
            pass
        else:
            raise ValueError(f"Forme d'image non supportée: {arr.shape}")

        if arr.dtype != np.uint8:
            # Suppose float ou autre: on force dans [0,1]
            arr = arr.astype("float32")
            amin, amax = float(arr.min()), float(arr.max())
            if amax > amin:
                # Si dynamique extrêmement faible (<1/255), on étire
                if (amax - amin) < 1/255:
                    arr = (arr - amin) * (255.0 / max((amax - amin), 1e-12))
                else:
                    arr = (arr - amin) / (amax - amin) * 255.0
            else:
                # Image totalement constante -> on répète la valeur *255
                arr = np.full_like(arr, fill_value=amin * 255.0)
            arr = np.clip(arr + 0.5, 0, 255).astype(np.uint8)

        # Diagnostic
        uniq = np.unique(arr)
        if uniq.size == 1:
            print(f"⚠️ Image constante (valeur {uniq[0]}) → vérifier la source/mask.")
        elif uniq.size < 10:
            print(f"ℹ️ Faible diversité de pixels ({uniq.size} valeurs). Min={arr.min()} Max={arr.max()}")

        outp = Path(output_path)
        outp.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(arr).save(outp)
        print(f"✅ Image saved: {outp}")
    except Exception as e:
        print(f"❌ Error saving image: {str(e)}")

def write_metadata(metadata, output_path):
    """Writes metadata to a JSON file."""
    try:
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        print(f"✅ Metadata saved: {output_path}")
    except Exception as e:
        print(f"❌ Error saving metadata: {str(e)}")