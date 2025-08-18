from pathlib import Path
from typing import Iterable, List, Tuple
import concurrent.futures as cf
import numpy as np
from PIL import Image
from sentinelhub import (
    BBox, CRS, MimeType, SentinelHubRequest,
    DataCollection, bbox_to_dimensions
)
import sentinelhub as sh
from config.settings import settings
from core.context import get_sh_config

TRUE_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04"],units:"REFLECTANCE"}],output:{bands:3}};}
function evaluatePixel(s){return [s.B04,s.B03,s.B02];}
"""

ENHANCED_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){return{input:[{bands:["B02","B03","B04"],units:"REFLECTANCE"}],output:{bands:3}};}
function evaluatePixel(s){return [s.B04*1.15,s.B03*1.08,s.B02];}
"""

DEBUG = True  # mettre False quand validé

DEBUG_EVALSCRIPT = """//VERSION=3
function setup(){
  return {
    input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
    output: [
      {id:"rgb", bands:3},
      {id:"mask", bands:1}
    ]
  };
}
function evaluatePixel(s){
  return {
    rgb: [s.B04, s.B03, s.B02],
    mask: [s.dataMask]
  };
}
"""

def download_single(bbox: BBox,
                    time_interval: Tuple[str, str],
                    output_path: Path,
                    enhanced: bool = False) -> bool:
    cfg = settings()
    sh_cfg = get_sh_config()
    evalscript = (
        DEBUG_EVALSCRIPT
        if DEBUG else
        (ENHANCED_COLOR_EVALSCRIPT if enhanced else TRUE_COLOR_EVALSCRIPT)
    )

    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=time_interval,
                mosaicking_order="leastCC"
            )
        ],
        responses=(
            [
                SentinelHubRequest.output_response("rgb", MimeType.TIFF),
                SentinelHubRequest.output_response("mask", MimeType.TIFF)
            ] if DEBUG else
            [SentinelHubRequest.output_response("default", MimeType.PNG)]
        ),
        bbox=bbox,
        size=bbox_to_dimensions(bbox, resolution=cfg.IMAGE_RESOLUTION),
        config=sh_cfg
    )
    try:
        data = req.get_data()
        # Instrumentation & parsing robuste en mode DEBUG
        if DEBUG:
            # URLs brutes des téléchargements
            try:
                for dl in getattr(req, 'download_list', []) or []:
                    print(f"[TRACE] URL: {getattr(dl, 'url', '??')}")
            except Exception:
                pass
            print("[TRACE] Détails bruts des réponses:")
            for i, d in enumerate(data):
                if isinstance(d, np.ndarray):
                    print(f"  - idx={i} ndarray shape={d.shape} dtype={d.dtype} min={float(d.min()):.4f} max={float(d.max()):.4f}")
                else:
                    extra = ''
                    if isinstance(d, dict):
                        keys = list(d.keys())[:5]
                        extra = f" keys={keys}"
                    print(f"  - idx={i} type={type(d).__name__}{extra}")
                    # Inspection détaillée des valeurs dict pour détecter arrays cachés
                    if isinstance(d, dict):
                        for k, v in list(d.items())[:5]:
                            if isinstance(v, np.ndarray):
                                print(f"    · key={k} ndarray shape={v.shape} min={float(v.min()):.4f} max={float(v.max()):.4f}")
                            else:
                                print(f"    · key={k} type={type(v).__name__}")

        def try_extract_from_container(container):
            """Heuristique pour extraire rgb & mask de structures dict/list inattendues."""
            rgb_candidate = None
            mask_candidate = None
            if isinstance(container, dict):
                # Chercher clés explicites
                if 'rgb' in container and isinstance(container['rgb'], np.ndarray):
                    rgb_candidate = container['rgb']
                if 'mask' in container and isinstance(container['mask'], np.ndarray):
                    mask_candidate = container['mask']
                # Sinon scanner valeurs
                if rgb_candidate is None:
                    for v in container.values():
                        if isinstance(v, np.ndarray) and v.ndim == 3 and v.shape[-1] in (3, 4):
                            rgb_candidate = v
                            break
                if mask_candidate is None:
                    for v in container.values():
                        if isinstance(v, np.ndarray) and v.ndim == 2:
                            mask_candidate = v
                            break
                # Explorer listes imbriquées
                if rgb_candidate is None or mask_candidate is None:
                    for v in container.values():
                        if isinstance(v, list):
                            for vv in v:
                                if isinstance(vv, np.ndarray):
                                    if vv.ndim == 3 and rgb_candidate is None:
                                        rgb_candidate = vv
                                    elif vv.ndim == 2 and mask_candidate is None:
                                        mask_candidate = vv
            elif isinstance(container, list):
                for v in container:
                    if isinstance(v, np.ndarray):
                        if v.ndim == 3 and rgb_candidate is None:
                            rgb_candidate = v
                        elif v.ndim == 2 and mask_candidate is None:
                            mask_candidate = v
            return rgb_candidate, mask_candidate

        if DEBUG:
            # Cas nominal: deux réponses séparées
            if len(data) == 2 and all(isinstance(d, np.ndarray) for d in data):
                rgb, mask = data
            elif len(data) == 1:
                single = data[0]
                if isinstance(single, np.ndarray):
                    rgb = single
                    mask = np.ones(rgb.shape[:2], dtype=np.uint8)
                    print("[AVERTISSEMENT] Une seule ndarray reçue (attendu 2). Masque synthétique créé.")
                else:
                    rgb, mask = try_extract_from_container(single)
                    if rgb is None:
                        print("[ERREUR] Impossible d'extraire un tableau RGB depuis la réponse unique → abandon tuile.")
                        return False
                    if mask is None:
                        print("[AVERTISSEMENT] Masque introuvable dans la structure. Masque synthétique.")
                        mask = np.ones(rgb.shape[:2], dtype=np.uint8)
            else:
                # Dernier recours heuristique: scanner toute la liste
                rgb, mask = try_extract_from_container(data)
                if rgb is None:
                    raise RuntimeError(f"Réponses inattendues (len={len(data)}) sans rgb détecté.")
                if mask is None:
                    print("[AVERTISSEMENT] Masque absent après heuristique globale. Synthétique.")
                    mask = np.ones(rgb.shape[:2], dtype=np.uint8)

            if not isinstance(rgb, np.ndarray) or rgb.ndim != 3:
                print(f"[ERREUR] Format RGB invalide (type={type(rgb)} ndim={getattr(rgb,'ndim','?')})")
                return False
            if not isinstance(mask, np.ndarray):
                print("[ERREUR] Masque non ndarray.")
                return False
            if mask.shape[:2] != rgb.shape[:2]:
                print(f"[AVERTISSEMENT] Dimensions mask {mask.shape} != rgb {rgb.shape} → redimension simple.")
                try:
                    # Redimension par répétition / cropping simple
                    mask = mask[:rgb.shape[0], :rgb.shape[1]]
                except Exception:
                    return False

            # Diagnostics intensité
            rgb_min = float(rgb.min()); rgb_max = float(rgb.max())
            mratio = float(mask.mean())
            print(f"[DEBUG] bbox={bbox} mask_mean={mratio:.3f} rgb_minmax=({rgb_min:.3f},{rgb_max:.3f}) shape={rgb.shape}")
            if rgb_max - rgb_min == 0:
                print("[ALERTE] RGB uniforme → vérifier credentials, intervalle temporel ou mosaicking_order.")
            if mratio < 0.05:
                print("[DEBUG] dataMask très faible (<5%). Zone potentiellement hors couverture ou entièrement masquée.")

            # Normalisation + amélioration gamma éventuelle
            rgb = np.clip(rgb, 0, 1)
            if cfg.ENHANCEMENT_METHOD == "gamma":
                try:
                    rgb = np.power(np.clip(rgb, 1e-6, 1), cfg.GAMMA_VALUE)
                except Exception as _:
                    pass

            out_png = output_path.with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                Image.fromarray((rgb * 255 + 0.5).astype("uint8")).save(out_png)
                Image.fromarray((mask * 255).astype("uint8")).save(
                    output_path.with_name(output_path.stem + "_mask.png")
                )
            except Exception as e:
                print(f"[ERREUR] Écriture fichier échouée: {e}")
                return False
            return True
        else:
            # Mode non-debug (flux simple mono-réponse attendu)
            if not data:
                print("[ERREUR] Aucune donnée reçue.")
                return False
            img = data[0]
            if not isinstance(img, np.ndarray):
                print(f"[ERREUR] Réponse inattendue type={type(img)}")
                return False
            img = np.clip(img, 0, 1)
            if cfg.ENHANCEMENT_METHOD == "gamma":
                try:
                    img = np.power(np.clip(img, 1e-6, 1), cfg.GAMMA_VALUE)
                except Exception:
                    pass
            output_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray((img * 255 + 0.5).astype("uint8")).save(output_path)
            return True
    except Exception as e:
        print(f"[ERREUR] Téléchargement échec: {e}")
        return False

def run_download(bboxes: Iterable[BBox],
                 time_interval: Tuple[str, str],
                 prefix: str = "patch",
                 enhanced: bool = True,
                 workers: int = 1) -> List[Path]:
    cfg = settings()
    results: List[Path] = []

    def task(item):
        idx, bb = item
        out = cfg.OUTPUT_DIR / f"{prefix}_{idx}.png"
        ok = download_single(bb, time_interval, out, enhanced=enhanced)
        return out if ok else None

    items = list(enumerate(bboxes, start=1))
    if workers > 1:
        with cf.ThreadPoolExecutor(max_workers=workers) as ex:
            for p in ex.map(task, items):
                if p:
                    results.append(p)
    else:
        for it in items:
            p = task(it)
            if p:
                results.append(p)
    return results

if __name__ == "__main__":
    # Test manuel optionnel (désactivé par défaut pour éviter erreurs lors d'import)
    DEBUG_RUN = False
    if DEBUG_RUN:
        try:
            print(f"[INFO] sentinelhub version: {sh.__version__}")
            wgs84 = getattr(CRS, "WGS84", None)
            if wgs84 is None:
                try:
                    wgs84 = CRS(4326)  # type: ignore
                except Exception:
                    wgs84 = None
            if wgs84 is None:
                raise RuntimeError("Impossible de récupérer CRS WGS84 pour le test manuel.")
            test_bbox = BBox([2.27, 48.84, 2.30, 48.86], crs=wgs84)
            ok_paths = run_download([test_bbox], ("2024-06-01", "2024-06-05"), prefix="test", enhanced=True)
            print(ok_paths)
        except Exception as e:
            print(f"[AVERTISSEMENT] Test manuel ignoré: {e}")

