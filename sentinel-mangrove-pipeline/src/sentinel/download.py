from pathlib import Path
from typing import Iterable, List, Tuple
import concurrent.futures as cf
import os
import numpy as np
from PIL import Image, ImageFilter
from sentinelhub import (
    BBox, CRS, MimeType, SentinelHubRequest,
    DataCollection, bbox_to_dimensions
)
import sentinelhub as sh
from config.settings import settings
from core.context import get_sh_config

TRUE_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:3, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    return [s.B04, s.B03, s.B02];
}
"""

ENHANCED_COLOR_EVALSCRIPT = """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
        output: [{id:"default", bands:3, sampleType:"FLOAT32"}]
    };
}
function evaluatePixel(s){
    // Légère accentuation canal rouge et vert pour distinguer végétation
    return [s.B04*1.2, s.B03*1.1, s.B02];
}
"""

DEBUG = os.getenv("DEBUG_SH", "1") not in ("0", "false", "False")  # mettre DEBUG_SH=0 pour désactiver

DEBUG_EVALSCRIPT = """//VERSION=3
function setup(){
    return {
        input: [{bands:["B02","B03","B04","dataMask"], units:"REFLECTANCE"}],
        output: [
            {id:"rgb", bands:3, sampleType:"FLOAT32"},
            {id:"mask", bands:1, sampleType:"UINT8"}
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
                    output_path: Path,
                    enhanced: bool = False) -> bool:
    cfg = settings()
    sh_cfg = get_sh_config()
    evalscript = (
        DEBUG_EVALSCRIPT
        if DEBUG else
        (ENHANCED_COLOR_EVALSCRIPT if enhanced else TRUE_COLOR_EVALSCRIPT)
    )

    # Récupère l'intervalle de temps centralisé (YYYY-MM-DD, YYYY-MM-DD)
    start_date, end_date = cfg.time_interval_tuple
    req = SentinelHubRequest(
        evalscript=evalscript,
        input_data=[
            SentinelHubRequest.input_data(
                data_collection=DataCollection.SENTINEL2_L2A,
                time_interval=(start_date, end_date),
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

            # --- Normalisation robuste ---
            # Cas attendus:
            #  - FLOAT32 déjà en 0..1
            #  - UINT8 / valeurs 0..255 (sampleType implicite)
            #  - Valeurs entières 0..10000 (échelle réflectance *10000)
            rmax = float(rgb.max()); rmin = float(rgb.min())
            scale_info = None
            if rmax <= 1.5:  # déjà 0..1
                scale_info = "float_0_1"
            elif rmax <= 255 and rmin >= 0:
                rgb = rgb / 255.0
                scale_info = "uint8_scaled"
            elif rmax <= 10000 and rmin >= 0:
                rgb = rgb / 10000.0
                scale_info = "uint16_reflectance_scaled"
            else:
                # Normalisation min-max fallback
                if rmax > rmin:
                    rgb = (rgb - rmin) / (rmax - rmin)
                    scale_info = "minmax_fallback"
                else:
                    rgb = np.zeros_like(rgb, dtype="float32")
                    scale_info = "degenerate"

            # Conversion du masque (0/1 ou 0/255) en 0..1
            if mask.dtype != np.bool_:
                mmax = float(mask.max()) if mask.size else 1.0
                if mmax > 1.5:
                    mask = mask / mmax
            mask = np.clip(mask, 0, 1)

            # Amélioration gamma éventuelle
            if cfg.ENHANCEMENT_METHOD == "gamma":
                try:
                    rgb = np.power(np.clip(rgb, 1e-6, 1), cfg.GAMMA_VALUE)
                except Exception as _:
                    pass

            out_png = output_path.with_suffix(".png")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                # --- Amélioration avancée sans histogrammes (version affinée) ---
                # 1. Stretch percentile par canal
                clip = float(getattr(cfg, 'CLIP_VALUE', 2.2))
                stretched = []
                for c in range(3):
                    ch = rgb[..., c]
                    p_low, p_high = np.percentile(ch, (clip, 100-clip))
                    if p_high > p_low:
                        ch = (ch - p_low)/(p_high - p_low)
                    stretched.append(np.clip(ch, 0, 1))
                rgb = np.stack(stretched, axis=-1)

                scientific = os.getenv('SCIENTIFIC_MODE', '0') in ('1','true','True')

                if not scientific:
                    # 2. Courbe tonale douce: relever ombres, compresser hautes lumières
                    shadow_lift = float(os.getenv('SHADOW_LIFT', '0.06'))  # ~ +6%
                    highlight_comp = float(os.getenv('HIGHLIGHT_COMPRESS', '0.04'))  # ~ -4%
                    # Lift: fonction décroissante avec la luminance (plus forte sur les basses valeurs)
                    luminance = (0.299*rgb[...,0] + 0.587*rgb[...,1] + 0.114*rgb[...,2])
                    lift_factor = (1 - luminance**1.5)  # proche de 1 dans les ombres
                    rgb = np.clip(rgb + shadow_lift*lift_factor[...,None], 0, 1)
                    # Compression hautes lumières progressive au-dessus de 0.65
                    hl_mask = np.clip((luminance - 0.65)/0.35, 0, 1)
                    rgb = np.clip(rgb - highlight_comp*hl_mask[...,None], 0, 1)

                    # 3. Gamma d'affichage (approx sRGB) pour rendu carto
                    display_gamma = float(os.getenv('DISPLAY_GAMMA', '2.2'))
                    rgb = np.power(np.clip(rgb,0,1), 1.0/display_gamma)

                    # 4. Réduction légère saturation globale (-10 à -20%)
                    sat_factor = float(os.getenv('SATURATION_FACTOR', '0.9'))  # <1 réduit
                    px = rgb
                    mean_channels = px.mean(axis=-1, keepdims=True)
                    px = mean_channels + (px - mean_channels)*sat_factor
                    # 5. Atténuation sélective eau trop bleue/cyan
                    blue_atten = float(os.getenv('BLUE_ATTEN_FACTOR', '0.97'))
                    r,g,b = px[...,0], px[...,1], px[...,2]
                    blue_dom = (b > r*1.05) & (b > g*1.05)
                    # Atténuation progressive selon dominance
                    dominance = np.clip((b - np.maximum(r,g))/0.2, 0, 1)
                    b = b - dominance * (1-blue_atten) * b
                    px = np.stack([r,g,b], axis=-1)
                    rgb = np.clip(px,0,1)
                else:
                    # Mode scientifique: on conserve radiométrie stretchée (linéaire) sans gamma/sat
                    pass

                # 6. Netteté douce guidée edges (éviter halos littoral)
                try:
                    pil = Image.fromarray((rgb*255+0.5).astype('uint8'), mode='RGB')
                    radius = float(os.getenv('SHARP_RADIUS', '0.5'))
                    amount = float(os.getenv('SHARP_AMOUNT', '0.6'))
                    if amount > 0:
                        # Edge mask via FIND_EDGES + flou
                        edge = pil.filter(ImageFilter.FIND_EDGES).convert('L')
                        edge = edge.filter(ImageFilter.GaussianBlur(radius=1.0))
                        edge_arr = np.array(edge).astype('float32')/255.0
                        edge_arr = edge_arr / (edge_arr.max()+1e-6)
                        # Unsharp
                        blur = pil.filter(ImageFilter.GaussianBlur(radius=radius))
                        orig = np.array(pil).astype('float32')
                        bl = np.array(blur).astype('float32')
                        detail = orig - bl
                        sharpened = orig + amount * detail * edge_arr[...,None]
                        pil = Image.fromarray(np.clip(sharpened,0,255).astype('uint8'), 'RGB')
                except Exception:
                    pil = Image.fromarray((rgb*255+0.5).astype('uint8'))

                pil.save(out_png)
                # Sauvegarde masque uniquement (plus besoin hist)
                try:
                    Image.fromarray((mask * 255).astype('uint8')).save(output_path.with_name(output_path.stem + '_mask.png'))
                except Exception:
                    pass
                print(f"[TRACE] Améliorations appliquées (stretch, tone curve, saturation-, sharpen doux). Fichier: {out_png.name}")
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
            # Normalisation équivalente au chemin debug
            imax = float(img.max()); imin = float(img.min())
            if imax <= 1.5:
                pass
            elif imax <= 255 and imin >= 0:
                img = img / 255.0
            elif imax <= 10000 and imin >= 0:
                img = img / 10000.0
            else:
                if imax > imin:
                    img = (img - imin) / (imax - imin)
                else:
                    img = np.zeros_like(img, dtype="float32")
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
                 prefix: str = "patch",
                 enhanced: bool = True,
                 workers: int = 1) -> List[Path]:
    cfg = settings()
    results: List[Path] = []

    def task(item):
        idx, bb = item
        out = cfg.OUTPUT_DIR / f"{prefix}_{idx}.png"
        ok = download_single(bb, out, enhanced=enhanced)
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
            ok_paths = run_download([test_bbox], prefix="test", enhanced=True)
            print(ok_paths)
        except Exception as e:
            print(f"[AVERTISSEMENT] Test manuel ignoré: {e}")

