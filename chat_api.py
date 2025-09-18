# chat_api.py — CLIP search (lazy init) + Firestore (lazy init) + robust fallbacks
from __future__ import annotations
import base64, io, os, time, json
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from flask import Blueprint, request, jsonify

import torch
from transformers import CLIPModel, CLIPProcessor

# ---------------- Firebase (lazy) ----------------
_db = None
_db_err: Exception | None = None

def get_db():
    """
    Lazily initialize Firestore so the app doesn't crash at import time if
    GOOGLE_APPLICATION_CREDENTIALS isn't set yet.
    """
    global _db, _db_err
    if _db is not None or _db_err is not None:
        return _db

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    try:
        import firebase_admin
        from firebase_admin import credentials, firestore

        if not cred_path or not os.path.exists(cred_path):
            raise RuntimeError("Set GOOGLE_APPLICATION_CREDENTIALS to your service account JSON")

        if not firebase_admin._apps:
            firebase_admin.initialize_app(credentials.Certificate(cred_path))
        _db = firestore.client()
        return _db
    except Exception as e:
        _db_err = e
        return None

def _product_collection():
    db = get_db()
    if db is None:
        return None
    from firebase_admin import firestore as _fs  # type: ignore
    return db.collection("app").document("products").collection("items")

# ---------------- CLIP (lazy) ----------------
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_MODEL_ID = "openai/clip-vit-base-patch32"  # 512-dim
_clip_model: CLIPModel | None = None
_clip_proc: CLIPProcessor | None = None

def _ensure_clip():
    global _clip_model, _clip_proc
    if _clip_model is None or _clip_proc is None:
        _clip_model = CLIPModel.from_pretrained(_MODEL_ID).to(_DEVICE).eval()
        _clip_proc  = CLIPProcessor.from_pretrained(_MODEL_ID)

def _embed_text(text: str) -> np.ndarray | None:
    if not text or not text.strip():
        return None
    _ensure_clip()
    ins = _clip_proc(text=[text], return_tensors="pt", padding=True, truncation=True)  # type: ignore
    ins = {k: v.to(_DEVICE) for k, v in ins.items()}
    with torch.no_grad():
        z = _clip_model.get_text_features(**ins)  # type: ignore
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    return z.squeeze(0).detach().cpu().numpy().astype("float32")

def _embed_image_b64(img_b64: str) -> np.ndarray | None:
    try:
        pil = Image.open(io.BytesIO(base64.b64decode(img_b64))).convert("RGB")
    except Exception:
        return None
    _ensure_clip()
    ins = _clip_proc(images=[pil], return_tensors="pt")  # type: ignore
    ins = {k: v.to(_DEVICE) for k, v in ins.items()}
    with torch.no_grad():
        z = _clip_model.get_image_features(**ins)  # type: ignore
    z = z / z.norm(p=2, dim=-1, keepdim=True)
    return z.squeeze(0).detach().cpu().numpy().astype("float32")

# ---------------- Cache: products + embeddings ----------------
_CACHE = {"ts": 0.0, "arr": np.zeros((0,512), dtype="float32"), "items": []}

def _load_products() -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    now = time.time()
    if _CACHE["arr"] is not None and now - _CACHE["ts"] < 60:
        return _CACHE["arr"], _CACHE["items"]

    col = _product_collection()
    if col is None:
        # DB not ready -> empty arrays (handler will 503)
        _CACHE.update({"ts": now, "arr": np.zeros((0,512), dtype="float32"), "items": []})
        return _CACHE["arr"], _CACHE["items"]

    docs = list(col.limit(5000).stream())
    items, vecs = [], []
    for d in docs:
        x = d.to_dict() or {}
        emb = x.get("clip_embedding") or []
        if isinstance(emb, list) and len(emb) == 512:
            items.append({
                "id": x.get("id") or d.id,
                "title": x.get("title") or "",
                "brand": x.get("brand") or "",
                "price_text": x.get("price_text") or x.get("price") or "",
                "product_url": x.get("product_url") or x.get("url") or "",
                "image_url": x.get("image_url") or x.get("cropped_image_url") or "",
                "color_distribution": x.get("color_distribution") or {},
                "traits": x.get("traits") or {},
            })
            vecs.append(emb)

    arr = np.asarray(vecs, dtype="float32")
    if arr.size:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        arr = arr / np.clip(norms, 1e-8, None)

    _CACHE.update({"ts": now, "arr": arr, "items": items})
    return arr, items

def _cosine_topk(q: np.ndarray | None, arr: np.ndarray, k: int = 10):
    if arr is None or arr.size == 0 or q is None:
        return [], []
    sims = arr @ q
    k = min(k, arr.shape[0])
    idx = np.argpartition(-sims, k-1)[:k]
    idx = idx[np.argsort(-sims[idx])]
    return idx.tolist(), sims[idx].tolist()

def _apply_filters(cands, sims, filters):
    colors = {c.lower() for c in (filters.get("colors") or [])}
    brand  = (filters.get("brand") or "").lower().strip()
    max_price = filters.get("max_price")

    def price_to_float(p):
        if not p: return None
        s = "".join(ch for ch in str(p) if ch.isdigit() or ch==".")
        try: return float(s) if s else None
        except: return None

    out = []
    for item, sim in zip(cands, sims):
        bonus = 0.0
        if colors and isinstance(item.get("color_distribution"), dict):
            cd = {k.lower(): float(v) for k, v in item["color_distribution"].items() if isinstance(v,(int,float))}
            color_score = sum(cd.get(c, 0.0) for c in colors)
            bonus += 0.10 * color_score
        if brand and brand in (item.get("brand") or "").lower():
            bonus += 0.05
        score = float(sim) + bonus
        if max_price is not None:
            p = price_to_float(item.get("price_text",""))
            if p is None or p > max_price:
                continue
        out.append((score, item))
    out.sort(key=lambda x: x[0], reverse=True)
    return [dict(item, _score=round(score, 4)) for score, item in out]

def _get_ref_embedding(ref_id: str) -> np.ndarray | None:
    if not ref_id: return None
    col = _product_collection()
    if col is None:
        return None
    snap = col.document(ref_id).get()
    if not snap.exists: return None
    x = snap.to_dict() or {}
    emb = x.get("clip_embedding")
    if isinstance(emb, list) and len(emb) == 512:
        v = np.asarray(emb, dtype="float32")
        n = np.linalg.norm(v)
        return v / max(n, 1e-8)
    return None

bp = Blueprint("chat_api", __name__)

@bp.route("/ping")
def ping():
    return {"ok": True}

@bp.route("/chat", methods=["POST"])
def chat_text_only():
    data = request.get_json(force=True) or {}
    msg = (data.get("message") or "").strip()
    top_k = int(data.get("top_k") or 8)
    filters = data.get("filters") or {}
    if not msg:
        return jsonify({"reply":"Tell me what you’re looking for.", "items":[]})
    arr, items = _load_products()
    if not getattr(arr, "size", 0):
        return jsonify({"reply":"Database not ready yet—try again in a moment.", "items":[]}), 503
    q = _embed_text(msg)
    idxs, sims = _cosine_topk(q, arr, k=max(top_k*3, top_k))
    cands = [items[i] for i in idxs]
    ranked = _apply_filters(cands, [float(s) for s in sims], filters)[:top_k]
    return jsonify({"reply": f'I looked for: “{msg}”.', "items": ranked})

@bp.route("/chat/messages", methods=["POST"])
def chat_messages():
    """
    Body:
    {
      "message": "blue denim jacket under $40",
      "image_base64": "<optional>",
      "ref_product_id": "<optional>",
      "top_k": 8,
      "filters": {"max_price": 40, "colors": ["blue"], "brand": "hm"}
    }
    """
    data = request.get_json(force=True) or {}
    text = (data.get("message") or "").strip()
    img_b64 = data.get("image_base64")
    ref_id = data.get("ref_product_id")
    top_k = int(data.get("top_k") or 8)
    filters = data.get("filters") or {}

    e_text = _embed_text(text) if text else None
    e_img  = _embed_image_b64(img_b64) if img_b64 else None
    e_ref  = _get_ref_embedding(ref_id) if ref_id else None

    parts, weights = [], []
    if e_ref is not None and e_img is not None and e_text is not None:
        parts, weights = [e_ref, e_img, e_text], [0.55, 0.25, 0.20]
    elif e_ref is not None and e_text is not None:
        parts, weights = [e_ref, e_text], [0.70, 0.30]
    elif e_img is not None and e_text is not None:
        parts, weights = [e_img, e_text], [0.60, 0.40]
    elif e_ref is not None:
        parts, weights = [e_ref], [1.0]
    elif e_img is not None:
        parts, weights = [e_img], [1.0]
    elif e_text is not None:
        parts, weights = [e_text], [1.0]
    else:
        return jsonify({"reply":"Tell me what you’re looking for or upload a photo.", "items":[]})

    q = np.zeros(512, dtype="float32")
    for p, w in zip(parts, weights):
        q += w * p
    n = np.linalg.norm(q)
    if n > 0: q = q / n

    arr, items = _load_products()
    if not getattr(arr, "size", 0):
        return jsonify({"reply":"Database not ready yet—try again in a moment.", "items":[]}), 503

    idxs, sims = _cosine_topk(q, arr, k=max(top_k*3, top_k))
    cands = [items[i] for i in idxs]
    ranked = _apply_filters(cands, [float(s) for s in sims], filters)[:top_k]

    bits = []
    if text: bits.append(f'“{text}”')
    if ref_id: bits.append(f'like item {ref_id}')
    if img_b64: bits.append('your photo')
    looked = " + ".join(bits) if bits else "your request"
    if filters.get("max_price") is not None:
        try:
            looked += f' under ${float(filters["max_price"]):.0f}'
        except:  # keep it robust
            pass
    if filters.get("colors"):
        looked += f' favoring {", ".join(filters["colors"])}'
    if filters.get("brand"):
        looked += f' and brand {filters["brand"]}'
    reply = f"I searched for {looked}. "
    reply += ("No good matches yet." if not ranked else f"Top match: {ranked[0].get('title') or 'Untitled'}.")
    return jsonify({"reply": reply, "items": ranked})
