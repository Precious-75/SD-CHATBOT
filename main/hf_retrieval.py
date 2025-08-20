# hf_retrieval.py
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import numpy as np

from smart_retrieval import SmartCSVChatbot
from csv_chatbot import CSVChatbot  # only to satisfy type hints/import order


class HFSmartCSVChatbot(SmartCSVChatbot):
    def __init__(self):
        # Initializes knowledge_base, responses, patterns, sentiment_analyzer, etc.
        super().__init__()

        # --- config & safety fallbacks ---
        if not hasattr(self, "config_file"):
            self.config_file = "chatbot_config.json"

        # In case any parent edit forgets to create these
        if not hasattr(self, "responses") or not hasattr(self, "patterns"):
            self.responses = getattr(self, "responses", {})
            self.patterns = getattr(self, "patterns", {
                'greeting': [r'hello', r'hi', r'hey', r'good morning', r'good afternoon'],
                'goodbye':  [r'bye', r'goodbye', r'see you', r'farewell', r'thanks', r'thank you'],
            })

        # --- retrieval options / weights ---
        self.use_hybrid = True
        self.use_embeddings = True
        self.use_cross_encoder = False  # optional re-ranker; safe no-op if model missing

        self._top_k = 5
        self._emb_weight = 0.6
        self._tfidf_weight = 0.25
        self._min_conf = 0.15

        # --- metadata context & boosts ---
        self.user_context: Dict[str, Any] = {
            "os": None,
            "brand": None,
            "categories": None,        # Optional[List[str]]
            "school_levels": None,     # Optional[List[str]]
            "strict_category_filter": False,
        }
        self._cat_boost = 0.10
        self._level_boost = 0.08
        self._mismatch_penalty = 0.05  # when strict=False and category mismatches

        # --- internal state (trained objects) ---
        self._vectorizer = None
        self._matrix = None
        self._rows_meta: List[Dict[str, str]] = []

        self._hf_model = None        # SentenceTransformer (embeddings)
        self._emb_matrix = None      # np.ndarray [N, D]

        self._ce_model = None        # CrossEncoder (optional re-ranker)

    # ------------------- persistence -------------------
    def save_config(self, csv_file_path: str) -> None:
        try:
            with open(self.config_file, "w", encoding="utf-8") as f:
                json.dump({"csv_file_path": csv_file_path}, f, indent=2)
        except Exception as e:
            print(f" Error saving config: {e}")

    def load_config(self) -> Optional[str]:
        try:
            if os.path.exists(self.config_file):
                with open(self.config_file, "r", encoding="utf-8") as f:
                    cfg = json.load(f)
                    return cfg.get("csv_file_path")
        except Exception as e:
            print(f" Error loading config: {e}")
        return None

    # Make sure we (re)train whenever a CSV is loaded
    def load_csv_file(self, file_path: str, save_to_config: bool = True) -> bool:
        ok = super().load_csv_file(file_path, save_to_config)
        if ok:
            self._train_vectors()
            if self.use_embeddings:
                self._train_embeddings()
            if save_to_config:
                self.save_config(file_path)
        return ok

    # ------------------- metadata helpers -------------------
    def get_meta_vocab(self) -> Dict[str, List[str]]:
        cats, levels = set(), set()
        for _, data in self.knowledge_base.items():
            meta = data.get('meta', {})
            cats.add(meta.get('category', 'Uncategorized'))
            levels.add(meta.get('school_level', 'All'))
        return {"categories": sorted(cats), "school_levels": sorted(levels)}

    def set_context(
        self,
        os_name: Optional[str] = None,
        brand: Optional[str] = None,
        categories: Optional[List[str]] = None,
        school_levels: Optional[List[str]] = None,
        strict_category_filter: bool = False
    ):
        if os_name:
            self.user_context["os"] = os_name
        if brand:
            self.user_context["brand"] = brand
        self.user_context["categories"] = categories
        self.user_context["school_levels"] = school_levels
        self.user_context["strict_category_filter"] = bool(strict_category_filter)

    # ------------------- lazy loaders -------------------
    def _lazy_load_st_model(self) -> None:
        if self._hf_model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
            self._hf_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            print(f" Embeddings disabled: {e}")
            self._hf_model = None

    def _lazy_load_ce_model(self) -> None:
        if self._ce_model is not None:
            return
        try:
            from sentence_transformers import CrossEncoder
            # small, popular MS MARCO cross-encoder
            self._ce_model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f" Cross-encoder disabled: {e}")
            self._ce_model = None

    # ------------------- training -------------------
    def _ensure_trained(self) -> None:
        if not getattr(self, "_rows_meta", None):
            self._train_vectors()
        if self.use_embeddings and getattr(self, "_emb_matrix", None) is None:
            self._train_embeddings()

    def _train_vectors(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            self._vectorizer, self._matrix, self._rows_meta = None, None, []
            return

        texts: List[str] = []
        rows_meta: List[Dict[str, str]] = []
        for norm_q, data in self.knowledge_base.items():
            q = data['original_question']
            a = data['answer']
            m = data.get('meta', {})
            cat = m.get('category', 'Uncategorized')
            lvl = m.get('school_level', 'All')
            combo = f"{q} || {a} || CAT:{cat} LEVEL:{lvl}"
            texts.append(combo)
            rows_meta.append({'q': q, 'a': a, 'norm_q': norm_q})

        if not texts:
            self._vectorizer, self._matrix, self._rows_meta = None, None, []
            return

        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            analyzer='char', ngram_range=(3, 5),
            lowercase=True, strip_accents='unicode', min_df=1
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self._rows_meta = rows_meta

    def _train_embeddings(self):
        if not self.knowledge_base:
            self._emb_matrix = None
            return
        self._lazy_load_st_model()
        if self._hf_model is None:
            self._emb_matrix = None
            return
        if not getattr(self, "_rows_meta", None):
            self._train_vectors()

        texts: List[str] = []
        for meta in self._rows_meta:
            kb = self.knowledge_base[meta['norm_q']]
            q, a = kb['original_question'], kb['answer']
            m = kb.get('meta', {})
            cat = m.get('category', 'Uncategorized')
            lvl = m.get('school_level', 'All')
            texts.append(f"{q} || {a} || CAT:{cat} LEVEL:{lvl}")
            embeddings = self._hf_model.encode(texts, normalize_embeddings=True)
            self._emb_matrix = np.asarray(embeddings, dtype=np.float32)

    # ------------------- scorers -------------------
    def _tfidf_search(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        vec = getattr(self, "_vectorizer", None)
        mat = getattr(self, "_matrix", None)
        if vec is None or mat is None:
            self._train_vectors()
            vec = self._vectorizer
            mat = self._matrix
            if vec is None or mat is None:
                return []
            cats = self.user_context.get("categories") or []
            lvls = self.user_context.get("school_levels") or []
            q = query
            if cats:
                q += " " + " ".join(f"CAT:{c}" for c in cats)
            if lvls:
                q += " " + " ".join(f"LEVEL:{l}" for l in lvls)
            qv = vec.transform([q])                  # (1, V)
            sims = (mat @ qv.T).toarray().ravel()    # (N,)
            if sims.size == 0:
                return []
            idxs = np.argsort(-sims)[:top_k]
            return [(int(i), float(sims[i])) for i in idxs if sims[i] > 0.0]

    def _embedding_hits(self, query: str, top_k: int) -> List[Tuple[int, float]]:
        if not self.use_embeddings:
            return []
        self._ensure_trained()
        if self._emb_matrix is None or not getattr(self, "_rows_meta", None):
            return []
        if self._hf_model is None:
            return []
        q_vec = self._hf_model.encode([query], normalize_embeddings=True)[0]  # (D,)
        sims = (self._emb_matrix @ q_vec).astype(float)                        # (N,)
        if sims.size == 0:
            return []
        idxs = np.argsort(-sims)[:top_k]
        return [(int(i), float(sims[i])) for i in idxs if sims[i] > 0.0]

    # add category/level boosts or strict filters
    def _apply_meta_bias(self, user_text: str, score: float, kb_meta: Dict) -> float:
        cats = self.user_context.get("categories")
        lvls = self.user_context.get("school_levels")
        strict = self.user_context.get("strict_category_filter", False)
        cat = (kb_meta or {}).get('category', 'Uncategorized')
        lvl = (kb_meta or {}).get('school_level', 'All')

        if strict and cats and cat not in cats:
            return -1.0
        if strict and lvls and lvl not in lvls:
            return -1.0

        if cats:
            score += self._cat_boost if cat in cats else -self._mismatch_penalty
        if lvls:
            score += self._level_boost if lvl in lvls else 0.0
        return max(min(score, 1.0), -1.0)

    # ------------------- hybrid ranking -------------------
    def _hybrid_candidates(self, user_question: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        norm_q = self.normalize_text(user_question)
        kb_row = self.knowledge_base.get(norm_q)
        if kb_row:
            return [(kb_row["answer"], 0.999, {"q": kb_row["original_question"], "meta": kb_row.get("meta", {}), "exact": True})]
        self._ensure_trained()
        top_k = self._top_k
        tfidf_hits = self._tfidf_search(user_question, top_k) or []
        emb_hits = (self._embedding_hits(user_question, top_k) if self.use_embeddings else []) or []
        pool_indices = {i for i, _ in tfidf_hits} | {i for i, _ in emb_hits}
        classic_ans, classic_sc = super().find_best_match(user_question)
        classic_best = (classic_ans, float(classic_sc), {"q": "classic_best"}) if (classic_ans and classic_sc > 0) else None
        results: List[Tuple[str, float, Dict[str, Any]]] = []
        if not pool_indices:
            if classic_best:
                ans_c, sc_c, dbg_c = classic_best
                classic_w = max(0.0, 1.0 - self._emb_weight - self._tfidf_weight)
                results.append((ans_c, float(classic_w * sc_c + 0.05), dbg_c))
                return results
            return []
        user_norm = self.normalize_text(user_question)
        user_kw = set(self.extract_keywords(user_question))
        tfidf_map = dict(tfidf_hits)
        emb_map = dict(emb_hits)
        for idx in pool_indices:
            if not getattr(self, "_rows_meta", None) or idx < 0 or idx >= len(self._rows_meta):
                continue
            meta_row = self._rows_meta[idx]
            data = self.knowledge_base.get(meta_row["norm_q"])
            if not data:
                continue
            tfidf_cos = tfidf_map.get(idx, 0.0)
            emb_cos = emb_map.get(idx, 0.0)
            jacc = self.calculate_text_similarity(user_norm, meta_row["norm_q"])
            stored_kw = set(data["keywords"])
            kw_score = (len(user_kw & stored_kw) / max(len(user_kw), len(stored_kw), 1)) if (user_kw and stored_kw) else 0.0
            classic = (jacc * 0.7) + (kw_score * 0.3)
            emb_w = self._emb_weight if self.use_embeddings else 0.0
            tfidf_w = self._tfidf_weight
            classic_w = max(0.0, 1.0 - emb_w - tfidf_w)
            hybrid = (emb_w * emb_cos) + (tfidf_w * tfidf_cos) + (classic_w * classic)
            hybrid = self._apply_meta_bias(user_question, hybrid, data.get("meta", {}))
            if hybrid < 0:
                continue
            results.append((data["answer"], float(hybrid), {"cosine_tfidf": tfidf_cos, "cosine_emb": emb_cos, "classic": classic, "q": data["original_question"], "meta": data.get("meta", {})}))
        if classic_best:
            ans_c, sc_c, dbg_c = classic_best
            if not any(ans_c == r[0] for r in results):
                classic_w = max(0.0, 1.0 - self._emb_weight - self._tfidf_weight)
                results.append((ans_c, float(classic_w * sc_c + 0.05), dbg_c))
        if self.use_cross_encoder and results:
            self._lazy_load_ce_model()
            if self._ce_model is not None:
                pairs = []
                for ans, _, dbg in results:
                    m = dbg.get("meta", {})
                    cat = m.get("category", "Uncategorized")
                    lvl = m.get("school_level", "All")
                    cand_text = f'{dbg.get("q","")} || {ans} || CAT:{cat} LEVEL:{lvl}'
                    pairs.append((user_question, cand_text))
                try:
                    ce_scores = self._ce_model.predict(pairs)
                    ce_scores = np.asarray(ce_scores, dtype=np.float32)
                    if ce_scores.size > 0:
                        ce_min, ce_max = float(ce_scores.min()), float(ce_scores.max())
                        ce_norm = (ce_scores - ce_min) / (ce_max - ce_min) if ce_max > ce_min else np.zeros_like(ce_scores)
                        results = [(ans, float(0.6 * ce + 0.4 * hyb), dbg) for (ans, hyb, dbg), ce in zip(results, ce_norm.tolist())]
                except Exception as e:
                    print(f" Cross-encoder scoring failed: {e}")
        results.sort(key=lambda x: x[1], reverse=True)
        return results
