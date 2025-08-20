# smart_retrieval.py
# PURPOSE: Adds "smart" semantic retrieval on top of CSVChatbot using
# TF-IDF char n-grams + your original Jaccard/keyword signal (hybrid).
# Keeps your original behavior via fallback and a toggle.

from __future__ import annotations
import numpy as np
from typing import Any, Dict, List, Optional, Tuple
from csv_chatbot import CSVChatbot


class SmartCSVChatbot(CSVChatbot):
    def __init__(self):
        super().__init__()

        # toggles / weights
        self.use_hybrid = True
        self.use_embeddings = False  # embeddings handled in HF subclass
        self._top_k = 5

        # NEW: weight for blending TF-IDF cosine with classic signals
        self._cosine_weight = 0.6
        self._min_conf = 0.15

        # TF-IDF index state
        self._vectorizer = None
        self._matrix = None
        self._rows_meta: List[Dict[str, str]] = []

    def _urgent_cheatsheet(self, text: str) -> Optional[str]:
        t = text.lower()
    
        if any(k in t for k in ("can't join", "cant join", "join meeting", "zoom", "teams", "video call", "meeting link")):
            return (
                "Quick fix:\n"
                "1) Reopen the meeting link (copy + paste into browser).\n"
                "2) Try the other client: browser ↔ desktop app.\n"
                "3) Check mic/camera permissions (browser/site settings).\n"
                "4) Switch network (Wi-Fi → phone hotspot) or VPN off.\n"
                "5) Restart the app; sign out/in.\n"
                "6) If it still fails: ask host to re-invite or provide dial-in."
            )
        # Network/Wi-Fi down
        if any(k in t for k in ("wifi", "wi-fi", "network down", "no internet")):
            return (
                "Quick fix:\n"
                "1) Toggle Wi-Fi off/on; forget & re-join the SSID.\n"
                "2) Try a different network; reboot access point if local.\n"
                "3) Check airplane/VPN; disable proxy.\n"
                "4) Get IP via DHCP (no static); renew lease."
            )
        return None

    # Ensure semantic index is (re)built whenever KB changes
    def load_csv_file(self, file_path: str, save_to_config: bool = True) -> bool:
        ok = super().load_csv_file(file_path, save_to_config=save_to_config)
        if ok:
            self._train_vectors()
        return ok

    # ---------- SMART RETRIEVAL CORE ----------
    def _train_vectors(self) -> None:
        """Build a TF-IDF character n-gram index over 'question || answer'."""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
        except Exception:
            # If scikit-learn is missing, disable hybrid gracefully.
            self._vectorizer, self._matrix, self._rows_meta = None, None, []
            return

        texts: List[str] = []
        meta: List[Dict[str, str]] = []
        for norm_q, data in self.knowledge_base.items():
            q = data['original_question']
            a = data['answer']
            texts.append(f"{q} || {a}")
            meta.append({'q': q, 'a': a, 'norm_q': norm_q})

        if not texts:
            self._vectorizer, self._matrix, self._rows_meta = None, None, []
            return

        self._vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=(3, 5),
            lowercase=True,
            strip_accents='unicode',
            min_df=1
        )
        self._matrix = self._vectorizer.fit_transform(texts)
        self._rows_meta = meta

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
            qv = vec.transform([q])                     # (1, V)
            sims = (mat @ qv.T).toarray().ravel()       # (N,)
            if sims.size == 0:
                return []
            idxs = np.argsort(-sims)[:top_k]
            return [(int(i), float(sims[i])) for i in idxs if sims[i] > 0.0]


    def _hybrid_candidates(self, user_question: str) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        Blend TF-IDF cosine with your Jaccard/keyword approach.
        Returns list of (answer, hybrid_score, debug).
        """
        norm_q = self.normalize_text(user_question)
        kb_row = self.knowledge_base.get(norm_q)
        if kb_row:
            return [(kb_row['answer'], 0.999, {
                'q': kb_row['original_question'],
                'meta': kb_row.get('meta', {}),
                'exact': True
            })]

        # make sure index exists
        if self._vectorizer is None or self._matrix is None or not self._rows_meta:
            self._train_vectors()

        # 1) Semantic hits
        tfidf_hits = self._tfidf_search(user_question, self._top_k)

        results: List[Tuple[str, float, Dict[str, Any]]] = []
        user_norm = self.normalize_text(user_question)
        user_kw = set(self.extract_keywords(user_question))

        for idx, cos in tfidf_hits:
            meta = self._rows_meta[idx]
            data = self.knowledge_base.get(meta['norm_q'])
            if not data:
                continue

            # Your original signals
            jacc = self.calculate_text_similarity(user_norm, meta['norm_q'])
            stored_kw = set(data['keywords'])
            if user_kw and stored_kw:
                kw_overlap = len(user_kw & stored_kw)
                kw_max = max(len(user_kw), len(stored_kw))
                kw_score = kw_overlap / kw_max if kw_max > 0 else 0.0
            else:
                kw_score = 0.0

            classic = (jacc * 0.7) + (kw_score * 0.3)

            # Hybrid fusion
            hybrid = (self._cosine_weight * cos) + ((1 - self._cosine_weight) * classic)

            results.append((
                data['answer'],
                float(hybrid),
                {'cosine': float(cos), 'classic': float(classic),
                 'jaccard': float(jacc), 'kw': float(kw_score), 'q': data['original_question']}
            ))

        # Fallback to original matcher if no semantic results
        if not results:
            ans, sc = super().find_best_match(user_question)
            if ans:
                results = [(ans, sc, {'fallback': True})]

        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _synthesize(self, hits):
        if not hits:
            return None, 0.0
        
        best_ans, best_score, best_dbg = hits[0]
        best_cat = (best_dbg or {}).get("meta", {}).get("category")
        
        bullets = []
        for ans, sc, dbg in hits[1:]:
            if sc < best_score - 0.06:
                continue
            if best_cat:
                cat = (dbg or {}).get("meta", {}).get("category")
                if cat and cat != best_cat:
                    continue
            if ans and ans != best_ans:
                first = ans.strip().split("\n")[0]
                bullets.append(first if len(first) <= 160 else (first[:157] + "..."))
            if len(bullets) == 2:
                break
                
        if bullets:
            fused = f"{best_ans}\n\nAlso note:\n- " + "\n- ".join(bullets)
            return fused, best_score
            
        return best_ans, best_score

    def _progression_hint(self, text: str) -> Optional[str]:
        t = text.lower()
        if "still" in t and any(k in t for k in ("laptop", "computer", "pc")):
            return ("Next steps: 1) Try Safe Mode (or hold Shift while rebooting on Windows). "
                    "2) Remove recent USB devices. 3) Run a malware scan. "
                    "4) Check disk space; if <5%, free storage. 5) If it persists, open an IT ticket.")
        if "printer" in t and "still" in t:
            return ("Next steps: 1) Print network config/test page. 2) Re-add the printer by IP. "
                    "3) Update the driver. 4) Power cycle router + printer. "
                    "5) If school-managed, contact IT to check print server.")
        return None

    # ---------- PUBLIC: smarter response ----------
    def get_response(self, user_input: str) -> str:
        """Use hybrid retrieval (toggleable). Falls back to original behavior."""
        if not self.use_hybrid:
            return super().get_response(user_input)

        if not user_input.strip():
            return "Please ask me something!"

        # Sentiment + intent as in base class
        sentiment_data = self.sentiment_analyzer.analyze_sentiment(user_input)
        intent = self.detect_intent(user_input)

        if intent in self.responses and intent != 'question':
            import random
            base_response = random.choice(self.responses[intent])
            if intent == 'greeting' and sentiment_data['sentiment'] == 'negative':
                return "Hello! I can see you might be having some issues. How can I help you with IT support today?"
            elif intent == 'greeting' and sentiment_data['is_urgent']:
                return "Hi there! I understand you need urgent help. What IT issue can I assist you with right away?"
            return base_response

        # Hybrid retrieval
        hits = self._hybrid_candidates(user_input) or []
        answer, confidence = (None, 0.0)
        if hits:
            syn = self._synthesize(hits)
            if isinstance(syn, tuple):
                answer, confidence = syn
            elif syn:
                # use top hit score if available, at least min_conf
                answer = syn
                confidence = max(hits[0][1], self._min_conf)
            else:
                answer, confidence = hits[0][0], hits[0][1]
        else:
            answer, confidence = (None, 0.0)
            
        sentiment_prefix = self.get_sentiment_response_prefix(sentiment_data)
        if answer and confidence >= self._min_conf:
            return f"{sentiment_prefix}\n\n{answer}" if sentiment_prefix else answer

        # Compute hint lazily (only if we didn't return above)
        try:
            hint = self._progression_hint(user_input)
        except AttributeError:
            hint = None
            
        # Fall through to your original no-match handling
        if sentiment_data['is_frustrated']:
            return "I understand your frustration. I don't have a direct answer for that, but could you rephrase your question? I want to make sure I help you properly."
        elif sentiment_data['is_urgent']:
            return "I want to help you with this urgent issue, but I need more details. Could you describe the problem differently so I can find the best solution?"
        elif hint:
            return hint
        elif sentiment_data['is_confused']:
            return "No problem! I don't have that exact information, but let's try a different approach. Can you tell me more about what you're trying to do?"
        else:
            import random
            return random.choice(self.responses['default'])
