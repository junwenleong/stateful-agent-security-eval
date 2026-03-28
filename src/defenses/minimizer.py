"""TF-IDF relevance filter defense."""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.defenses.base import Defense, DefenseLog


class Minimizer(Defense):
    """TF-IDF relevance filter. Strips context not relevant to user query.

    Lightweight proxy — NOT the Bhagwatkar LLM-based firewall.

    KNOWN LIMITATION: TF-IDF operates on token overlap. Encoded content
    (Braille, Base64) will have near-zero similarity to English queries,
    causing the Minimizer to strip encoded payloads as "irrelevant."
    This is documented as a confound: the Minimizer may block obfuscated
    attacks for the wrong reason (irrelevance, not injection detection).
    The paper must report Minimizer-only vs Sanitizer-only ablations
    to distinguish these effects.
    """

    def __init__(self, relevance_threshold: float = 0.1):
        self.relevance_threshold = relevance_threshold

    def apply(self, user_input: str, context: dict | None = None) -> tuple[str, DefenseLog]:
        """Score each context chunk by TF-IDF similarity to user query.
        Remove chunks below threshold. If all would be removed, keep the highest-scoring one.
        """
        # Split into chunks on sentence or paragraph boundaries
        chunks = [c.strip() for c in re.split(r"\. |\n", user_input) if c.strip()]

        if len(chunks) <= 1:
            return user_input, DefenseLog(
                original_input=user_input,
                modified_input=user_input,
                modifications=[],
            )

        # Use the first chunk as the query
        query = chunks[0]
        documents = chunks[1:]

        try:
            vectorizer = TfidfVectorizer()
            all_texts = [query] + documents
            tfidf_matrix = vectorizer.fit_transform(all_texts)
            query_vec = tfidf_matrix[0]
            doc_vecs = tfidf_matrix[1:]
            scores = cosine_similarity(query_vec, doc_vecs)[0]
        except ValueError:
            # Vectorizer can fail on degenerate inputs (e.g. all stop words)
            return user_input, DefenseLog(
                original_input=user_input,
                modified_input=user_input,
                modifications=[],
            )

        retained = []
        removed = []
        for i, (chunk, score) in enumerate(zip(documents, scores)):
            if score >= self.relevance_threshold:
                retained.append(chunk)
            else:
                removed.append((i, chunk, float(score)))

        # Always keep at least the highest-scoring chunk
        if not retained:
            best_idx = int(np.argmax(scores))
            retained = [documents[best_idx]]
            removed = [(i, c, float(s)) for i, (c, s) in enumerate(zip(documents, scores)) if i != best_idx]

        modifications = []
        for idx, chunk, score in removed:
            modifications.append(
                f"Removed chunk (score={score:.3f} < threshold={self.relevance_threshold}): "
                f"{chunk[:60]}{'...' if len(chunk) > 60 else ''}"
            )

        # Reconstruct: always include the query (first chunk) + retained
        result_chunks = [query] + retained
        modified = ". ".join(result_chunks)

        return modified, DefenseLog(
            original_input=user_input,
            modified_input=modified,
            modifications=modifications,
        )
