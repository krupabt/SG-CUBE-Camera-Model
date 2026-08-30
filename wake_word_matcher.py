import re
import difflib
from typing import Tuple, Dict

class WakeWordMatcher:
    """
    Dedicated Two-Stage Acoustic & Phonetic Wake-Word Matcher for SG CUBE.
    High Recall for valid phonetic/ASR candidates + Low False Activation for normal conversation.
    """

    HOTWORD_PHRASES = {
        "sg cube", "hey sg cube", "sg", "s g", "cube", "sg cub", "sg cue",
        "sgq", "s g q", "ess gee", "ess gee cube", "es gee", "es gee cube",
        "esg", "esg cube", "hey sg", "hey s g", "hey ess gee", "hey ess gee cube",
        "heysgcube", "sgcube", "ksg q", "sg q", "kyube", "hi sg cube", "hi sg",
        "ok sg cube", "ok sg", "hey cube", "hi cube", "hello sg cube", "hello sg",
        "hey sgcube", "s g cube", "ess-gee-cube", "es-gee-cube", "s.g. cube",
        "cuube", "kewb", "esse gee cube"
    }

    SG_TOKENS = {"sg", "s g", "ess gee", "es gee", "esgee", "essgee", "ksg", "esg", "s", "g"}
    CUBE_TOKENS = {"cube", "kyube", "q", "cub", "cue", "cuube", "kewb"}
    PREFIX_TOKENS = {"hey", "hi", "ok", "okay", "hello", "yo"}

    NON_WAKE_TERMS = {
        "facebook", "execute", "play music", "good morning everyone",
        "what are you doing", "call my friend", "open the file", "look at this",
        "tell me a joke", "how are you", "what is the weather", "turn on the lights",
        "google", "youtube", "computer", "everyone", "conversation", "my name is",
        "one two three", "testing one two", "random sentence"
    }

    @classmethod
    def normalize_text(cls, text: str) -> str:
        if not text:
            return ""
        norm = text.lower()
        norm = re.sub(r'[^a-z0-9\s]', ' ', norm)
        norm = re.sub(r'\s+', ' ', norm).strip()
        return norm

    @classmethod
    def _phonetic_similarity(cls, text: str, target: str) -> float:
        """ Calculates character-level sequence similarity """
        return difflib.SequenceMatcher(None, text, target).ratio()

    @classmethod
    def evaluate(cls, raw_text: str) -> Tuple[bool, float, str, str, Dict]:
        """
        Evaluates input text against dedicated acoustic & phonetic hotword confidence model.
        Returns tuple: (is_match: bool, confidence_score: float, reason: str, norm_text: str, debug_dict: dict)
        """
        if not raw_text or not raw_text.strip():
            return False, 0.0, "NO_SPEECH", "", {}

        norm_text = cls.normalize_text(raw_text)
        tokens = norm_text.split()
        word_count = len(tokens)

        debug_info = {
            "raw": raw_text.strip(),
            "normalized": norm_text,
            "word_count": word_count
        }

        # 1. Check for explicit negative conversational phrases
        for non_wake in cls.NON_WAKE_TERMS:
            if non_wake in norm_text:
                # If exact hotword phrase is also present at start/end
                if norm_text in cls.HOTWORD_PHRASES:
                    return True, 0.95, "HOTWORD_EXACT_OVERRIDE", norm_text, debug_info
                return False, 0.0, "NEGATIVE_PHRASE_DETECTED", norm_text, debug_info

        # Single word common conversation reject (e.g. "hello", "facebook", "music", "play")
        if norm_text in {"hello", "hi", "hey", "facebook", "play", "music", "good", "morning", "night", "yes", "no", "okay", "stop"}:
            return False, 0.0, "COMMON_SINGLE_WORD_REJECT", norm_text, debug_info

        # 2. Exact match against acoustic/phonetic hotword dictionary
        if norm_text in cls.HOTWORD_PHRASES:
            return True, 0.98, "HOTWORD_EXACT_MATCH", norm_text, debug_info

        # 3. Combined SG + CUBE token presence in compact utterance (<= 4 words)
        if word_count <= 4:
            has_sg = any(sg in norm_text for sg in ("sg", "s g", "ess gee", "es gee", "esg", "ksg", "esgee"))
            has_cube = any(cb in norm_text for cb in ("cube", "cub", "cue", "kyube", "cuube", "sgq"))

            if has_sg and has_cube:
                return True, 0.92, "HOTWORD_SG_CUBE_COMBO", norm_text, debug_info

            # Partial token combinations: "hey sg", "hi sg", "ok sg", "hey cube"
            if word_count <= 3:
                has_prefix = any(p in tokens for p in cls.PREFIX_TOKENS)
                if has_prefix and (has_sg or has_cube):
                    return True, 0.88, "HOTWORD_PREFIX_COMBO", norm_text, debug_info

                if has_sg and word_count <= 2:
                    # e.g. "sg", "ess gee", "esg"
                    return True, 0.85, "HOTWORD_SG_STANDALONE", norm_text, debug_info

        # 4. Phonetic similarity matching against core targets
        if word_count <= 3:
            for target in ["sg cube", "hey sg cube", "ess gee cube", "es gee cube", "sgq"]:
                sim = cls._phonetic_similarity(norm_text, target)
                if sim >= 0.80:
                    return True, float(round(sim, 2)), "HOTWORD_PHONETIC_SIMILARITY", norm_text, debug_info

        # Long utterances (> 4 words) are normal conversation unless they begin with clean wake trigger
        if word_count > 4:
            prefix_match = False
            for target in ["hey sg cube", "sg cube", "ok sg cube", "hi sg cube"]:
                if norm_text.startswith(target):
                    prefix_match = True
                    break
            if prefix_match:
                return True, 0.86, "HOTWORD_PREFIX_SENTENCE", norm_text, debug_info

        return False, 0.0, "NON_WAKE_SPEECH", norm_text, debug_info
