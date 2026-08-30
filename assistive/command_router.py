import re
from typing import Dict, Optional, Tuple

OFFICIAL_INTRODUCTION = """Hi! I’m SG CUBE — your AI vision companion. 👋

I can see, listen, remember, and help you understand the world around you.

I can recognize faces, read text, find objects, detect currency, remember useful information, and talk with you naturally.

Basically, I’m like a helpful friend… except I never ask, ‘Where did I keep my phone?’ while holding it in my hand. 😂

See. Understand. Remember. Assist. That’s SG CUBE."""

class CommandRouter:
    """
    Intelligent Intent Parser & Command Router for Voice Queries.
    Maps natural language spoken commands to specialized perception & memory routines.
    """

    def extract_memory_key_and_fact(self, text: str) -> Tuple[str, str]:
        """
        Intelligently extracts search key phrase and clean fact text.
        Examples:
        'Remember that my favorite color is blue' -> key: 'favorite color', fact: 'My favorite color is blue.'
        'Remember my favorite fruit is apple.' -> key: 'favorite fruit', fact: 'My favorite fruit is apple.'
        'Save this information' -> key: 'contextual', fact: ''
        'Save this info' -> key: 'contextual', fact: ''
        'Save this' -> key: 'contextual', fact: ''
        'Remember this' -> key: 'contextual', fact: ''
        'Save this: meeting at 3pm' -> key: 'meeting', fact: 'Meeting at 3pm.'
        """
        clean = text.strip()
        # Strip any combination of leading command prefixes
        body = re.sub(
            r'^(?:please\s+)?(?:remember|save|store)\s+(?:(?:that|this|info|information|the\s+fact\s+that|a\s+note\s+that|note\s+that|to\s+memory)\s+)*(?::\s*)?',
            '', clean, flags=re.IGNORECASE
        ).strip()
        body_cleaned = re.sub(r'[^\w\s]', '', body).strip().lower()

        # If bare command without specific inline fact (e.g. 'save this', 'save info', 'remember this', 'save this information')
        if not body or body_cleaned in ["", "this", "that", "it", "info", "information", "detail", "details", "memory", "note", "notes", "this info", "this information", "this detail", "this note"]:
            return "contextual", ""

        # Strip any leading 'this:' or 'that:' or 'info:' if present
        body_clean_val = re.sub(r'^(?:this|that|info|information)\s*:\s*', '', body, flags=re.IGNORECASE).strip()
        body_lower = body_clean_val.lower()

        # Subject-verb pattern: "... is ..." or "... are ..."
        if " is " in body_lower or " are " in body_lower:
            match = re.search(r'(.+?)\s+(?:is|are)\s+(.+)', body_clean_val, re.IGNORECASE)
            if match:
                subj = match.group(1).strip()
                key = re.sub(r'^(?:my|the|a|an)\s+', '', subj, flags=re.IGNORECASE).strip()
                key = re.sub(r'[^\w\s]', '', key).strip().lower().replace("  ", " ")
                val = body_clean_val[0].upper() + body_clean_val[1:]
                if not val.endswith('.'):
                    val += '.'
                return key if key else body_lower, val

        key_words = [w for w in re.sub(r'[^\w\s]', '', body_lower).split() if w not in ["that", "my", "the", "a", "an", "this", "it", "info", "information", "is", "are"]]
        key = " ".join(key_words[:2]) if key_words else body_cleaned
        val = body_clean_val[0].upper() + body_clean_val[1:] if body_clean_val else clean
        if not val.endswith('.'):
            val += '.'
        return key, val

    def route_intent(self, text: str) -> Dict:
        """
        Parses user speech query and returns intent dict:
        {
          'intent': str,
          'target': Optional[str],
          'params': Dict
        }
        """
        if not text:
            return {"intent": "GENERAL", "target": None, "params": {}}

        clean_text = text.strip().lower()

        # 1. Face Memory Enrollment ("Remember this person as Rahul", "Save this face as Sahana", "Save this face", "Enroll face as Alex")
        if any(w in clean_text for w in ["person", "face", "this person", "this face", "face as", "save face", "remember face"]):
            if any(p in clean_text for p in ["remember", "save", "enroll", "store"]):
                remember_face_match = re.search(
                    r'(?:remember|save|enroll|store)\s+(?:this\s+)?(?:person|face|them|him|her)?\s*(?:as)?\s*([a-zA-Z0-9_\s]*)',
                    clean_text
                )
                if remember_face_match:
                    name_raw = remember_face_match.group(1).strip()
                    name_str = re.sub(r'^(?:person|face|as|this\s+person|this\s+face|this)\s*', '', name_raw, flags=re.IGNORECASE).strip().title()
                    if name_str and name_str.lower() not in ["that", "this", "me", "my", "it", "face", "person", ""]:
                        return {"intent": "FACE_REMEMBER", "target": name_str, "params": {"name": name_str}}
                    else:
                        # Bare face enrollment request (e.g. "Save this face", "Remember this face")
                        return {"intent": "FACE_REMEMBER", "target": None, "params": {"name": None}}

        # 2. Clear All Persistent Memories ("Clear all memories", "Forget everything you know about me")
        if any(p in clean_text for p in ["forget everything", "clear all memories", "clear my memories", "erase all memories"]):
            return {"intent": "MEMORY_CLEAR", "target": None, "params": {}}

        # 3. List / Show Memories ("Show my memories", "List all memories", "What memories do you have")
        if any(p in clean_text for p in ["show my memories", "list all memories", "what do you remember about me", "show stored memories", "list memories"]):
            return {"intent": "MEMORY_LIST", "target": None, "params": {}}

        # 4. Explicit & Contextual Memory Save ("Save this", "Save this information", "Remember this", "Remember my favorite color is blue")
        is_save_cmd = (
            clean_text.startswith("remember") or
            clean_text.startswith("please remember") or
            clean_text.startswith("save") or
            clean_text.startswith("please save") or
            clean_text.startswith("store") or
            "save this" in clean_text or
            "save that" in clean_text or
            "save info" in clean_text or
            "save information" in clean_text or
            clean_text.startswith("my name is")
        )
        if is_save_cmd:
            # Exclude recall question forms ("What do you remember", "Do you remember", "Who is")
            if not any(w in clean_text for w in ["do you", "can you", "what", "who", "where", "how", "person", "face"]):
                key, fact_val = self.extract_memory_key_and_fact(text)
                return {"intent": "MEMORY_SAVE", "target": key, "params": {"fact": fact_val, "key": key}}

        # 5. Forget Specific Memory or Face ("Forget that Rahul is my friend", "Forget my favorite color", "Forget Rahul")
        forget_match = re.search(r'forget (?:that|my|the|face of|person)?\s*(.+)', clean_text)
        if forget_match:
            target_str = forget_match.group(1).strip()
            if "all faces" in target_str or "everyone" in target_str:
                return {"intent": "FACE_FORGET_ALL", "target": None, "params": {}}
            elif "face" in clean_text or "person" in clean_text or target_str.title() in ["Rahul", "Sarah", "Mom", "Dad", "Alex", "Sahana"]:
                return {"intent": "FACE_FORGET", "target": target_str, "params": {"name": target_str}}
            else:
                key = re.sub(r'^(?:that|my|the|a|an)\s+', '', target_str, flags=re.IGNORECASE).strip().lower()
                return {"intent": "MEMORY_FORGET", "target": key, "params": {"key": key}}

        # 6. Self-Introduction Query ("Introduce yourself", "Who are you?", "Tell me about yourself", "Give your introduction", "What is SG CUBE?")
        if any(p in clean_text for p in [
            "introduce yourself", "introduce you", "tell me about yourself", "who are you",
            "give your introduction", "give an introduction", "give me your introduction",
            "what is sg cube", "what are you", "give me your intro", "give your intro",
            "self introduction", "tell me who you are", "who is sg cube"
        ]) or clean_text in ["introduce", "introduction", "who are you", "who is sg cube", "what is sg cube"]:
            return {"intent": "INTRODUCE", "target": None, "params": {}}

        # 7. Memory Recall Query ("What is my favorite color?", "What's my name?", "Do you remember...", "Who is Rahul")
        if any(p in clean_text for p in ["do you remember", "what is my", "what's my", "do you know my", "who is", "what do you know about", "what do you remember", "tell me what you remember", "do you know"]):
            # Exclude standard visual queries ("Who is in front of me")
            if not any(w in clean_text for w in ["in front of me", "around me", "this person", "this face"]):
                return {"intent": "MEMORY_RECALL", "target": clean_text, "params": {"query": clean_text}}

        # 7. Face Recognition Query ("Who is in front of me?")
        if any(p in clean_text for p in ["who is in front of me", "who is this", "do you know this person", "who just entered", "do you recognize"]):
            return {"intent": "FACE_IDENTIFY", "target": None, "params": {}}

        # 8. Face Listing
        if any(p in clean_text for p in ["who do you know", "list people", "who do you remember", "list all faces", "show how many people"]):
            return {"intent": "FACE_LIST", "target": None, "params": {}}

        # 9. Currency Query
        if any(p in clean_text for p in ["how much money", "what currency", "how much is this", "what denomination", "rupee note", "banknote"]):
            return {"intent": "CURRENCY", "target": None, "params": {}}

        # 10. OCR / Text Reading Query
        if any(p in clean_text for p in ["read this", "read the sign", "read text", "read document", "read label", "what does this say"]):
            return {"intent": "OCR", "target": None, "params": {}}

        # 11. Environment Query
        if any(p in clean_text for p in ["what is around me", "describe the environment", "describe my surroundings", "what is in front of me"]):
            return {"intent": "ENVIRONMENT", "target": None, "params": {}}

        # 12. Object Search Query ("Find my phone", "Where is my bottle?")
        find_match = re.search(r'(?:find|where is|can you see|is there a|look for)\s*(?:my|a|the)?\s*([a-zA-Z0-9_\s]+)', clean_text)
        if find_match and not any(w in clean_text for w in ["who", "read", "money"]):
            obj_name = find_match.group(1).strip()
            return {"intent": "OBJECT_SEARCH", "target": obj_name, "params": {"object_name": obj_name}}

        # 13. Safety / Hazard Query
        if any(p in clean_text for p in ["are there stairs", "is it safe", "any obstacles", "anything dangerous"]):
            return {"intent": "SAFETY", "target": None, "params": {}}

        # 14. Settings Toggles
        if "turn greetings off" in clean_text or "stop greetings" in clean_text:
            return {"intent": "SETTINGS", "target": "greetings", "params": {"setting": "greeting_enabled", "value": False}}

        if "turn greetings on" in clean_text or "start greetings" in clean_text:
            return {"intent": "SETTINGS", "target": "greetings", "params": {"setting": "greeting_enabled", "value": True}}

        if "turn environment monitoring on" in clean_text or "continuous monitoring on" in clean_text:
            return {"intent": "SETTINGS", "target": "continuous", "params": {"setting": "environment_monitor_enabled", "value": True}}

        if "turn environment monitoring off" in clean_text or "continuous monitoring off" in clean_text:
            return {"intent": "SETTINGS", "target": "continuous", "params": {"setting": "environment_monitor_enabled", "value": False}}

        # 15. Color Identification Query ("What color is this shirt?", "Tell me the colors in front of me")
        if any(p in clean_text for p in ["what color", "tell me the color", "identify color", "color of this"]):
            return {"intent": "COLOR_IDENTIFY", "target": None, "params": {}}

        # 16. Light Level / Room Illumination Check ("Are the lights on?", "Is it dark in here?")
        if any(p in clean_text for p in ["are the lights on", "is it dark", "check light level", "how is the light", "is the room lit"]):
            return {"intent": "LIGHT_LEVEL_CHECK", "target": None, "params": {}}

        # 17. Product / Barcode / Expiration Date Scan ("Scan this product", "Read expiration date", "Scan barcode")
        if any(p in clean_text for p in ["scan product", "scan barcode", "scan qr", "read expiration", "expiry date", "what medicine", "is this medicine"]):
            return {"intent": "PRODUCT_SCAN", "target": None, "params": {}}

        # 18. Sleep / Deactivate Command
        if any(p in clean_text for p in ["go to sleep", "stop listening", "sleep mode", "deactivate"]):
            return {"intent": "SLEEP", "target": None, "params": {}}

        # Fallback to General Gemini Live Reasoning
        return {"intent": "GENERAL", "target": None, "params": {}}
