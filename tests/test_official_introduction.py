import unittest
from assistive.command_router import CommandRouter, OFFICIAL_INTRODUCTION
from assistive.vision_engine import VisionEngine

class TestOfficialIntroduction(unittest.TestCase):
    def setUp(self):
        self.router = CommandRouter()
        self.engine = VisionEngine(data_dir="data")

    def test_official_introduction_exact_text(self):
        """ Verify official introduction contains exact required sections, emojis, and phrases """
        self.assertTrue(OFFICIAL_INTRODUCTION.startswith("Hi! I’m SG CUBE — your AI vision companion. 👋"))
        self.assertTrue(OFFICIAL_INTRODUCTION.endswith("See. Understand. Remember. Assist. That’s SG CUBE."))
        
        # Verify key required sentences from official spec
        self.assertIn("I can see, listen, remember, and help you understand the world around you.", OFFICIAL_INTRODUCTION)
        self.assertIn("I can recognize faces, read text, find objects, detect currency, remember useful information, and talk with you naturally.", OFFICIAL_INTRODUCTION)
        self.assertIn("Basically, I’m like a helpful friend… except I never ask, ‘Where did I keep my phone?’ while holding it in my hand. 😂", OFFICIAL_INTRODUCTION)
        self.assertIn("See. Understand. Remember. Assist. That’s SG CUBE.", OFFICIAL_INTRODUCTION)

    def test_command_routing_introduce_queries(self):
        """ Verify natural language self-introduction queries route to INTRODUCE intent """
        queries = [
            "introduce yourself",
            "Introduce yourself",
            "Can you please introduce yourself",
            "Hey SG CUBE introduce yourself",
            "who are you",
            "Who are you?",
            "tell me about yourself",
            "Tell me about yourself",
            "give your introduction",
            "give an introduction",
            "give me your introduction",
            "what is sg cube",
            "What is SG CUBE?",
            "what are you",
            "give me your intro",
            "self introduction",
            "introduction",
            "introduce"
        ]
        for q in queries:
            route = self.router.route_intent(q)
            self.assertEqual(route["intent"], "INTRODUCE", f"Failed for query: '{q}'")

    def test_vision_engine_process_user_speech_query_introduce(self):
        """ Verify vision engine returns exact official introduction upon query """
        resp = self.engine.process_user_speech_query("introduce yourself")
        self.assertEqual(resp, OFFICIAL_INTRODUCTION)

    def test_vision_engine_who_are_you_query(self):
        """ Verify 'who are you' query returns exact official introduction """
        resp = self.engine.process_user_speech_query("who are you")
        self.assertEqual(resp, OFFICIAL_INTRODUCTION)

if __name__ == "__main__":
    unittest.main()
