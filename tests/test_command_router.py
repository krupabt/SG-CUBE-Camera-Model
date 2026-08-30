import unittest
from assistive.command_router import CommandRouter

class TestCommandRouter(unittest.TestCase):
    def setUp(self):
        self.router = CommandRouter()

    def test_routing_intents(self):
        # Face Remember
        r1 = self.router.route_intent("Remember this person as Rahul")
        self.assertEqual(r1["intent"], "FACE_REMEMBER")
        self.assertEqual(r1["params"]["name"], "Rahul")

        # Memory Save
        r_mem = self.router.route_intent("Remember that my favorite color is blue")
        self.assertEqual(r_mem["intent"], "MEMORY_SAVE")
        self.assertEqual(r_mem["params"]["key"], "favorite color")
        self.assertEqual(r_mem["params"]["fact"], "My favorite color is blue.")

        # Face Identify
        r2 = self.router.route_intent("Who is in front of me?")
        self.assertEqual(r2["intent"], "FACE_IDENTIFY")

        # Currency
        r3 = self.router.route_intent("How much money is this?")
        self.assertEqual(r3["intent"], "CURRENCY")

        # OCR
        r4 = self.router.route_intent("Read the sign")
        self.assertEqual(r4["intent"], "OCR")

        # Environment
        r5 = self.router.route_intent("What is around me?")
        self.assertEqual(r5["intent"], "ENVIRONMENT")

        # Object Search
        r6 = self.router.route_intent("Find my phone")
        self.assertEqual(r6["intent"], "OBJECT_SEARCH")
        self.assertEqual(r6["params"]["object_name"], "phone")

        # Settings
        r7 = self.router.route_intent("Turn greetings off")
        self.assertEqual(r7["intent"], "SETTINGS")
        self.assertFalse(r7["params"]["value"])

if __name__ == "__main__":
    unittest.main()
