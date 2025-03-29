#!/usr/bin/env python3
import json
import os
import time
import serial
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EffectReviewer:
    def __init__(self):
        self.port = os.getenv("SERIAL_PORT")
        self.baud = os.getenv("SERIAL_BAUD")
        self.selections_file = "effect_selections.json"
        self.selections = self.load_selections()

        # Load effects from state.json
        with open("state.json", "r") as f:
            state = json.load(f)
            self.effects = state["effects"]

        # Connect to WLED
        self.ser = serial.Serial(self.port, self.baud, timeout=1)

    def load_selections(self):
        if os.path.exists(self.selections_file):
            with open(self.selections_file, "r") as f:
                return json.load(f)
        return {}

    def save_selections(self):
        with open(self.selections_file, "w") as f:
            json.dump(self.selections, f, indent=2)

    def send_effect(self, effect_index):
        message = {
            "tt": 0,
            "seg": [{"id": 0, "fx": effect_index}]
        }
        json_data = json.dumps(message) + '\n'
        self.ser.write(json_data.encode())

    def review_effects(self):
        print("\nWLED Effect Reviewer")
        print("===================")
        print("Type 'y' to keep an effect, 'n' to skip it, or 'q' to quit")
        print("Type 'r' to restart from the beginning")
        print("Type 's' to save and exit\n")

        for i, effect in enumerate(self.effects):
            # Skip if already reviewed
            if str(i) in self.selections:
                continue

            print(f"\nEffect {i}: {effect}")
            self.send_effect(i)

            while True:
                response = input("Keep this effect? (y/n/r/s/q): ").lower()

                if response == 'q':
                    print("\nQuitting without saving...")
                    return
                elif response == 's':
                    print("\nSaving and exiting...")
                    self.save_selections()
                    return
                elif response == 'r':
                    print("\nRestarting from beginning...")
                    self.selections = {}
                    self.save_selections()
                    return self.review_effects()
                elif response in ['y', 'n']:
                    self.selections[str(i)] = response == 'y'
                    self.save_selections()
                    break
                else:
                    print("Invalid input. Please try again.")

            # Small delay to let the effect be visible
            time.sleep(0.2)

        print("\nReview complete! All effects have been evaluated.")
        print(f"Selected effects: {sum(1 for v in self.selections.values() if v)}")
        print(f"Skipped effects: {sum(1 for v in self.selections.values() if not v)}")

    def get_selected_effects(self):
        """Returns a list of indices of selected effects"""
        return [int(i) for i, selected in self.selections.items() if selected]

    def cleanup(self):
        self.ser.close()

if __name__ == "__main__":
    reviewer = EffectReviewer()
    try:
        reviewer.review_effects()
    finally:
        reviewer.cleanup()
