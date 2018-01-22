from src.dl_tensorflow.model import SimpleRNN
import unittest
import numpy as np


class TestTensorflowSimpleRNN(unittest.TestCase):

    def test_state_storage(self):
        model = SimpleRNN(input_size=5, hidden_size=2, num_layers=3, num_classes=2)

        batch_size = 64

        # Create some random states
        states = [model.initial_state() for _ in range(batch_size)]

        # Convert representation that can be recognized by tensorflow
        imported_states = model.import_state(states)

        # Get representation for each sample that can be later stored
        exported_states = model.export_state(imported_states)

        # Again convert into tensorflow representation
        imported_states2 = model.import_state(exported_states)

        self.assertTrue(len(states) == len(exported_states))
        # s1 and s2 should correspond to the same sample
        for s1, s2 in zip(states, exported_states):
            # Each sample has multiple states (different layers and (h,c) states)
            # Assert that corresponding states are equal
            self.assertTrue(len(s1) == len(s2))
            for e1, e2 in zip(s1, s2):
                self.assertTrue(np.array_equal(e1, e2))

        # Imported states should be formatted as n-tuple (n layers) of tuples with (c,h) states
        self.assertTrue(len(imported_states) == len(imported_states2))
        for layer1, layer2 in zip(imported_states, imported_states2):
            self.assertTrue(len(layer1) == len(layer2))
            for state1, state2 in zip(layer1, layer2):
                self.assertTrue(np.array_equal(state1, state2))

if __name__ == '__main__':
    unittest.main()
