"""Fixed-schedule no-validation contract, with optimizer/model stubs only."""
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from src.training import train_target_model


class NoValidationTrainingTests(unittest.TestCase):
    def test_missing_validation_cannot_enable_stopping_or_best_epoch_selection(self):
        x, y = np.ones((4, 2), dtype=np.float32), np.array([0, 1, 0, 1])
        for config in [dict(), dict(early_stopping=False), dict(restore_best=False)]:
            with self.subTest(config=config), patch('src.training.build_mlp') as build:
                with self.assertRaisesRegex(ValueError, 'Without validation'):
                    train_target_model(x, y, None, None, **config)
                build.assert_not_called()

    def test_no_validation_never_queries_heldout_data_or_restores_weights(self):
        x, y = np.ones((4, 2), dtype=np.float32), np.array([0, 1, 0, 1])
        model = MagicMock()
        model.to.return_value = model
        loss, optimizer = MagicMock(), MagicMock()
        loss.item.return_value = .4
        with patch('src.training.build_mlp', return_value=model), \
                patch('src.training.torch.optim.Adam', return_value=optimizer), \
                patch('src.training.DataLoader', return_value=[(torch.ones((4, 2)), torch.tensor(y))]), \
                patch('src.training.nn.CrossEntropyLoss', return_value=lambda *args: loss), \
                patch('src.training.evaluate_model') as evaluate, \
                patch('builtins.print'):
            result, history, metadata = train_target_model(x, y, None, None,
                early_stopping=False, restore_best=False, epochs=3, monitor_train=False)
        self.assertIs(result, model)
        evaluate.assert_not_called()
        model.load_state_dict.assert_not_called()
        self.assertEqual(optimizer.step.call_count, 3)  # Stub optimizer, no weight updates.
        self.assertEqual(metadata['epochs_run'], 3)
        self.assertEqual(metadata['returned_epoch'], 3)
        self.assertIsNone(metadata['best_epoch'])
        self.assertTrue(all('validation_log_loss' not in row for row in history))


if __name__ == '__main__':
    unittest.main()
