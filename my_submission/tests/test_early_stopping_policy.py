import unittest

from my_submission.train import should_early_stop


class EarlyStoppingPolicyTest(unittest.TestCase):
    def test_patience_is_ignored_before_minimum_epoch(self) -> None:
        self.assertFalse(
            should_early_stop(
                epoch=2,
                epochs_without_improvement=5,
                patience=5,
                minimum_epochs=10,
            )
        )

    def test_patience_can_stop_after_minimum_epoch(self) -> None:
        self.assertTrue(
            should_early_stop(
                epoch=10,
                epochs_without_improvement=5,
                patience=5,
                minimum_epochs=10,
            )
        )

    def test_zero_patience_disables_early_stopping(self) -> None:
        self.assertFalse(
            should_early_stop(
                epoch=20,
                epochs_without_improvement=100,
                patience=0,
                minimum_epochs=10,
            )
        )


if __name__ == "__main__":
    unittest.main()
