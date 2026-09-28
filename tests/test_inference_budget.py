import unittest
from unittest.mock import patch
from inference_budget import Budget

class BudgetTests(unittest.TestCase):
    def test_rounds_share_one_deadline(self):
        with patch('inference_budget.time.monotonic',return_value=20):budget=Budget(10)
        with patch('inference_budget.time.monotonic',return_value=28):
            self.assertEqual(budget.remaining(50),2)
        with patch('inference_budget.time.monotonic',return_value=30):
            with self.assertRaises(TimeoutError):budget.remaining()
