import unittest

from orders import apply_discount, format_total, total_price


class OrderTests(unittest.TestCase):
    def test_format_total(self):
        self.assertEqual(format_total(9.9), "¥9.90")
        self.assertEqual(format_total(19.999), "¥20.00")

    def test_apply_discount(self):
        self.assertEqual(apply_discount(99, 10), 89.1)

    def test_total_price(self):
        items = [
            {"price": 10, "quantity": 2},
            {"price": 5.5, "quantity": 3},
        ]
        self.assertEqual(total_price(items), 36.5)


if __name__ == "__main__":
    unittest.main()
