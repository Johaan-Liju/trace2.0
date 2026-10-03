import math
import unittest
from trace.rules import ZoneRule, inside

ZONE = [(0.5, 0.2), (1, 0.2), (1, 1), (0.5, 1)]


class RuleTests(unittest.TestCase):
    def test_boundary_is_inside(self):
        self.assertTrue(inside((.5, .5), ZONE))
        self.assertFalse(inside((.49, .5), ZONE))

    def test_dwell_once_and_reentry(self):
        r = ZoneRule(ZONE, dwell_seconds=1)
        self.assertEqual(r.update(0, {1: (.7, .7)}), [])
        self.assertEqual(r.update(.5, {1: (.7, .7)}), [])
        self.assertEqual(len(r.update(1, {1: (.7, .7)})), 1)
        self.assertEqual(r.update(1.5, {1: (.7, .7)}), [])
        r.update(2, {1: (.1, .7)})
        r.update(2.5, {1: (.7, .7)})
        r.update(3, {1: (.7, .7)})
        self.assertEqual(len(r.update(3.5, {1: (.7, .7)})), 1)

    def test_single_frame_does_not_alert(self):
        r = ZoneRule(ZONE)
        r.update(0, {1: (.7, .7)})
        self.assertEqual(r.update(1, {}), [])
        self.assertEqual(r.update(2, {1: (.7, .7)}), [])

    def test_missing_time_does_not_count_as_observed(self):
        r = ZoneRule(ZONE)
        r.update(0, {1: (.7, .7)})
        r.update(.5, {1: (.7, .7)})
        r.update(.6, {})
        self.assertEqual(r.update(1.1, {1: (.7, .7)}), [])
        self.assertEqual(len(r.update(1.6, {1: (.7, .7)})), 1)

    def test_gap_expires_even_without_empty_updates(self):
        r = ZoneRule(ZONE)
        r.update(0, {1: (.7, .7)})
        self.assertEqual(r.update(5, {1: (.7, .7)}), [])

    def test_independent_people(self):
        r = ZoneRule(ZONE, dwell_seconds=.5)
        r.update(0, {1: (.7, .7), 2: (.2, .7)})
        events = r.update(.5, {1: (.7, .7), 2: (.7, .7)})
        self.assertEqual([e['track_id'] for e in events], [1])

    def test_invalid_settings_and_timestamps(self):
        for value in (-1, math.nan, math.inf):
            with self.assertRaises(ValueError):
                ZoneRule(ZONE, dwell_seconds=value)
        with self.assertRaises(ValueError):
            ZoneRule([(0, 0), (0, 0), (0, 0)])
        r = ZoneRule(ZONE)
        r.update(1, {})
        with self.assertRaises(ValueError):
            r.update(.5, {})


if __name__ == '__main__':
    unittest.main()
