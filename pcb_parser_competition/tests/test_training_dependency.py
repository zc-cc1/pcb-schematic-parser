import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

from pcb_parser.training_dependency import prerequisite_state


class TrainingDependencyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        folder = Path(self.temp.name)
        self.job = dict(name="YOLO trial", status_path=str(folder/"status.json"),
                        training_record_path=str(folder/"training.json"), expected_run="trial",
                        expected_process_id=123, training_protocol_sha256="known-protocol")
        self.status = dict(status="completed", pid=123)
        # Early stopping is valid completion; do not require all 120 planned epochs.
        self.training = dict(status="trained", completed_epochs=93, protocol_sha256="known-protocol")
        self.alive = Mock(return_value=False)

    def state(self):
        Path(self.job["status_path"]).write_text(json.dumps(self.status), encoding="utf-8")
        Path(self.job["training_record_path"]).write_text(json.dumps(self.training), encoding="utf-8")
        return prerequisite_state(self.job, self.alive)

    def test_successful_early_stop_and_controller_exit_are_ready(self):
        self.assertTrue(self.state()["ready"])
        self.alive.assert_called_once_with(123)

    def test_training_evaluation_failure_and_dead_process_do_not_release_queue(self):
        for status in ("training", "evaluating_ap", "failed", "waiting_for_gpu"):
            with self.subTest(status=status):
                self.status["status"] = status
                self.assertFalse(self.state()["ready"])
        self.alive.assert_not_called()

    def test_completed_status_requires_trained_record(self):
        self.training["status"] = "training"
        self.assertFalse(self.state()["ready"])
        self.alive.assert_not_called()

    def test_completed_controller_must_also_exit(self):
        self.alive.return_value = True
        self.assertEqual(self.state()["reason"], "controller_still_running")
        self.assertFalse(self.state()["ready"])

    def test_unrelated_run_protocol_and_process_do_not_release_queue(self):
        for target, key, value in ((self.status, "run", "other-run"),
                                   (self.status, "pid", 456),
                                   (self.training, "protocol_sha256", "other-protocol")):
            with self.subTest(key=key):
                previous = target.copy()
                target[key] = value
                self.assertFalse(self.state()["ready"])
                target.clear(); target.update(previous)

    def test_missing_or_partly_written_completion_record_keeps_waiting(self):
        self.assertFalse(prerequisite_state(self.job, self.alive)["ready"])
        self.state()
        for content in ("{", "null", "[]"):
            Path(self.job["status_path"]).write_text(content, encoding="utf-8")
            self.assertFalse(prerequisite_state(self.job, self.alive)["ready"])

    def test_completed_status_requires_positive_integer_epoch_count(self):
        for count in (None, 0, -1, True, "120"):
            self.training["completed_epochs"] = count
            self.assertFalse(self.state()["ready"])


if __name__ == "__main__":
    unittest.main()
