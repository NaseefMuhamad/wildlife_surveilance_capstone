import unittest
from backend.live_video import LiveCamera, validate_source


class LiveCameraTests(unittest.TestCase):
    def test_camera_source_validation(self):
        self.assertEqual(validate_source("http://192.168.1.20:8080/video"), "http://192.168.1.20:8080/video")
        for value in (None, "file:///private", "rtsp://camera/live", "http://user:secret@camera/video", "http://"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_source(value)

    def test_disconnect_removes_stale_detections(self):
        camera = LiveCamera("unused.pt")
        camera.jpeg = b"old image"
        camera.state = {"status": "live", "detections": [{"class": "human"}], "frames": 2}
        camera.stop()
        state, image = camera.snapshot()
        self.assertIsNone(image)
        self.assertEqual(state["detections"], [])
        self.assertEqual(state["status"], "disconnected")

    def test_old_frame_is_not_reported_as_live(self):
        camera = LiveCamera("unused.pt")
        camera.state = {"status": "live", "updated": 1}
        self.assertEqual(camera.snapshot()[0]["status"], "stalled")
