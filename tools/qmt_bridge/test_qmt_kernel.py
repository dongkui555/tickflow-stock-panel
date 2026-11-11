from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase, main
from unittest.mock import Mock, patch

from qmt_http_api import QmtKernel


class QmtKernelTest(TestCase):
    def test_start_uses_current_token_after_client_restart(self):
        with TemporaryDirectory() as folder:
            config = Path(folder) / "notebook.py"
            config.write_text("c.NotebookApp.token = 'old-token'\n", encoding="utf-8")
            kernel = QmtKernel("http://127.0.0.1:8888", config)
            config.write_text("c.NotebookApp.token = 'new-token'\n", encoding="utf-8")
            response = Mock()
            response.json.return_value = {"id": "new-kernel"}
            with patch("qmt_http_api.requests.post", return_value=response) as post:
                kernel.start()
            self.assertEqual(post.call_args.kwargs["headers"], {"Authorization": "token new-token"})
            self.assertEqual(kernel.kernel_id, "new-kernel")

    def test_existing_kernel_health_uses_current_token(self):
        with TemporaryDirectory() as folder:
            config = Path(folder) / "notebook.py"
            config.write_text("c.NotebookApp.token = 'old-token'\n", encoding="utf-8")
            kernel = QmtKernel("http://127.0.0.1:8888", config)
            kernel.kernel_id = "retained-kernel"
            config.write_text("c.NotebookApp.token = 'new-token'\n", encoding="utf-8")
            with (
                patch("qmt_http_api.requests.get", return_value=Mock(ok=True)) as get,
                patch("qmt_http_api.requests.post") as post,
            ):
                kernel.start()
            self.assertEqual(get.call_args.kwargs["headers"], {"Authorization": "token new-token"})
            post.assert_not_called()

    def test_invalid_current_config_does_not_use_old_credentials(self):
        with TemporaryDirectory() as folder:
            config = Path(folder) / "notebook.py"
            config.write_text("c.NotebookApp.token = 'old-token'\n", encoding="utf-8")
            kernel = QmtKernel("http://127.0.0.1:8888", config)
            config.write_text("# config is being replaced\n", encoding="utf-8")
            with (
                patch("qmt_http_api.requests.post") as post,
                self.assertRaisesRegex(RuntimeError, "token not found"),
            ):
                kernel.start()
            post.assert_not_called()


if __name__ == "__main__":
    main()
