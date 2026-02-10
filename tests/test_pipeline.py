
import unittest
from unittest.mock import MagicMock, patch
import os
import shutil
from src.pipeline import IsolationForestPipeline

class TestPipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Use existing model files
        cls.model_path = "data/isolation_forest_model.joblib"
        cls.imputer_path = "data/imputer.joblib"
        if not os.path.exists(cls.model_path):
            raise unittest.SkipTest("Model files not found")

    def test_predict(self):
        pipeline = IsolationForestPipeline(self.model_path, self.imputer_path)
        data = {
            "p_pdg": 1.0,
            "p_tpt": 2.0,
            "t_tpt": 3.0,
            "p_pck_up": 4.0,
            "t_pck_down": 5.0
        }
        prediction = pipeline.predict(data)
        self.assertIn("score", prediction)
        self.assertIn("is_anomaly", prediction)
        self.assertIsInstance(prediction["score"], float)
        self.assertIsInstance(prediction["is_anomaly"], bool)

    @patch("src.pipeline.requests.post")
    def test_generate_infographic_pdf(self, mock_post):
        pipeline = IsolationForestPipeline(self.model_path, self.imputer_path)

        # Mock Groq response
        mock_response = MagicMock()
        mock_response.ok = True
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": "<html><body><h1>Test Report</h1><p>Engineer View</p></body></html>"
                }
            }]
        }
        mock_post.return_value = mock_response

        prediction = {
            "score": -0.5,
            "is_anomaly": True,
            "input_data": {"test": 1}
        }
        user_role = "Engineer"
        output_path = "test_report.pdf"

        # Generate PDF
        result_path = pipeline.generate_infographic_pdf(prediction, user_role, output_path)

        self.assertEqual(result_path, output_path)
        self.assertTrue(os.path.exists(output_path))

        # Cleanup
        if os.path.exists(output_path):
            os.remove(output_path)

if __name__ == "__main__":
    unittest.main()
