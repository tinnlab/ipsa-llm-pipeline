"""
Server-level tests for pipeline step numbering.

After removing the old Therapeutic Step 5, report generation became Step 5, so the
pipeline is 1-5. The /api/pipeline/step/{n} endpoint must reject out-of-range steps
(0 and >5) with a 400.
"""

from fastapi.testclient import TestClient

from src.api.server import app

client = TestClient(app)


def _body(step_number):
    return {
        "step_number": step_number if 1 <= step_number <= 5 else 1,  # body constraint ge/le
        "input_data": {
            "pathways": [{
                "name": "Cell cycle", "source": "KEGG", "pathwayId": "hsa04110",
                "pValue": 0.01, "pValueFDR": 0.02, "genes": ["CDK1"],
            }],
        },
        "previous_results": {},
    }


def test_step6_is_rejected():
    """Step 6 no longer exists (report is Step 5) → rejected as out of range."""
    r = client.post("/api/pipeline/step/6", json=_body(6))
    assert r.status_code == 400
    assert "must be 1-5" in r.json()["detail"].lower()


def test_out_of_range_step_rejected():
    assert client.post("/api/pipeline/step/7", json=_body(7)).status_code == 400
    assert client.post("/api/pipeline/step/0", json=_body(0)).status_code == 400
