import os, requests, time

RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
RUNPOD_ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "")

BASE_URL = f"https://api.runpod.ai/v2/{RUNPOD_ENDPOINT_ID}"

HEADERS = {
    "Authorization": f"Bearer {RUNPOD_API_KEY}",
    "Content-Type": "application/json",
}

def run_query(payload: dict, poll: bool = True, timeout: int = 40):
    if not RUNPOD_API_KEY or not RUNPOD_ENDPOINT_ID:
        raise RuntimeError("RUNPOD_API_KEY or RUNPOD_ENDPOINT_ID missing")

    r = requests.post(f"{BASE_URL}/run", headers=HEADERS, json={"input": payload})
    r.raise_for_status()
    job = r.json()
    job_id = job.get("id")

    if not poll:
        return job

    start = time.time()
    while time.time() - start < timeout:
        q = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS)
        q.raise_for_status()
        js = q.json()
        if js.get("status") == "COMPLETED":
            return js.get("output")
        if js.get("status") in ("FAILED", "CANCELLED"):
            raise RuntimeError(f"RunPod job failed: {js}")
        time.sleep(1.5)

    raise TimeoutError("RunPod job timed out")
