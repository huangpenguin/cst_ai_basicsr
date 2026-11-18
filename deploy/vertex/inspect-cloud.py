"""Read-only current Endpoint, identity and replica samples. Never predicts."""

import datetime as dt
import json
import sys

sys.path.insert(0, "/usr/lib/google-cloud-sdk/lib/third_party")
import google.auth
from google.auth.transport.requests import AuthorizedSession

credentials, _ = google.auth.load_credentials_from_file(
    "/adc.json",
    scopes=[
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/userinfo.email",
        "openid",
    ],
)
session = AuthorizedSession(credentials)
identity_session = AuthorizedSession(credentials.with_quota_project(None))
now = dt.datetime.now(dt.timezone.utc)
project = "hallowed-pipe-497802-i0"
headers = {"x-goog-user-project": project}
endpoint_url = "https://asia-northeast1-aiplatform.googleapis.com/v1beta1/projects/580558088344/locations/asia-northeast1/endpoints/8911645097217490944"
filt = (
    'metric.type="aiplatform.googleapis.com/prediction/online/replicas" '
    'AND resource.type="aiplatform.googleapis.com/Endpoint" '
    'AND resource.labels.location="asia-northeast1" '
    'AND resource.labels.endpoint_id="8911645097217490944" '
    'AND metric.labels.deployed_model_id="1239217774224474112"'
)
result = {"observed_utc": now.isoformat(), "prediction_requests_sent": 0}
for name, url, params in [
    ("identity", "https://openidconnect.googleapis.com/v1/userinfo", {}),
    ("endpoint", endpoint_url, {}),
    (
        "replicas",
        f"https://monitoring.googleapis.com/v3/projects/{project}/timeSeries",
        {
            "filter": filt,
            "interval.startTime": (now - dt.timedelta(minutes=20)).isoformat(),
            "interval.endTime": now.isoformat(),
            "pageSize": 100,
        },
    ),
]:
    r = (identity_session if name == "identity" else session).get(
        url, params=params, headers={} if name == "identity" else headers, timeout=60
    )
    result[name] = {"http_status": r.status_code, "response": r.json()}
print(json.dumps(result, indent=2))
