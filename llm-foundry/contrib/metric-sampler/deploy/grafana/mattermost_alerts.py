import argparse
import base64
import json
import urllib.request
from pathlib import Path

DEFAULT_GRAFANA_URL = "http://127.0.0.1:3000"
DEFAULT_CONTACT_POINT = "metric-sampler alerts"
GRAFANA_ADMIN_USER = "admin"
REQUEST_TIMEOUT_SECONDS = 60
# Grafana's default alert message, posted as the {"text": ...} body an incoming Mattermost webhook takes.
WEBHOOK_PAYLOAD_TEMPLATE = '{{ coll.Dict "text" (tmpl.Exec "default.message" .) | data.ToJSON }}'
ADMIN_PASSWORD_FILE = "grafana_admin_password"
WEBHOOK_URL_FILE = "mattermost_alert_webhook"


class GrafanaClient:
    def __init__(self, grafana_url: str, admin_password: str) -> None:
        self._grafana_url = grafana_url
        self._authorization = "Basic " + base64.b64encode(f"{GRAFANA_ADMIN_USER}:{admin_password}".encode()).decode()

    def call(self, path: str, method: str = "GET", body: dict | None = None) -> dict | list:
        request = urllib.request.Request(
            self._grafana_url + path, method=method, data=None if body is None else json.dumps(body).encode(),
            headers={"Authorization": self._authorization, "Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_SECONDS) as response:
            raw = response.read()
        return json.loads(raw) if raw.strip() else {}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sends every Grafana alert to a Mattermost incoming webhook: creates or updates the contact "
                    "point and makes it the root of the notification policy.")
    parser.add_argument("--secrets-dir", type=Path, required=True,
                        help=f"directory holding {ADMIN_PASSWORD_FILE} and {WEBHOOK_URL_FILE}; never printed")
    parser.add_argument("--webhook-ca-certificate", type=Path,
                        help="PEM certificate of the webhook's CA when Grafana's image does not trust it")
    parser.add_argument("--contact-point", default=DEFAULT_CONTACT_POINT)
    parser.add_argument("--grafana-url", default=DEFAULT_GRAFANA_URL)
    return parser.parse_args()


def contact_point(arguments: argparse.Namespace) -> dict:
    settings: dict = {
        "url": (arguments.secrets_dir / WEBHOOK_URL_FILE).read_text().strip(),
        "httpMethod": "POST",
        "payload": {"template": WEBHOOK_PAYLOAD_TEMPLATE},
    }
    if arguments.webhook_ca_certificate is not None:
        settings["tlsConfig"] = {"caCertificate": arguments.webhook_ca_certificate.read_text()}
    return {"name": arguments.contact_point, "type": "webhook", "disableResolveMessage": False, "settings": settings}


def main() -> None:
    arguments = parse_arguments()
    grafana = GrafanaClient(
        arguments.grafana_url, (arguments.secrets_dir / ADMIN_PASSWORD_FILE).read_text().strip())
    point = contact_point(arguments)
    existing = [
        existing_point for existing_point in grafana.call("/api/v1/provisioning/contact-points")
        if existing_point["name"] == arguments.contact_point
    ]
    if existing:
        uid = existing[0]["uid"]
        grafana.call(f"/api/v1/provisioning/contact-points/{uid}", "PUT", {**point, "uid": uid})
    else:
        grafana.call("/api/v1/provisioning/contact-points", "POST", point)
    policy = grafana.call("/api/v1/provisioning/policies")
    policy["receiver"] = arguments.contact_point
    grafana.call("/api/v1/provisioning/policies", "PUT", policy)
    print(f"contact point {arguments.contact_point!r} {'updated' if existing else 'created'}; every alert goes to it")


if __name__ == "__main__":
    main()
