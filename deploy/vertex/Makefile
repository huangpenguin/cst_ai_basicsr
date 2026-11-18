# Run from the repository root. Host configuration stays on the host's local disk.
UI_ENV_FILE ?= /home/huang/.config/basicsr-vertex/ui.env
-include $(UI_ENV_FILE)
export UI_BIND_IP UI_ADC_PATH UI_PORT UI_CONTAINER UI_IMAGE
.PHONY: ui-check ui-start ui-status ui-logs cloud-status training-io-test
ui-check:
	bash deploy/vertex/start-internal-ui.sh --check
ui-start:
	bash deploy/vertex/start-internal-ui.sh
ui-status:
	docker inspect "$${UI_CONTAINER:-denoise-vertex-ui}" --format '{{.State.Status}} {{.Config.Image}} {{json .HostConfig.PortBindings}}'
ui-logs:
	docker logs --tail 100 "$${UI_CONTAINER:-denoise-vertex-ui}"
cloud-status:
	bash deploy/vertex/cloud-cli.sh ai endpoints describe 8911645097217490944 --project=hallowed-pipe-497802-i0 --billing-project=hallowed-pipe-497802-i0 --region=asia-northeast1 --format=json
training-io-test:
	python3 -m unittest discover -s deploy/vertex/training -p 'test_*.py' -v
