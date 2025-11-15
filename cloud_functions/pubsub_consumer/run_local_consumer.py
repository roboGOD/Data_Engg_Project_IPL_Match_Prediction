#!/usr/bin/env python3
"""Local runner to invoke the Cloud Function `pubsub_consumer` locally.

Usage:
  python run_local.py --file sample_event.json
  python run_local.py --json '{"eventType":"match","payload":{...}}'

This script builds a fake Pub/Sub event by base64-encoding the JSON payload
and calls `pubsub_consumer(event, context)` from `consumer.py`.

Note: The BigQuery client will still attempt to connect to GCP unless you
mock or set credentials. For a quick dry-run, set the environment variable
`BQ_DRY_RUN=1` to skip the actual insert (the script will still call the
function but the function will attempt insertion unless you modify it).
"""

import argparse
import base64
import json
import os
import sys

from cloud_functions.pubsub_consumer.main import pubsub_consumer


class FakeContext:
    def __init__(self, event_id="local-event", timestamp="1970-01-01T00:00:00Z"):
        self.event_id = event_id
        self.timestamp = timestamp


def build_event_from_json_obj(obj: dict) -> dict:
    raw = json.dumps(obj)
    return {"data": base64.b64encode(raw.encode("utf-8")).decode("utf-8")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", help="Path to a JSON file containing the message body")
    parser.add_argument("--json", help="Inline JSON message body")
    parser.add_argument("--event-id", default="local-event", help="Fake event id")
    args = parser.parse_args()

    if not args.file and not args.json:
        print("Provide either --file or --json", file=sys.stderr)
        parser.print_help()
        sys.exit(2)

    if args.file:
        with open(args.file, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    else:
        obj = json.loads(args.json)

    event = build_event_from_json_obj(obj)
    context = FakeContext(event_id=args.event_id)

    print("Invoking pubsub_consumer with payload:\n", json.dumps(obj, indent=2))

    try:
        pubsub_consumer(event, context)
        print("Invocation completed without uncaught exceptions.")
    except Exception as e:
        print("Function raised an exception:", repr(e))
        # Re-raise with non-zero exit code to make failures visible in CI if used
        raise


if __name__ == "__main__":
    main()
