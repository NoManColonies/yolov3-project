import threading
import queue
import uuid
import requests
from evidence_tracker import EvidenceTracker
from processible_evidence import ProcessibleEvidence
from jwt import (
    JWT,
    jwk_from_pem,
)
from jwt.utils import get_int_from_datetime
from datetime import datetime, timedelta, timezone
from os import getenv


class EvidenceProcessor(threading.Thread):
    def __init__(self, queue: queue.PriorityQueue):
        threading.Thread.__init__(self)
        threading.Thread.daemon = True
        cf_token = str(getenv("CF_TOKEN", ""))
        cf_account = str(getenv("CF_ACCOUNT", ""))
        self.id = uuid.uuid4()
        self.queue = queue
        self.video_upload_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/stream"
        self.image_upload_url = f"https://api.cloudflare.com/client/v4/accounts/{cf_account}/images/v1"
        self.headers = {
            "Authorization": f"Bearer {cf_token}",
        }

    def __upload_evidences(self, ids):
        for id in ids:
            print(f"begin upload evidence id: {id} ...")
            image_files = {
                'file': open(f'result/{id}.png', 'rb')
            }
            image_upload_response = requests.post(self.image_upload_url,
                                                  files=image_files, headers=self.headers).json()
            video_files = {
                'file': open(f'result/{id}.mp4', 'rb')
            }
            video_upload_response = requests.post(self.video_upload_url,
                                                  files=video_files, headers=self.headers).json()
            message = {
                'vid': video_upload_response['result']['uid'],
                'vdata': str(id) + ',' + image_upload_response['result']['id']
            }
            requests.post('https://royaltraffic.katsu-r-alias.workers.dev/api/queue', json=message, headers=self.headers)
            print(f"begin upload evidence id: {id} ... [done]")

    def run(self):
        try:
            print("initiating processor task...")
            while True:
                ids = []

                print("waiting for queue item to arrive...")
                evidences: ProcessibleEvidence = self.queue.get()
                print("queue item arrived. processing...")
                for tracker in evidences.trackers:
                    id = tracker.compose_evidence(evidences.frames.copy())
                    if id is not None:
                        ids.append(id)

                self.__upload_evidences(ids)

                print("queue item process done")
                self.queue.task_done()
        except BaseException as e:
            print(e)
