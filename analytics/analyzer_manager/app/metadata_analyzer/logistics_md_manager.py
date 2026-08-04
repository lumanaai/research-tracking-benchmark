import os
import time

from general.analyzer_general import ANALYTIC_ALERT_TYPE, MULTIPLE_ENUM
from general.core import Metadata, ThumbnailType, AlertType
from general.img_utils import scale_image_by_width
from metadata_analyzer.image_handler import ImageHandler


class LogisticsMetadataManager:
    def __init__(self, context):
        self.analytic_config = context.config
        self.config = context.config
        self.text_analyzer = context.text_analyzer
        self.thumb_width = int(self.config["thumbnailPolicy"]["thum_width"])
        self.snapshot_width = int(self.config["thumbnailPolicy"]["snap_width"])
        self.magnitude = self.config["post_process"]["pixelateThumbnail"]["magnitude"]
        self.last_snapshot = ""
        self.snapshot_cycle = False
        self.last_thumbnail = ""
        self.last_thumbnail_ts = -1
        self.update_empty_json: bool = bool(self.config["jsonPolicy"]["updateEmptyInfo"])
        # image handling
        train_path = os.path.join(
            context.app_config["locations"]["trainingThumbnails"], context.edge_id, context.camera_id
        )

        self.thumb_handler = ImageHandler(
            os.path.join(context.app_config["locations"]["thumbnails"], context.edge_id, context.camera_id),
            limit_per_period=1,
            period_sec=int(self.config["thumbnailPolicy"]["thumbnailsDuration"]) / 1000,
            name_builder="thumb_counter",
            nvjpeg_encoder_address=context.nvjpeg_encoder_address,
            jpg_quality=int(self.config["thumbnailPolicy"]["jpegquality"]),
        )

        self.snapshot_handler = ImageHandler(
            train_path,
            limit_per_period=1,
            prefix="snapshot",
            period_sec=int(self.config["thumbnailPolicy"]["snapshotDuration"]) / 1000,
            name_builder="norm_timestamp",
            nvjpeg_encoder_address=context.nvjpeg_encoder_address,
            jpg_quality=int(self.config["thumbnailPolicy"]["snapshotquality"]),
        )

    def run(self, image_batch, text_results):
        results = Metadata()
        timestamps = [image.timestamp for image in image_batch]
        thumbnails, snapshots = self.generate_thumbs(image_batch)

        sync_report = None
        alerts = []
        for text in text_results:
            mock_alert_json = {
                "alertType": ANALYTIC_ALERT_TYPE,
                "detectionType": AlertType.motionScore.value,
                "thumbnails": None,
                "eventId": self.config.get("logistic", {}).get("eventId", "text detection"),
                "alertMessage": " ".join(text["text"]),
                "alertData": text,
                "type": MULTIPLE_ENUM,
            }
            alerts.append(mock_alert_json)

        if alerts or self.update_empty_json or self.snapshot_cycle:
            sync_report = {}
            sync_report["alerts"] = alerts
            sync_report["mainThumbnail"] = self.last_thumbnail
            sync_report["mainThumbnailTimestamp"] = self.last_thumbnail_ts
            sync_report["timestamp"] = str(timestamps[-1])
            sync_report["misc"] = ""
            sync_report["info"] = []
            sync_report["objects"] = {}

            if self.snapshot_cycle:
                sync_report["mainSnapshot"] = self.last_snapshot
                self.snapshot_cycle = False

        if sync_report is not None:
            metadata = {
                "timestamp": str(int(round(time.time() * 1000))),
                "misc": {"snapshotInfo": snapshots},
                "snapshots": snapshots,
            }
            results.info = {"metadata": metadata, "results": [sync_report]}
        results.snapshots.extend(snapshots)
        results.thumbnails.extend(thumbnails)

        return results

    def generate_thumbs(self, image_batch):
        thumbnails = []
        snapshots = []
        for i, im in enumerate(image_batch):
            if self.thumb_handler.can_send(im.timestamp):
                annotated_image = scale_image_by_width(im.frame, self.thumb_width)
                thumb_name, normalized_ts = self.thumb_handler.send(annotated_image, im.timestamp)
                thumbanil_info = {
                    "normalizedTimestamp": normalized_ts,
                    "events": [
                        {
                            "alerts": [],
                            "mainThumbnail": thumb_name,
                            "timestamp": im.timestamp,
                            "thumType": ThumbnailType.Thumbnail.value,
                        }
                    ],
                }
                thumbnails.append(thumbanil_info)
                self.last_thumbnail = thumb_name
                self.last_thumbnail_ts = im.timestamp
            if self.snapshot_handler.can_send(im.timestamp):
                scaled_image = scale_image_by_width(im.frame, self.snapshot_width)
                thumb_name, normalized_ts = self.snapshot_handler.send(scaled_image, im.timestamp)
                snapshots.append(
                    {"thumbnail": thumb_name, "timestamp": im.timestamp, "thumType": ThumbnailType.Snapshot.value}
                )
                self.last_snapshot = thumb_name
                self.snapshot_cycle = True
        return thumbnails, snapshots
