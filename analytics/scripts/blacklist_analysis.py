import csv
import os
from pathlib import PurePosixPath
from tabulate import tabulate

from scripts.helpers import save_dict_list_to_csv, run_singlestore_query, timestamp_to_tz, training_bucket
from utils.infra.s3_io import S3Handler

base_path = "/mnt/c/Temp/blacklist/"


def convert_file_structure():
    camera_dict = get_camera_metadata()
    for cam_id, md in camera_dict.items():
        edge_id = md["edgeId"]
        orig_path = os.path.join(base_path, edge_id, cam_id)
        dest_path = os.path.join(base_path, md["orgName"], md["cameraName"])
        if os.path.exists(orig_path) and not os.path.exists(dest_path):
            os.makedirs(dest_path)
            os.rename(orig_path, dest_path)


def convert_csv():
    camera_dict = get_camera_metadata()
    csv_file = os.path.join(base_path, "all.csv")
    new_data = []
    with open(csv_file, newline="", encoding="utf-8") as csvfile:
        # Create a CSV DictReader
        data_reader = csv.DictReader(csvfile)

        # Iterate over the rows in the CSV file
        for row in data_reader:
            cam_id = row["cameraId"]
            row["camera_name"] = camera_dict.get(cam_id, {}).get("cameraName", cam_id)
            row["org_name"] = camera_dict.get(cam_id, {}).get("orgName", "Null")

            new_data.append(row)  # 'row' is a dictionary

    save_dict_list_to_csv(new_data, os.path.join(base_path, "all_c.csv"))


def get_camera_metadata():
    md_query = f"SELECT cameraId, cameraName, edgeId, orgName FROM orgdevices WHERE cameraId <> 'null'"
    camera_dict_res = run_singlestore_query(md_query)
    return {item["cameraId"]: item for item in camera_dict_res}


def main():
    handler = S3Handler(bucket_name=training_bucket)
    camera_dict = get_camera_metadata()

    # get all edges and cameras
    query = (
        f"SELECT DISTINCT cameraId, edgeId FROM trackers "
        f"WHERE timestamp/1000 >= UNIX_TIMESTAMP('2023-10-17 00:00:00') "
        f"AND JSON_MATCH_ANY(attributes::?globalType.value, MATCH_PARAM_DOUBLE_STRICT() in (3)) "
        f"LIMIT 150;"
    )
    cameras_result = run_singlestore_query(query)

    all_data = []
    for cam_res in cameras_result:
        try:
            cam_id = cam_res["cameraId"]
            org_name = camera_dict.get(cam_id).get("orgName")
            cam_name = camera_dict.get(cam_id).get("cameraName", cam_id)
            edge_id = cam_res["edgeId"]

            query = (
                f"SELECT idBase, idIndex, bestImage, cameraId, edgeId FROM trackers"
                f" WHERE cameraId = '{cam_id}' "
                f"AND timestamp/1000 >= UNIX_TIMESTAMP('2023-10-20 00:00:00') "
                f"AND JSON_MATCH_ANY(attributes::?globalType.value, MATCH_PARAM_DOUBLE_STRICT() in (3)) "
                f"ORDER BY timestamp LIMIT 1000;"
            )

            # Run the query
            local_path = os.path.join(base_path, org_name, cam_name)
            os.makedirs(local_path, exist_ok=True)

            results = run_singlestore_query(query)
            for res in results:
                image = res["bestImage"]
                res["time"] = timestamp_to_tz(res["idBase"]).strftime("%d-%m-%Y %H:%M:%S")
                res["camera_name"] = cam_name
                res["org_name"] = org_name
                try:
                    if image is not None:
                        local_file_path = os.path.join(local_path, image)
                        if not os.path.exists(local_file_path):
                            s3_path = PurePosixPath("crop").joinpath(edge_id, cam_id, image)
                            handler.download_single_file_s3(str(s3_path), local_file_path)
                except Exception as e:
                    print(e)
            table = tabulate(results, headers="keys", tablefmt="grid")
            all_data += results
            # Save the table to a text file
            with open(os.path.join(local_path, "output_table.txt"), "wt") as f:
                f.write(table)
        except Exception as e:
            print(e)

    # write all data to csv
    save_dict_list_to_csv(all_data, os.path.join(base_path, "all.csv"))


if __name__ == "__main__":
    convert_csv()


# create table trackers
# (
#     id             bigint                not null,
#     edgeId         varchar(24)           not null,
#     cameraId       varchar(24)           not null,
#     cameraIdHash   bigint                not null,
#     idBase         bigint                not null,
#     timestamp      bigint                not null,
#     idIndex        bigint                not null,
#     conf           float                 null,
#     attributes     JSON collate utf8_bin null,
#     bestImage      varchar(64)           null,
#     trackerTypeId  tinyint               null,
#     zoomImage      varchar(64)           null,
#     faceConfidence varchar(20)           null,
#     descriptor     varchar(4096)         null,
#     faceId         varchar(4096)         null,
#     lowerbodyType  as attributes::lowerbodyType PERSISTED JSON collate utf8_bin null,
#     upperbodyType  as attributes::upperbodyType PERSISTED JSON collate utf8_bin null,
#     carryingType   as attributes::carryingType PERSISTED JSON collate utf8_bin null,
#     accessoryType  as attributes::accessoryType PERSISTED JSON collate utf8_bin null,
#     genderType     as attributes::genderType PERSISTED JSON collate utf8_bin null,
#     ageType        as attributes::ageType PERSISTED JSON collate utf8_bin null,
#     footwearColor  as attributes::footwearColor PERSISTED JSON collate utf8_bin null,
#     hairColor      as attributes::hairColor PERSISTED JSON collate utf8_bin null,
#     lowerbodyColor as attributes::lowerbodyColor PERSISTED JSON collate utf8_bin null,
#     upperbodyColor as attributes::upperbodyColor PERSISTED JSON collate utf8_bin null,
#     vehicleType    as attributes::vehicleType PERSISTED JSON collate utf8_bin null,
#     vehicleMake    as attributes::vehicleMake PERSISTED JSON collate utf8_bin null,
#     vehicleModel   as attributes::vehicleModel PERSISTED JSON collate utf8_bin null,
#     vehicleColors  as attributes::vehicleColors PERSISTED JSON collate utf8_bin null,
#     vehiclePlate   as attributes::vehiclePlate PERSISTED JSON collate utf8_bin null,
#     vehicleRegion  as attributes::vehicleRegion PERSISTED JSON collate utf8_bin null,
#     globalType     as attributes::globalType PERSISTED JSON collate utf8_bin null,
#     orgId          varchar(24)           null,
#     maxNeighbors   JSON collate utf8_bin null,
#     dwell          float                 null,
#     orgIdHash      bigint                null,
#     descriptorBin  blob                  null,
#     faceIdBin      blob                  null,
#     groupId        bigint                null,
#     status         varchar(20)           null,
#     attributesHigh JSON collate utf8_bin null,
#     constraint primaryIndex
#         unique (id) using columnstore hash
# );
