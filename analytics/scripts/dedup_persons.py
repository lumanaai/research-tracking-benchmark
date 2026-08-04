import os
from utils.infra.singlestore_io import SingleStoreDb
from utils.infra.credentials import credentials_file_to_env
from pathlib import Path

#### script to generate a file with DB instruction due to permissions issue ####
if __name__ == "__main__":
    org_hash = 1841476311690434
    credentials_file_to_env(Path(__file__).parent.parent.joinpath("env_cred.json"))
    out_file = "/mnt/d/dedup_persons.txt"
    if os.path.exists(out_file):
        os.remove(out_file)
    queries_to_save = []
    db = SingleStoreDb()
    get_repetitions_query = f"select name, count(*) as cnt from persons where orgIdHash={org_hash} group by name  having cnt > 1 order by cnt desc"
    results = db.run_query(get_repetitions_query)
    for res in results:
        name = res["name"]
        get_persons_query = f"select personId from persons where orgIdHash={org_hash} and name='{name}'"
        person_ids =tuple([p["personId"] for p in  db.run_query(get_persons_query)])
        if len(person_ids) > 1:
            reference_id = person_ids[0]
            queries_to_save.append(f"update trackers2 set personId={reference_id} where orgIdHash={org_hash} and  personId in {person_ids[1:]};")
            queries_to_save.append(f"update person_representatives set personId={reference_id} where orgIdHash={org_hash} and personId in {person_ids[1:]};")
            queries_to_save.append(f"delete from persons where orgIdHash={org_hash} and personId in {person_ids[1:]};")

    # write all queries to out file
    with open(out_file, "w") as f:
        f.write("\n".join(queries_to_save))
