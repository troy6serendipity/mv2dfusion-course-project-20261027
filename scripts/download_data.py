"""Download official nuScenes Mini after explicit license acceptance.

Existing complete data is reused. Raw data never belongs in the public Git repo.
"""
import argparse,json,os,tarfile,time
from pathlib import Path
import requests

ROOT=Path(__file__).resolve().parents[1]
URL="https://www.nuscenes.org/data/v1.0-mini.tgz"
SIZE=4167696325

def complete(root):
    metadata=root/"v1.0-mini/sample_data.json"
    if not metadata.exists():
        return False
    rows=json.loads(metadata.read_text())
    if not rows:
        return False
    return all((root/r["filename"]).is_file() for r in rows)

def download(archive):
    if archive.exists() and archive.stat().st_size==SIZE:
        print("Existing complete archive reused.")
        return
    partial=archive.with_suffix(archive.suffix+".part")
    for attempt in range(5):
        offset=partial.stat().st_size if partial.exists() else 0
        if offset==SIZE:
            partial.replace(archive)
            return
        try:
            headers={"Range":"bytes=%d-"%offset} if offset else {}
            with requests.get(URL,headers=headers,stream=True,timeout=(20,120)) as response:
                response.raise_for_status()
                if offset and response.status_code==206:
                    if not response.headers.get("Content-Range","").startswith("bytes %d-"%offset):
                        raise RuntimeError("Unexpected server byte range")
                    mode="ab"
                else:
                    offset=0
                    mode="wb"
                with partial.open(mode) as out:
                    for chunk in response.iter_content(1024*1024):
                        if chunk: out.write(chunk)
            if partial.stat().st_size!=SIZE:
                raise RuntimeError("Incomplete archive; preserved for resume")
            partial.replace(archive)
            return
        except Exception as error:
            print("Attempt %d failed: %s"%(attempt+1,str(error)[:180]),flush=True)
            if attempt==4: raise
            time.sleep(2)

def extract(archive,root):
    root.mkdir(parents=True,exist_ok=True)
    base=root.resolve()
    with tarfile.open(archive,"r:gz") as package:
        for member in package:
            target=(base/member.name).resolve()
            if os.path.commonpath([str(base),str(target)])!=str(base):
                raise RuntimeError("Archive path escapes data root")
            if member.issym() or member.islnk():
                raise RuntimeError("Unexpected archive symlink")
            package.extract(member,path=root)
    if not complete(root):
        raise RuntimeError("Extracted sample_data files are incomplete")
    (root/".mv2d_mini_extraction_complete").write_text("official v1.0-mini\n")
    print("Complete official Mini data extracted:",root)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--accept-license",action="store_true",
        help="Confirm you have read and accept nuScenes dataset terms.")
    p.add_argument("--data-root",type=Path,default=Path(os.environ.get("NUSCENES_ROOT",str(ROOT/"data/nuscenes"))))
    p.add_argument("--archive",type=Path,default=ROOT/"data/downloads/v1.0-mini.tgz")
    a=p.parse_args()
    if complete(a.data_root):
        print("Existing complete nuScenes Mini reused; no download performed.")
        return
    if not a.accept_license:
        raise SystemExit("Read https://www.nuscenes.org/terms-of-use and rerun with --accept-license. Dataset CC BY-NC-SA 4.0 and provider terms apply.")
    a.archive.parent.mkdir(parents=True,exist_ok=True)
    download(a.archive)
    extract(a.archive,a.data_root)
    print("Verified recorded archive length; no claim of official MD5 comparison.")
if __name__=="__main__": main()
