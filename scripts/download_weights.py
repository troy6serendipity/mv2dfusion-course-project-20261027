"""Retrieve the author's public MV2DFusion R50 checkpoint or reuse a local file."""
import argparse,hashlib,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
FILE_ID="1Sf5jeH8CMHauxJRRNuwYd7aXy2n4dlEo"
SIZE=881149879
RECORDED_SHA256="0b78315efcd6ab9a0b6f6a76467d01e2381f3c6a9efb3b4a5879a92fe4a867b0"
SOURCE="https://drive.google.com/drive/folders/1UEB-ynVrprZrUzY6UWzEZFPjaF-FEzpx"
def verify(path):
    if path.stat().st_size!=SIZE:
        raise RuntimeError("Checkpoint size differs from recorded R50 file")
    digest=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b""): digest.update(chunk)
    if digest.hexdigest()!=RECORDED_SHA256:
        raise RuntimeError("Checkpoint differs from this reproduction's recorded file. Existing file preserved.")
    print("Recorded local-reference SHA256:",digest.hexdigest())
    print("This reference was computed locally, not published as an author checksum.")
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--output",type=Path,default=Path(os.environ.get("MV2DFUSION_CHECKPOINT",str(ROOT/"weights/mv2dfusion_r50_nusc_latest.pth"))))
    p.add_argument("--proxy",default=None,help="Optional HTTP proxy; no proxy settings are changed globally.")
    a=p.parse_args()
    if a.output.exists():
        verify(a.output)
        print("Existing checkpoint reused; no download performed.")
        return
    print("Author source:",SOURCE)
    print("External weights retain the author's terms; not redistributed in this repository.")
    import gdown
    a.output.parent.mkdir(parents=True,exist_ok=True)
    for attempt in range(3):
        try:
            gdown.download(id=FILE_ID,output=str(a.output),resume=True,use_cookies=False,proxy=a.proxy)
            verify(a.output)
            return
        except Exception as error:
            print("Attempt %d: %s"%(attempt+1,str(error)[:180]),flush=True)
            if attempt==2: raise
            time.sleep(2)
if __name__=="__main__": main()
