"""B-side public TARGET DATA transport only; no model or prediction imports."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import urllib.request

from download_stage17_data import ranged


def request_range(url, end=1023):
    request = urllib.request.Request(url+("&" if "?" in url else "?")+"stage17_metadata=1",
                                     headers={"Range": f"bytes=0-{end}", "User-Agent": "GOFA-F-Stage17"})
    with urllib.request.urlopen(request, timeout=30) as response:
        raw = response.read(end+1)
        total = int(response.headers["Content-Range"].split("/")[-1])
    return raw, total


def noaa_metadata(root):
    url = "https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2025/"
    text = urllib.request.urlopen(url, timeout=30).read().decode()
    names = re.findall(r'href="(CRNH0203-2025-[^"]+\.txt)"', text)
    def station(name):
        raw, size = request_range(url+name)
        fields = raw.decode().splitlines()[0].split()
        return {"filename": name, "size": size, "wban": fields[0],
                "longitude": float(fields[6]), "latitude": float(fields[7])}
    with ThreadPoolExecutor(max_workers=6) as pool:
        stations = list(pool.map(station, names))
    (root/"noaa_station_metadata.json").write_text(json.dumps(stations, indent=2))
    return stations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--noaa", action="store_true")
    parser.add_argument("--noaa-files", action="store_true")
    parser.add_argument("--pems08", action="store_true")
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.noaa:
        noaa_metadata(root)
        return
    if args.noaa_files:
        selected = json.loads((root/"noaa_selected_stations.json").read_text())
        destination = root/"noaa"
        destination.mkdir(exist_ok=True)
        for station in selected:
            path = destination/station["filename"]
            if not path.exists():
                ranged("https://www.ncei.noaa.gov/pub/data/uscrn/products/hourly02/2025/"+station["filename"],
                       path, station["size"])
        return
    if args.pems08:
        base = "https://raw.githubusercontent.com/guoshnBJTU/ASTGNN/9c2e19b98c4cedf1f35214d8789685b6381b3aad/data/PEMS08/"
        for filename in ("PEMS08.npz", "PEMS08.csv"):
            _, size = request_range(base+filename, 0)
            ranged(base+filename, root/filename, size)
        return
    tree = ast.parse(Path("src/gofa_f/generative_data.py").read_text())
    drive = next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "DRIVE" for t in node.targets))
    for name, descriptor in drive.items():
        output = root/(name+".csv")
        if output.exists():
            continue
        url = f"https://drive.usercontent.google.com/download?id={descriptor[0]}&export=download&confirm=t"
        _, size = request_range(url, 0)
        ranged(url, output, size)
        print(name, size, hashlib.sha256(output.read_bytes()).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
