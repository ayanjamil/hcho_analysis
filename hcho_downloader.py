import os, netCDF4 as nc
import numpy as np, pandas as pd, requests, time
from tqdm import tqdm
from getpass import getpass
from datetime import datetime

# ------------------------------
# Force CPU (local machine version)
# ------------------------------
xp = np
GPU_AVAILABLE = False
print("Running in CPU mode with NumPy (GPU skipped)")


# ------------------------------
# Local CSV Selection
# ------------------------------
csv_file = input("Enter path to your CSV file: ").strip()
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"CSV file not found: {csv_file}")
print(f"Using file: {csv_file}")


# ------------------------------
# Earthdata .netrc Setup
# ------------------------------
print("\nPlease enter your NASA Earthdata Login credentials")
username = input("Username (default=ayan_7869): ").strip() or "ayan_7869"
password = getpass("Password: ")

# Create ~/.netrc locally
netrc_path = os.path.expanduser("~/.netrc")
with open(netrc_path, "w") as f:
    f.write(f"machine urs.earthdata.nasa.gov login {username} password {password}\n")
os.chmod(netrc_path, 0o600)

print(f"Created .netrc at {netrc_path}")


# ------------------------------
# Helpers
# ------------------------------
skipped_rows = []  # global log for skipped/problematic files


def safe_download(session, url, out_path, retries=3, timeout=60):
    """Robust file downloader with retry, timeout, and HTML detection."""
    for attempt in range(retries):
        try:
            with session.get(url, stream=True, allow_redirects=True, timeout=timeout) as resp:
                if resp.status_code == 200:
                    with open(out_path, "wb") as f:
                        for chunk in resp.iter_content(8192):
                            if chunk:
                                f.write(chunk)

                    # sanity check size
                    size = os.path.getsize(out_path)
                    if size < 10_000:
                        with open(out_path, "r", errors="ignore") as fh:
                            head = fh.read(500)
                            if "<html" in head.lower():
                                reason = "Downloaded HTML (login/error page)"
                            else:
                                reason = "Downloaded file too small"
                        skipped_rows.append({"File": os.path.basename(out_path), "Reason": reason})
                        print(f"{os.path.basename(out_path)} skipped → {reason}")
                        os.remove(out_path)
                        return False
                    return True
                else:
                    raise ValueError(f"HTTP {resp.status_code}")
        except Exception as e:
            print(f"Download error ({e}), attempt {attempt+1}/{retries}")
            time.sleep(3)
    skipped_rows.append({"File": os.path.basename(out_path), "Reason": "Download failed"})
    return False


def summarize_netcdf(file_path, date=None, processing_timestamp=None):
    rows = []
    try:
        with nc.Dataset(file_path, 'r') as ds:
            if "latitude" not in ds.variables or "longitude" not in ds.variables:
                reason = "Missing latitude/longitude"
                print(f"{file_path} skipped → {reason}")
                skipped_rows.append({"File": os.path.basename(file_path), "Reason": reason})
                return []

            lat, lon = ds.variables["latitude"][:], ds.variables["longitude"][:]

            if "key_science_data" not in ds.groups:
                reason = "Missing key_science_data group"
                print(f"{file_path} skipped → {reason}")
                skipped_rows.append({"File": os.path.basename(file_path), "Reason": reason})
                return []

            if "column_amount" not in ds.groups["key_science_data"].variables:
                reason = "Missing column_amount variable"
                print(f"{file_path} skipped → {reason}")
                skipped_rows.append({"File": os.path.basename(file_path), "Reason": reason})
                return []

            try:
                var = ds.groups["key_science_data"].variables["column_amount"]
                arr = xp.array(var[:], dtype=float)
                arr[arr < -1e20] = xp.nan
            except MemoryError:
                reason = "MemoryError while loading array"
                print(f"{file_path} skipped → {reason}")
                skipped_rows.append({"File": os.path.basename(file_path), "Reason": reason})
                return []

            if xp.isnan(arr).all():
                reason = "All values NaN"
                print(f"{file_path} skipped → {reason}")
                skipped_rows.append({"File": os.path.basename(file_path), "Reason": reason})
                return []

            # Stats
            mean_val = float(np.nanmean(arr))
            min_val  = float(np.nanmin(arr))
            max_val  = float(np.nanmax(arr))

            try:
                idx_min = int(np.nanargmin(arr))
                idx_max = int(np.nanargmax(arr))
                coords_min, coords_max = np.unravel_index(idx_min, arr.shape), np.unravel_index(idx_max, arr.shape)
                min_lat, min_lon = float(lat[coords_min[0]]), float(lon[coords_min[1]])
                max_lat, max_lon = float(lat[coords_max[0]]), float(lon[coords_max[1]])
            except Exception:
                min_lat = min_lon = max_lat = max_lon = None

            DU_factor = 2.69e16

            rows.append({
                "Dataset": os.path.basename(file_path).split("_")[0],
                "File": os.path.basename(file_path),
                "date": date,
                "processing_timestamp": processing_timestamp,
                "Units": "molecules/cm²",
                "HCHO_Mean (molecules/cm²)": f"{mean_val:.3e}" if not np.isnan(mean_val) else "NA",
                "HCHO_Min (molecules/cm²)": f"{min_val:.3e}" if not np.isnan(min_val) else "NA",
                "HCHO_Max (molecules/cm²)": f"{max_val:.3e}" if not np.isnan(max_val) else "NA",
                "HCHO_Mean (DU)": f"{mean_val/DU_factor:.2f}" if not np.isnan(mean_val) else "NA",
                "HCHO_Min (DU)": f"{min_val/DU_factor:.2f}" if not np.isnan(min_val) else "NA",
                "HCHO_Max (DU)": f"{max_val/DU_factor:.2f}" if not np.isnan(max_val) else "NA",
                "Min_Lat": f"{min_lat:.2f}" if min_lat is not None else "NA",
                "Min_Lon": f"{min_lon:.2f}" if min_lon is not None else "NA",
                "Max_Lat": f"{max_lat:.2f}" if max_lat is not None else "NA",
                "Max_Lon": f"{max_lon:.2f}" if max_lon is not None else "NA",
            })

    except Exception as e:
        print(f"{file_path} skipped → {e}")
        skipped_rows.append({"File": os.path.basename(file_path), "Reason": f"Processing error: {e}"})

    return rows


# ------------------------------
# Stream download + process
# ------------------------------
def stream_process(csv_file, threshold=30):
    df = pd.read_csv(csv_file)
    if "urls" not in df.columns:
        raise ValueError("CSV must have a column named 'urls'")

    if "date" not in df.columns or "processing_timestamp" not in df.columns:
        raise ValueError("CSV must have 'date' and 'processing_timestamp' columns")

    session = requests.Session()
    all_rows = []

    # Prepare results directory
    results_dir = "./results"
    os.makedirs(results_dir, exist_ok=True)

    # Build timestamped output file names
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(results_dir, f"{base_name}_results_{timestamp}.csv")
    skipped_csv = os.path.join(results_dir, f"{base_name}_skipped_{timestamp}.csv")

    output_dir = "./downloads"
    os.makedirs(output_dir, exist_ok=True)

    for i, row in enumerate(tqdm(df.itertuples(), desc="Processing URLs", unit="url")):
        url = row.urls
        date = getattr(row, "date", None)
        processing_timestamp = getattr(row, "processing_timestamp", None)

        fname = os.path.basename(url.split("?")[0])
        out_path = os.path.join(output_dir, fname)

        # Download
        if not safe_download(session, url, out_path):
            continue

        # Summarize
        rows = summarize_netcdf(out_path, date, processing_timestamp)
        all_rows.extend(rows)

        # Cleanup
        try:
            if os.path.exists(out_path):
                os.remove(out_path)
        except Exception as e:
            print(f"Could not delete {out_path}: {e}")

        print(f"Processed {i+1}/{len(df)} → {fname}")

        # Save periodically
        if (i + 1) % threshold == 0 or (i + 1) == len(df):
            try:
                pd.DataFrame(all_rows).to_csv(output_csv, index=False)
                print(f"Saved {len(all_rows)} rows to {output_csv} (up to file {i+1})")
            except Exception as e:
                print(f"Failed saving main CSV: {e}")

            if skipped_rows:
                try:
                    pd.DataFrame(skipped_rows).to_csv(skipped_csv, index=False)
                except Exception as e:
                    print(f"Failed saving skipped log: {e}")

    print(f"\nFinished. Summary saved to {output_csv}")
    if skipped_rows:
        print(f"{len(skipped_rows)} files skipped → {skipped_csv}")


# ------------------------------
# Run
# ------------------------------
stream_process(csv_file, threshold=50)
