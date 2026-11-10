import pandas as pd
import os
import requests
from tqdm import tqdm
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIGURATION ---
# Get the absolute path of the directory where this script is located
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_FILE_PATH = os.path.join(CURRENT_DIR, 'sdss_catalog.csv')
FITS_DOWNLOAD_DIR = os.path.join(CURRENT_DIR, 'fits_files/')

# --- Download & Retry Configuration ---
MAX_WORKERS = 16  # Number of parallel downloads
MAX_RETRIES = 5  # Number of retries for failed downloads
RETRY_DELAY_SECONDS = 5  # Wait time between retries
CONNECT_TIMEOUT_SECONDS = 30
ASTRA_VERSION = '0.6.0'  # Current Astra version


# ---------------------

def get_spectra_list_from_csv(csv_path):
    """
    Reads the CSV file and returns a set of unique sdss_ids.
    """
    try:
        print(f"Reading data from: {csv_path}")
        df = pd.read_csv(csv_path)

        # Check for sdss_id column
        if 'sdss_id' not in df.columns:
            print(f"[ERROR] Column 'sdss_id' not found in CSV.")
            return set()

        # Drop NaNs and convert to integer
        valid_ids = df['sdss_id'].dropna().astype(int).unique()

        print(f"-> Found {len(valid_ids)} unique sdss_ids to download.")
        return set(valid_ids)

    except FileNotFoundError:
        print(f"[ERROR] File not found at '{csv_path}'.")
        return set()
    except Exception as e:
        print(f"[ERROR] An error occurred while reading CSV: {e}")
        return set()


def download_single_fits(sdss_id, download_dir):
    """
    Downloads a single FITS file based on sdss_id using Astra path logic.
    """
    s_id_str = str(sdss_id)

    # --- ASTRA URL LOGIC ---
    # YY: Last 2 digits
    yy_group = s_id_str[-2:]
    # XX: 4th and 3rd digits from the end
    xx_group = s_id_str[-4:-2]

    # Handle short IDs (padding if necessary)
    if len(s_id_str) < 4:
        s_id_str_padded = s_id_str.zfill(4)
        yy_group = s_id_str_padded[-2:]
        xx_group = s_id_str_padded[-4:-2]

    file_name = f"mwmStar-{ASTRA_VERSION}-{s_id_str}.fits"

    # Construct full URL
    url = f"https://data.sdss.org/sas/dr19/spectro/astra/{ASTRA_VERSION}/spectra/star/{xx_group}/{yy_group}/{file_name}"

    # Local file path
    local_file_path = os.path.join(download_dir, file_name)

    # 1. Check if file already exists
    if os.path.exists(local_file_path):
        # Quick check for file size (> 1KB is likely valid)
        if os.path.getsize(local_file_path) > 1000:
            return sdss_id, True, "Already exists"

    # 2. Retry Loop
    for attempt in range(MAX_RETRIES):
        try:
            # Stream request to save memory
            with requests.get(url, stream=True, timeout=CONNECT_TIMEOUT_SECONDS) as response:

                if response.status_code == 404:
                    return sdss_id, False, f"HTTP 404 Not Found (URL: {url})"

                response.raise_for_status()

                # Write data to file
                with open(local_file_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=32768):  # 32KB chunks
                        if chunk:
                            f.write(chunk)

            return sdss_id, True, "Downloaded"

        except requests.exceptions.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY_SECONDS)
            else:
                return sdss_id, False, f"Failed after {MAX_RETRIES} attempts: {e}"

    return sdss_id, False, "Unexpected error"


if __name__ == '__main__':
    # 1. Get list of IDs
    all_spectra_ids = get_spectra_list_from_csv(CSV_FILE_PATH)
    total_to_process = len(all_spectra_ids)

    if total_to_process == 0:
        print("No IDs to process. Exiting.")
    else:
        # 2. Create download directory
        os.makedirs(FITS_DOWNLOAD_DIR, exist_ok=True)
        print(f"FITS files will be saved to: '{FITS_DOWNLOAD_DIR}'")

        # 3. Start Multi-threaded Download
        print(f"Starting download with {MAX_WORKERS} parallel workers...")

        successful_downloads = 0
        already_existed = 0
        failed_downloads = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            # Submit tasks
            future_to_id = {
                executor.submit(download_single_fits, s_id, FITS_DOWNLOAD_DIR): s_id
                for s_id in all_spectra_ids
            }

            # Progress Bar
            with tqdm(total=total_to_process, desc="Progress", unit="file") as pbar:
                for future in as_completed(future_to_id):
                    spec_id, success, message = future.result()

                    if success:
                        if message == "Downloaded":
                            successful_downloads += 1
                        else:
                            already_existed += 1
                    else:
                        failed_downloads.append((spec_id, message))
                        # Print error without breaking progress bar
                        tqdm.write(f"[ERROR] ID {spec_id}: {message}")

                    pbar.update(1)

        # 4. Summary
        print("\n" + "=" * 40)
        print("DOWNLOAD SUMMARY")
        print("=" * 40)
        print(f"Total files:      {total_to_process}")
        print(f"Successfully DL:  {successful_downloads}")
        print(f"Already existed:  {already_existed}")
        print(f"Failed:           {len(failed_downloads)}")

        # Log errors to file
        if failed_downloads:
            error_log_path = os.path.join(CURRENT_DIR, 'download_errors.txt')
            with open(error_log_path, 'w') as f:
                for fid, msg in failed_downloads:
                    f.write(f"{fid},{msg}\n")
            print(f"\nDetailed error log saved to: {error_log_path}")

        print("\nDone.")
