import requests
from tqdm import tqdm

login_url = "https://cloudstore.uni-ulm.de"
l2 = "https://urldefense.com/v3/__https://hci.iwr.uni-heidelberg.de/node/6132/download/91c8592a075171860bc19ca5e6684f18__;!!KwNVnqRv!G9x8bphjyVoRB0nLgAAx1uPgVHtnmlNGmbt5Hi6t98UXlkxU4R_90zLBpHMlw1GW4R6UlocZrgLOfQSjCNS1HznWFbIHYcCgPVz7QUaB$"
# download_url = "https://cloudstore.uni-ulm.de/training/DTLD/Essen.zip"  # Replace with the URL of the zip file you want to download
# save_path = "/mnt/c/Users/austi/Downloads/Essen.zip"  # Replace with the desired save path and file name
download_url = 'https://hci.iwr.uni-heidelberg.de/system/files/private/datasets/1883532915/dataset_test_rgb.zip.006'
save_path = '/mnt/c/Users/austi/Downloads/dataset_test_rgb.zip.006'

# Hannover
# Kassel
# Koeln

login_data = {
    "username": "dtld",
    "password": "pv06-X72u-RwNS-ocl8"
}

CHUNK_SIZE = 8192  # Size of each chunk in bytes  8MB
with requests.Session() as session:
    login_response = session.get(l2)
    login_response.raise_for_status()

    download_response = session.get(download_url, stream=True) # , auth=(login_data['username'], login_data["password"])

    total_size = int(download_response.headers.get("content-length", 0))
    progress_bar = tqdm(total=total_size, unit="B", unit_scale=True)
    download_response.raise_for_status()  # Check for any download errors

    with open(save_path, "wb") as file:
        for chunk in download_response.iter_content(CHUNK_SIZE):
            file.write(chunk)
            progress_bar.update(len(chunk))

    progress_bar.close()

print("File downloaded successfully!")
