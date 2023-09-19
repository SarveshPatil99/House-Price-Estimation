import urllib.request
import pickle
from pathlib import Path
import gdown
import zipfile

# Download latitude-longitude data
url = "https://drive.google.com/uc?id=1-5wjM299AmXeCGpciCbMbnUQ1ZnksB_f"
output = "data.zip"
gdown.download(url, output, quiet=False)
with zipfile.ZipFile(output, "r") as zip_ref:
    zip_ref.extractall()
with open('df.pkl', 'rb') as f:
    data = pickle.load(f)
data_np = data[['zpid', 'latitude', 'longitude']].values
# Download satellite images from Google Static API using lat-lon data
Path('satellite_images/').mkdir(parents=True, exist_ok=True)
API_KEY = 'API_KEY_HERE'
for i in range(data_np.shape[0]):
    params = 'center={data_np[i][1]},{data_np[i][2]}&zoom=20&size=640x640&maptype=satellite'
    url = f'https://maps.googleapis.com/maps/api/staticmap?{params}&key={API_KEY}'
    urllib.request.urlretrieve(url, f'satellite_images/{data_np[i][0]}.png')
