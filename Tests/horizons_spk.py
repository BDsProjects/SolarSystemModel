import sys
import json
import base64
import requests

# Define API URL and SPK filename:
url = 'https://ssd.jpl.nasa.gov/api/horizons.api'
spk_filename = 'spk_file.bsp'

# Define the time span:
start_time = '2030-01-01'
stop_time = '2031-01-01'

# Get the requested SPK-ID from the command-line:
if (len(sys.argv)) == 1:
  print("please specify SPK-ID on the command-line");
  sys.exit(2)
spkid = sys.argv[1]

# Build the appropriate URL for this API request:
# IMPORTANT: You must encode the "=" as "%3D" and the ";" as "%3B" in the
#            Horizons COMMAND parameter specification.
url += "?format=json&EPHEM_TYPE=SPK&OBJ_DATA=NO"
url += "&COMMAND='DES%3D{}%3B'&START_TIME='{}'&STOP_TIME='{}'".format(spkid, start_time, stop_time)

# Submit the API request and decode the JSON-response:
response = requests.get(url)
try:
  data = json.loads(response.text)
except ValueError:
  print("Unable to decode JSON results")

# If the request was valid...
if (response.status_code == 200):
  #
  # If the SPK file was generated, decode it and write it to the output file:
  if "spk" in data:
    #
    # If a suggested SPK file basename was provided, use it:
    if "spk_file_id" in data:
      spk_filename = data["spk_file_id"] + ".bsp"
    try:
      f = open(spk_filename, "wb")
    except OSError as err:
      print("Unable to open SPK file '{0}': {1}".format(spk_filename, err))
    #
    # Decode and write the binary SPK file content:
    f.write(base64.b64decode(data["spk"]))
    f.close()
    print("wrote SPK content to {0}".format(spk_filename))
    sys.exit()
  #
  # Otherwise, the SPK file was not generated so output an error:
  print("ERROR: SPK file not generated")
  if "result" in data:
    print(data["result"])
  else:
    print(response.text)
  sys.exit(1)

# If the request was invalid, extract error content and display it:
if (response.status_code == 400):
  data = json.loads(response.text)
  if "message" in data:
    print("MESSAGE: {}".format(data["message"]))
  else:
    print(json.dumps(data, indent=2))

# Otherwise, some other error occurred:
print("response code: {0}".format(response.status_code))
sys.exit(2)
