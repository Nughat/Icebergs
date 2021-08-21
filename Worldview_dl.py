import urllib
import datetime
import urllib.parse
from csv import DictReader
import cv2
import rasterio
from rasterio.plot import show
from pyproj import Transformer
import requests 

#espg:3031 takes center coordinates 
#construct bounding box around center
#lat -50 lon -40 -> y 3465996.967771 x -2908316.777318
#bbox -3158316.777317909,3215996.9677706333,-2658316.777317909,3715996.9677706333
#bbox1 -50.139089,-44.481553,-49.63791,-35.578835

#layers:
    #MODIS_Terra_CorrectedReflectance_Bands367 
    #MODIS_Terra_CorrectedReflectance_Bands721
    #MODIS_Terra_CorrectedReflectance_TrueColor
    
    #MODIS_Aqua_CorrectedReflectance_TrueColor
    
    #VIIRS_SNPP_CorrectedReflectance_TrueColor
    #VIIRS_NOAA20_CorrectedReflectance_TrueColor
    #VIIRS_NOAA20_CorrectedReflectance_BandsM11-I2-I1   
#time: enter YYYYMMDD
#bbox: enter lat/lon coord as tuple
#width&height (resolution: 250 m): 500x500 km -> 1172x1172 pixels

def tiffs(output,layer,time,latlon): 
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031") 
    x,y = transformer.transform(latlon[0],latlon[1]) #convert lat/lon to polar stereographic
    
    bbox = [x-150000,y-150000,x+150000,y+150000]
    BASE_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS={lyr}&CRS=EPSG:3031&TIME={tim}&WRAP=DAY&BBOX={bbox}&FORMAT=image/tiff&WIDTH=1172&HEIGHT=1172"
    dl = BASE_URL.format(
        lyr = layer,
        tim = time,
        bbox = ",".join([str(v) for v in bbox]),
        )
    r = requests.get(dl)
    if r.status_code == 200:
        if 'xml' in r.text[:40]:
            print(dl)
            raise Exception(r.content)
        else:
            with open(output, 'wb') as fh:
                fh.write(r.content)
    else:
        raise Exception(r.status)
        
def convertDate(date):
    daynum = date[4:]
    year = date[0:4]
    adj = daynum.rjust(3 + len(daynum), '0')
    res = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j").strftime("%Y-%m-%d")
    return(res)

def downloadCSV(file,path,output,timeRange=None):
    with open(file, 'r') as read_obj:
        csv_dict_reader = DictReader(read_obj)
        for row in csv_dict_reader:
            if (int(row['date']) <= int(timeRange[1])) and (int(row['date']) >= int(timeRange[0])):
                print(row['date'], row['lat'], row['lon'])
                dl = tiffs(path+output+"MODT367_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_Bands367',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(path+output+"MODT721_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_Bands721',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(path+output+"MODT_TRUE_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(path+output+"MODA_TRUE_"+convertDate(row['date']),'MODIS_Aqua_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(path+output+"VIR_SNPP_TRUE_"+convertDate(row['date']),'VIIRS_SNPP_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(path+output+"VIR_NOAA_TRUE_"+convertDate(row['date']),'VIIRS_NOAA20_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(path+output+"VIR_NOAA_M11I2I1_"+convertDate(row['date']),'VIIRS_NOAA20_CorrectedReflectance_BandsM11-I2-I1',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
    print('done')