import urllib.parse
import datetime
from csv import DictReader
from pyproj import Transformer
import requests

def tiffs(output,layer,time,latlon): 
 
    """Download Antarctic geotiffs (ESPG:3031) from NASA Worldview Snapshots tool: https://wvs.earthdata.nasa.gov/
       
       Borrowed with some modifications from:

        ***************************************************************************************
        *    Title: worldview_dl
        *    Author: leifdenby
        *    Date: Jan 29, 2020
        *    Code version: 2.0
        *    Availability: https://github.com/leifdenby/worldview_dl
        *
        ***************************************************************************************
        
        Args:
            output (str): output file
            layer (str): satellite product, list of layers: https://www.arcgis.com/home/item.html?id=cf930f39aecc464d8d6132656102faf4
            time (str): YYYY-MM-DD format
            latlon (tuple of floats): latitude and longitude of center of geotiff
            
        Resolution set to 250m
        Later version will take distance (km) as arg
        
        Return:
            None, geotiff is downloaded  
        
        Eg:
           tiffs(home/MODT367_2020-11-28,'MODIS_Terra_CorrectedReflectance_Bands367','2020-11-28',(-83,50))) 
    """
    
    transformer = Transformer.from_crs("epsg:4326", "epsg:3031") #convert from latlon to polar stereographic coordinates
    x,y = transformer.transform(latlon[0],latlon[1])
    bbox = [x-150000,y-150000,x+150000,y+150000] #bounding box is a 15km x 15km square around center coordinate
    BASE_URL = "https://wvs.earthdata.nasa.gov/api/v1/snapshot?REQUEST=GetSnapshot&LAYERS={lyr}&CRS=EPSG:3031&TIME={tim}&WRAP=DAY&BBOX={bbox}&FORMAT=image/tiff&WIDTH=1172&HEIGHT=1172" 
    dl = BASE_URL.format(  
        lyr = layer,
        tim = time,
        bbox = ",".join([str(v) for v in bbox]),
        ) #fill base url with arguements
    r = requests.get(dl) #download image from url
    if r.status_code == 200: #error handling
        if 'xml' in r.text[:40]:
            print(dl)
            raise Exception(r.content)
        else:
            with open(output, 'wb') as fh:
                fh.write(r.content)
    else:
        raise Exception(r.status)
        
def convertDate(date):
    """Convert date from day-of-year format to YYY-MM-DD format
    
        Args:
            date (str): day-of-year format
            
        Return:
            res (str): YYY-MM-DD format  
        Eg: convertdate('2020364')
    """
    daynum = date[4:] #day of year
    year = date[0:4] #year
    res = datetime.datetime.strptime(year + "-" + daynum, "%Y-%j").strftime("%Y-%m-%d") #convert 
    return(res) #return YYYY-MM-DD format

def downloadCSV(file,output,timeRange=None):
    """Use BYU data to download these layers:
            MODIS_Terra_CorrectedReflectance_Bands367
            MODIS_Terra_CorrectedReflectance_Bands721
            MODIS_Terra_CorrectedReflectance_TrueColor
            MODIS_Aqua_CorrectedReflectance_TrueColor
            VIIRS_SNPP_CorrectedReflectance_TrueColor
            VIIRS_NOAA20_CorrectedReflectance_TrueColor
            VIIRS_NOAA20_CorrectedReflectance_BandsM11-I2-I1
            
        Args:
            file (str): BYU csv file
            output (str): output path
            timeRange (tuple of str): for select time ranges in csv file, optional
            
        Return: 
            None, geotiffs are downloaded     
            
        Eg:
            downloadCSV('a68/a68f.csv',"/nbhome/Nuzhat.Khan/a68/a68f/",('2020360','2020364'))
    """
    with open(file, 'r') as read_obj: #open csv
        csv_dict_reader = DictReader(read_obj) #read csv
        for row in csv_dict_reader:
            if (timeRange != None): #if time range is specified
                if (int(row['date']) <= int(timeRange[1])) and (int(row['date']) >= int(timeRange[0])): #dowanload from earliest to latest date
                    print(row['date'])
                    dl = tiffs(output+"MODT367_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_Bands367',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                    dl = tiffs(output+"MODT721_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_Bands721',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                    dl = tiffs(output+"MODT_TRUE_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                    dl = tiffs(output+"MODA_TRUE_"+convertDate(row['date']),'MODIS_Aqua_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                    dl = tiffs(output+"VIR_SNPP_TRUE_"+convertDate(row['date']),'VIIRS_SNPP_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                    dl = tiffs(output+"VIR_NOAA_TRUE_"+convertDate(row['date']),'VIIRS_NOAA20_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                    dl = tiffs(output+"VIR_NOAA_M11I2I1_"+convertDate(row['date']),'VIIRS_NOAA20_CorrectedReflectance_BandsM11-I2-I1',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
            else: #else download all dates is csv
                print(row['date']) 
                dl = tiffs(output+"MODT367_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_Bands367',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(output+"MODT721_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_Bands721',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(output+"MODT_TRUE_"+convertDate(row['date']),'MODIS_Terra_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(output+"MODA_TRUE_"+convertDate(row['date']),'MODIS_Aqua_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(output+"VIR_SNPP_TRUE_"+convertDate(row['date']),'VIIRS_SNPP_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(output+"VIR_NOAA_TRUE_"+convertDate(row['date']),'VIIRS_NOAA20_CorrectedReflectance_TrueColor',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
                dl = tiffs(output+"VIR_NOAA_M11I2I1_"+convertDate(row['date']),'VIIRS_NOAA20_CorrectedReflectance_BandsM11-I2-I1',convertDate(row['date']),(float(row['lat']),float(row['lon'])))
    print('done')