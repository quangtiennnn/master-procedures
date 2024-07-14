import os
from transformers import pipeline
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiLineString, LineString
from shapely.ops import nearest_points
import pandas as pd
import pycountry
import pandas as pd
import numpy as np
####################### DATAFRAME SELECTION

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COASTLINE_PATH = os.path.join(BASE_DIR, 'database', 'coastline.json')
PROVINCELINE_PATH = os.path.join(BASE_DIR, 'database', 'diaphantinhenglish.geojson')

# GLOBAL VARIABLE
global coastline_gdf
coastline_gdf_global = gpd.read_file(COASTLINE_PATH)
global geojson_data 
province_gdf_global = gpd.read_file(PROVINCELINE_PATH)



"""
    Use pipeline in transnformers to detect language of comments
    Returns:
        list: language code (eg: 'vi','en','fr')
"""

pipe = pipeline("text-classification", model= "papluca/xlm-roberta-base-language-detection")    
def detect_language(text):
    try:
         result = pipe(text, top_k=1, truncation=True)
         return result[0]['label']
    except:
        return 'unknown'

def create_point(longitude, latitude):
    return Point(longitude, latitude)


def shortestDistance(hotel_coord):
    """
    Use geopandas to read the geometric dataframe and calculate base on norm function
    Returns:
        set: list of min distance to coastline and neareast point on coastline
    """
    coastline_gdf = coastline_gdf_global
    # Convert to a projected coordinate system (e.g., EPSG:3395 for meter-based distances)
    coastline_gdf = coastline_gdf.to_crs(epsg=3395)
    hotel_point = Point(hotel_coord)
    hotel_point = gpd.GeoSeries([hotel_point], crs='EPSG:4326').to_crs(epsg=3395)[0]
    min_distance = float('inf')
    nearest_coastline_point = None
    for geom in coastline_gdf.geometry:
        if geom.geom_type == 'MultiLineString':
            for line in geom.geoms:
                distance = hotel_point.distance(line)
                if distance < min_distance:
                    min_distance = distance
                    nearest_point = nearest_points(hotel_point, line)[1]
        elif geom.geom_type == 'LineString':
            distance = hotel_point.distance(geom)
            if distance < min_distance:
                min_distance = distance
                nearest_point = nearest_points(hotel_point, geom)[1]
    # Convert distance to kilometers
    min_distance_km = min_distance / 1000
    return round(min_distance_km, 3), nearest_point


def is_seaside(distance):
    """
    A function to detect whether the values in the df['dis2coast'] column are less than 1 km and return 1 if they are, otherwise return 0    
    """
    if distance < 1:
        return 1
    else:
        return 0


def map_score_to_sentiment(score):
    if score >= 0 and score < 5:
        return -1
    elif score >= 5 and score < 7:
        return 0
    else:
        return 1



"""
    Create a dictionary mapping country names to their ISO 3166-1 alpha-2 codes for Vietnamese
"""
country_to_code = {
    'Việt Nam': 'VN', 'Malaysia': 'MY', 'Trung Quốc': 'CN', 'Hoa Kỳ': 'US',
    'Philippines': 'PH', 'Vương quốc Anh': 'GB', 'Úc': 'AU', 'Pháp': 'FR',
    'Thụy Điển': 'SE', 'Hàn Quốc': 'KR', 'Yemen': 'YE', 'Thái Lan': 'TH',
    'Singapore': 'SG', 'Canada': 'CA', 'Bồ Đào Nha': 'PT', 'Ấn Độ': 'IN',
    'Campuchia': 'KH', 'Hồng Kông': 'HK', 'Thụy Sĩ': 'CH', 'New Zealand': 'NZ',
    'Đức': 'DE', 'Đài Loan': 'TW', 'Nga': 'RU', 'Na Uy': 'NO', 'Nhật Bản': 'JP',
    'Cộng hòa Séc': 'CZ', 'Indonesia': 'ID', 'Guernsey': 'GG', 'Mông Cổ': 'MN',
    'Phần Lan': 'FI', 'Áo': 'AT', 'Đan Mạch': 'DK', 'Bắc Macedonia': 'MK',
    'Sri Lanka': 'LK', 'Maldives': 'MV', 'Nam Phi': 'ZA', 'Lithuania': 'LT',
    'Bỉ': 'BE', 'Hà Lan': 'NL', 'Tây Ban Nha': 'ES', 'Hungary': 'HU',
    'Lebanon': 'LB', 'Ba Lan': 'PL', 'Israel': 'IL', 'Ý': 'IT', 'Ireland': 'IE',
    'Quần đảo Cayman': 'KY', 'Iceland': 'IS', 'Ukraine': 'UA', 'Myanmar': 'MM',
    'Qatar': 'QA', 'Ả Rập Xê Út': 'SA', 'Colombia': 'CO', 'Nepal': 'NP',
    'Lào': 'LA', 'Estonia': 'EE', 'Rumani': 'RO', 'Brazil': 'BR', 'Algeria': 'DZ',
    'Slovenia': 'SI', 'Mexico': 'MX', 'Brunei Darussalam': 'BN', 'Venezuela': 'VE',
    'Greenland': 'GL', 'Thổ Nhĩ Kỳ': 'TR', 'Slovakia': 'SK', 'Belarus': 'BY',
    'Zimbabwe': 'ZW', 'Chile': 'CL', 'Hy Lạp': 'GR', 'Pakistan': 'PK',
    'Kazakhstan': 'KZ', 'Argentina': 'AR', 'Croatia': 'HR', 'Latvia': 'LV',
    'Eswatini': 'SZ', 'Vanuatu': 'VU', 'Bhutan': 'BT', 'Ma Cao': 'MO', 'Oman': 'OM',
    'Luxembourg': 'LU', 'Nigeria': 'NG', 'Costa Rica': 'CR', 'Bangladesh': 'BD',
    'Jordan': 'JO', 'Afghanistan': 'AF', 'Bulgaria': 'BG', 'Mauritius': 'MU',
    'Kenya': 'KE', 'Vùng đất phía Nam & châu Nam Cực thuộc Pháp': 'TF',
    'Các Tiểu Vương Quốc Ả Rập Thống nhất': 'AE', 'Jersey': 'JE', 'Albania': 'AL',
    'Guam': 'GU', 'Morocco (Ma Rốc)': 'MA', 'Serbia': 'RS', 'Uruguay': 'UY',
    'Monaco': 'MC', 'Tunisia': 'TN', 'Bosnia và Hercegovina': 'BA', 'Armenia': 'AM',
    'Ai Cập': 'EG', 'Ecuador': 'EC', 'Ethiopia': 'ET', 'Zambia': 'ZM',
    'Montenegro': 'ME', 'Turkmenistan': 'TM', 'Malta': 'MT', 'Aruba': 'AW',
    'Tanzania': 'TZ', 'Honduras': 'HN', 'Montserrat': 'MS', 'Bahamas': 'BS',
    'Andorra': 'AD', 'Fiji': 'FJ', 'Papua New Guinea': 'PG', 'Bahrain': 'BH',
    'Bolivia': 'BO', 'Guatemala': 'GT', 'Gabon': 'GA', 'Cô oét (Kuwait)': 'KW',
    'Iraq': 'IQ', 'Peru': 'PE', 'Barbados': 'BB', 'Tân Thế Giới (New Caledonia)': 'NC',
    'Uzbekistan': 'UZ', 'Trinidad và Tobago': 'TT', 'Bermuda': 'BM', 'Cộng hoà Síp': 'CY',
    'Azerbaijan': 'AZ',
    'Quần đảo U.S. Virgin': 'VI', 'Jamaica': 'JM', 'Guadeloupe': 'GP', 'Quần đảo British Virgin': 'VG',
    'Polynesia thuộc Pháp': 'PF', 'Seychelles': 'SC', 'Kyrgyzstan': 'KG', 'Niger': 'NE',
    'Ghana': 'GH', 'Liên bang Micronesia': 'FM', 'Cộng Hòa Congo': 'CG', 'Cộng hòa Trung Phi': 'CF',
    'Angola': 'AO', 'Georgia': 'GE', 'Costa Rica': 'CR', 'Quần đảo Bắc Mariana': 'MP',
    'Martinique': 'MQ', 'Hàn Quốc Anh': 'KR', 'Dominica': 'DM', 'El Salvador': 'SV',
    'Senegal': 'SN', 'Samoa thuộc Mỹ': 'AS', 'Uganda': 'UG', 'Grenada': 'GD',
    'Turkmenistan': 'TM', 'Đông Timor': 'TL', 'Quần đảo Pitcairn': 'PN', 'Rwanda': 'RW',
    'Anguilla': 'AI', 'Namibia': 'NA', 'Gambia': 'GM', 'Đảo Man': 'IM', 'Vùng lãnh thổ Palestine': 'PS',
    'Tonga': 'TO', 'Guatemala': 'GT', 'Liechtenstein': 'LI', 'Bờ Biển Ngà': 'CI',
    'Saint Barthelemy': 'BL', 'Moldova': 'MD', 'Haiti': 'HT', 'Curacao': 'CW',
    'Cộng hòa Dân chủ Congo': 'CD', 'Wallis và Futuna': 'WF', 'Anh': 'GB', 'Đảo Giáng Sinh': 'CX',
    'Mauritania': 'MR', 'Lãnh thổ Ấn Độ Dương thuộc Anh': 'IO'
}

def national_to_iso_code(country,default = 'en'):
    if default == 'en':
        if pd.isna(country):
            return 'unknown'
        try:
            # Lookup country code using pycountry
            return pycountry.countries.lookup(country).alpha_2
        except LookupError:
            return 'unknown'
    else: 
        if pd.isna(country):
            return 'unknown'
        return country_to_code.get(country, 'unknown')        




def detect_province(point, buffer_distance=0.01):
    """
    Detects the province for a given point using the GeoJSON data.

    Parameters:
    point (shapely.geometry.Point): Point containing longitude and latitude of the location.
    buffer_distance (float): Distance by which to buffer the point for expanded search. Default is 0.01 degrees.

    Returns:
    str: The name of the province or 'Unknown' if not found.
    """
    geojson_data = province_gdf_global

    # First, check if the point is within any province
    for _, row in geojson_data.iterrows():
        if row['geometry'].contains(point):
            return row['Name']
    
    # If not found, buffer the point and check again
    buffered_point = point.buffer(buffer_distance)
    for _, row in geojson_data.iterrows():
        if row['geometry'].intersects(buffered_point):
            return row['Name']
    
    return 'unknown'

# Lists of provinces by region
central_vietnamese = [
    'Da Nang', 'Dak Lak', 'Dak Nong', 'Khanh Hoa', 'Kon Tum',
    'Lam Dong', 'Ninh Thuan', 'Phu Yen', 'Quang Binh', 'Quang Nam',
    'Quang Ngai', 'Quang Tri', 'Thua Thien - Hue','Binh Dinh'
]

south_vietnamese = [
    'An Giang', 'Ba Ria - Vung Tau', 'Bac Lieu', 'Ben Tre', 'Binh Duong',
    'Binh Phuoc', 'Binh Thuan', 'Ca Mau', 'Can Tho', 'Dong Nai',
    'Dong Thap', 'Hau Giang', 'Ho Chi Minh City', 'Kien Giang', 'Long An',
    'Soc Trang', 'Tay Ninh', 'Tien Giang', 'Tra Vinh', 'Vinh Long'
]

north_vietnamese = [
    'Bac Giang', 'Bac Kan', 'Bac Ninh', 'Cao Bang', 'Dien Bien', 'Gia Lai',
    'Ha Giang', 'Ha Nam', 'Ha Noi', 'Ha Tinh', 'Hai Duong', 'Hai Phong',
    'Hoa Binh', 'Hung Yen', 'Lai Chau', 'Lang Son', 'Lao Cai', 'Nam Dinh',
    'Nghe An', 'Ninh Binh', 'Phu Tho', 'Quang Ninh', 'Son La', 'Thai Binh',
    'Thai Nguyen', 'Thanh Hoa', 'Tuyen Quang', 'Vinh Phuc', 'Yen Bai'
]

def detect_region(province):
    if province in central_vietnamese:
        return 'Central Vietnam'
    elif province in south_vietnamese:
        return 'South Vietnam'
    elif province in north_vietnamese:
        return 'North Vietnam'
    else:
        return 'unknown'
    


####################### TOPIC MODELING
import string
import emoji
import re

def normalization(text):
    text = emoji.replace_emoji(text, replace='')
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    text = re.sub(r'\s+', ' ', text).strip()

    return text


from pyvi import ViTokenizer
import spacy

nlp = spacy.load('en_core_web_sm')
def word_segmentation(text,language):
    if language == 'vi':
        return ViTokenizer.tokenize(text)
    elif language == 'en':
        doc = nlp(text)
        return " ".join([token.text for token in doc])
    else:
        return text
    
    
def preprocess(df: pd.DataFrame,type = 'sample'):
    print("Preprocesing...")
    if type == 'sample':
        df['processed_comment'] = df['comment'].astype(str).apply(normalization)
        df['processed_comment'] = df.apply(lambda row: word_segmentation(row['processed_text'], row['language']), axis=1)
        return df
    elif type == 'population':
        file_path = 'database/processed_hotel_reviews.csv'
        if os.path.isfile(file_path):
            df = pd.read_csv(file_path, index_col=0)
        else:    
            df['processed_comment'] = df['comment'].astype(str).apply(normalization)
            df['processed_comment'] = df.apply(lambda row: word_segmentation(row['processed_text'], row['language']), axis=1)
            df.to_csv(file_path,index= True,encoding = 'utf-8-sig')
        return df


def embedding(docs,encoder,type='sample'):
    print("Embedding...")
    if type == 'sample':
        corpus_embeddings = encoder.encode(docs, show_progress_bar=False)
        return corpus_embeddings
    elif type == 'population':
        file_path = f'database/corpus_embeddings.npy'
        try:
            # Load pre-computed embeddings if they exist
            corpus_embeddings = np.load(file_path)
        except FileNotFoundError:
            # Compute embeddings if they do not exist
            corpus_embeddings = encoder.encode(docs, show_progress_bar=True)
            # Save embeddings for future use
            np.save(file_path, corpus_embeddings)
        return corpus_embeddings
    
def extract_embedding(df: pd.DataFrame,corpus_embedding):
    return corpus_embedding[df.index]