import pandas as pd
from .tools import *

class Selection:
    def __init__(self, df):
        """
        We extract only the necessary columns to optimize database storage.        
        """
        # df = df.reset_index(drop=True)
        # Hotel Informations:
        self.hotel_id = df['hotel_id']
        self.star_rating = df['star_rating']
        self.coordinate = df.apply(lambda row: create_point(row['longitude'], row['latitude']), axis=1)
        self.province = pd.Series(self.coordinate).apply(detect_province)
        self.region = pd.Series(self.province).apply(detect_region)
        if 'dis2coast' in df.columns:
            self.dis2coast = df['dis2coast']
        else:
            self.distance2coast_calculation()
        self.seaside = pd.Series(self.dis2coast).apply(is_seaside)
        
        # Review Informations:
        """
        Since our dataset is Vietnamese. By default: self.national = df['national'].apply(national_to_iso_code) 
        """
        self.comment = df['comment']
        self.sentiment = df['score'].apply(map_score_to_sentiment)
        self.national = df['national'].apply(national_to_iso_code, args=('vi',))
        
        ## Feature Extraction techniques:
        if 'language' in df.columns:
            self.language = df['language']
        else:
            self.language = self.comment.apply(detect_language)
            
        # DataFrame Visualization:
        self.hotel_review = pd.DataFrame(vars(self))
        self.hotel = self.hotel_review[self.get_hotel_cols()].drop_duplicates()
        
        
    def get_hotel_cols(self):
        return ['hotel_id','province','region','star_rating','coordinate','dis2coast','seaside']
        
        
    def distance2coast_calculation(self, nearest_point=False):
        """
        Nearest point returns the point that is declared as the nearest coastline.        
        """
        if not nearest_point:
            self.dis2coast = [shortestDistance(hotel_coord=point)[0] for point in self.coordinate]
        else:
            self.dis2coast, self.nearest_point = zip(*[shortestDistance(hotel_coord=point) for point in self.coordinate])
        
            
    def filtering(self, extracting = 'hotel',language = None,province=None, star_rating=None, region=None, dis2coast=None, seaside=None):
        """ The function will check if each attribute is provided in the input and then apply the appropriate filters

        Args:
            extracting ('hotel','review'): Declare Dataframe type to extracting. Defaults to 'hotel'. 
            province (list of string, optional): Defaults to None.
            star_rating (list of string, optional): Defaults to None.
            region (list of string, optional): Defaults to None.
            dis2coast (float or list, optional): Defaults to None.
            seaside (boolean, optional): Defaults to None.

        Returns:
            pd.DataFrame: Contains the dataframe with filtering criteria.
        """
        # Storing big DataFrame
        filtered_hotel = self.hotel
        filtered_review = self.hotel_review
        
        # Apply filters based on provided attributes
        if province is not None:
            filtered_hotel = filtered_hotel[filtered_hotel['province'].isin(province)]

        if star_rating is not None:
            filtered_hotel = filtered_hotel[filtered_hotel['star_rating'].isin(star_rating)]

        if region is not None:
            filtered_hotel = filtered_hotel[filtered_hotel['region'].isin(region)]

        if dis2coast is not None:
            """
            Input:
            - [1,5]: From 1 to 5 kilometers
            - 1: From 0 to 1 kilometers
            """
            if isinstance(dis2coast, list) and len(dis2coast) == 2:
                filtered_hotel = filtered_hotel[(filtered_hotel['dis2coast'] >= dis2coast[0]) & (filtered_hotel['dis2coast'] <= dis2coast[1])]
            else:
                filtered_hotel = filtered_hotel[filtered_hotel['dis2coast'] < dis2coast]

        if seaside is not None:
            filtered_hotel = filtered_hotel[filtered_hotel['seaside'] == seaside]

        if extracting == 'hotel':
            return filtered_hotel
        else:
            if language is not None:
                filter_reviews = filtered_review[filtered_review['hotel_id'].isin(filtered_hotel['hotel_id'])]
                return filter_reviews[filter_reviews['language'] == language]
            else:
                return filtered_review[filtered_review['hotel_id'].isin(filtered_hotel['hotel_id'])]        
        
        
