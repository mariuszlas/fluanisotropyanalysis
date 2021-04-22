import csv
import re
import string
import math
import warnings
import pandas as pd
import numpy as np
import ipywidgets as wg
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mtick
from itertools import product
from scipy.optimize import curve_fit
from IPython.display import display
from platemapping import plate_map as pm

# define custom errors
class DataError(Exception):
    pass

class PlateSizeError(Exception):
    pass

class DataTypeError(Exception):
    pass

# define well plate dimensions
plate_dim = {96:(8, 12), 384:(16, 24)}

# define header names for platemapping module
pm.header_names = {'Well ID': {'dtype':str, 'long':True, 'short_row': False, 'short_col':False},
                'Type': {'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                'Contents': {'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                'Protein Name': {'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                'Protein Concentration': {'dtype':float, 'long':True, 'short_row': True, 'short_col':True},
                'Tracer Name': {'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                'Tracer Concentration': {'dtype':float, 'long':True, 'short_row': True, 'short_col':True},
                'Competitor Name': {'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                'Competitor Concentration': {'dtype':float, 'long':True, 'short_row': True, 'short_col':True},
                'Concentration Units':{'dtype':str, 'long':True, 'short_row': True, 'short_col':True},
                }

class FA:
    """Class used for the analysis of fluorescence anisotropy data.
    
    :param data_dict: A dictionary contaning data frames with pre-processed data and metadata
    :type data_dict: dict
    :param g_factor: A value of g-factor
    :type g_factor: float 
    :param plate_map: A data frame with platemap containing information about each well
    :type plate_map: pandas df"""
    
    def __init__(self, data_dict, g_factor, plate_map):
        self.data_dict = data_dict
        self.g_factor = g_factor
        self.plate_map = plate_map
       
        # create list of all p and s data frames to run some stats
        frames = []   
        for repeat in self.data_dict.values():   
            metadata, data = repeat.values()
            p_channel, s_channel = data.values()
            frames.append(p_channel)
            frames.append(s_channel)
    
        new = pd.concat(frames, axis=1)   # join all p and s data frames into one df
        nan = new.size - new.describe().loc['count'].sum()   # find sum of 'nan' cells
        
        # create a data frame to store the final fitting parameters
        col_names = ['rmin', 'rmin error', 'rmax', 'rmax error', 'lambda', 'Kd', 'Kd error']
        p_names = self.plate_map['Protein Name'].dropna().unique()   # get list of all protein names
        t_names = self.plate_map['Tracer Name'].dropna().unique()   # get list of all tracer names 
        c_names = self.plate_map['Competitor Name'].dropna().unique()   # get list of all competitor names
        if len(c_names) == 0:   # if there are no comeptitors, replace nan with a string
            c_names = ['-']
            c_names_print = 'None'
        else:
            c_names_print = c_names
        
        final_fit = pd.DataFrame(index=pd.MultiIndex.from_product([p_names, t_names, c_names]), columns=col_names)  
        final_fit["lambda"] = 1   # set the default lambda value as 1
        self.final_fit = final_fit  
            
        print("Data was uploaded!\n")
        print(f"Number of repeats: {len(self.data_dict)} \nValue of g-factor: {self.g_factor} \nOverall number of empty cells is {int(nan)} in {len(frames)} data frames.\nProteins: {p_names}\nTracers: {t_names}\nCompetitors: {c_names_print}\n")
              
              
    @classmethod
    def read_in_envision(cls, data_csv, platemap_csv, data_type='plate', size=384):
        """Reads in the raw data from csv file along with a platemap and constructs the FA class boject.
        
        :param data_csv: File path of the raw data file in .csv format.
        :type data_csv: str
        :param platemap_csv: File path of the platemap file in .csv format.
        :type platemap_csv: str
        :param data_type: Format in which the raw data was exported (plate or list), defaults to plate.
        :type data_type: str
        :param size: Size of the well plate (384 or 96), defaults to 384.
        :type size: int
        :return: A dictionary contaning data frames with pre-processed data, g-factor and data frame containing platemap.
        :rtype: dict, float, pandas df """
        
        # ensure the plate size is either 384 or 96
        if size not in plate_dim:
            raise PlateSizeError('Invalid size of the well plate, should be 384 or 96.')
        
        # try to read in data in plate format
        if data_type == 'plate':
            try:
                data_dict, g_factor = FA._read_in_plate(data_csv, size)   # get data dictionary and g factor
                plate_map_df = pm.plate_map(platemap_csv, size)   # get platemap using the platemapping module
                return cls(data_dict, g_factor, plate_map_df)
            
            except (UnboundLocalError, IndexError, ValueError):
                raise DataError(f"Error occured during data read in. Check your file contains data in the 'plate' format and plate size is {size}.")
        
        # try to read in data in list format
        if data_type == 'list':
            try:
                data_dict, g_factor = FA._read_in_list(data_csv, size)   # get data dictionary and g factor
                plate_map_df = pm.plate_map(platemap_csv, size)   # get platemap using the platemapping module
                return cls(data_dict, g_factor, plate_map_df)
            
            except (UnboundLocalError, IndexError):
                raise DataError("Error occured during data read in. Check your file contains data in the 'list' format.")
        
        else:
            raise DataTypeError(f"'{data_type}' is not one of the two valid data types: plate or list.")
    

    def _read_in_plate(csv_file, size):
        """Reads the raw data file and finds the information needed to extract data. Passes those parameters to pre_process_plate function and executes it.
        Returns a tuple of two elemnts: dictionary of data frames and g-factor.

        :param csv_file: File path of the raw data file in .csv format
        :type csv_file: str
        :param well_ids: A list of well IDs for the pre-processed data frames
        :type well_ids: list
        :return: A tuple of dictionary of data frames and the g-factor 
        :rtype: pandas df, float """
        
        with open(csv_file) as file:
            all_data_lines = list(csv.reader(file, delimiter=','))   # read the csv file and cast it into a list containing all lines

        blank_indexes = list(index for index, item in enumerate(all_data_lines) if item == [])   # list containing indices of all blank rows
        if blank_indexes == []:   # case for the raw data file having commas instead of blank spaces
            blank_indexes = list(index for index, item in enumerate(all_data_lines) if set(item) == {''})   # treats a line filled only with commas (empty strings) as balnk
        blanks = np.array(blank_indexes)   # convert the list of blank indices to a numpy array
        read_in_info = []   # list to store the tuples with parameters needed for pandas to read in the csv file

        for index, item in enumerate(all_data_lines):   # iterate over list with all lines in the csv file
            
            if item != [] and re.findall(r"Plate information", item[0]) == ['Plate information'] and re.search(r'Results for', all_data_lines[index + 9][0]) == None and re.findall(r"Formula", all_data_lines[index+1][10]) != ['Formula']:
                skiprows = index + 9     # Set the skiprows parameter for raw data table
                skiprows_meta = index + 1     # Set the skiprows parameter for metadata table
                end_of_data = blanks[blanks > skiprows].min()     # Calculate the end of data table by finding the smallest blank index after the beginning of data table
                read_in_info.append((skiprows, end_of_data - skiprows + 1, skiprows_meta))     # add the skiprows, caculated number of data lines and skiprows for metadata parameters to the list as a tuple
                data_format = 'plate1'

            if item != [] and re.findall(r"Plate information", item[0]) == ['Plate information'] and re.search(r'Results for', all_data_lines[index + 9][0]) != None:
                skiprows = index + 10     # Set the skiprows parameter for raw data table
                skiprows_meta = index + 1     # Set the skiprows parameter for metadata table
                end_of_data = blanks[blanks > skiprows].min()     # Calculate the end of data table by finding the smallest blank index after the beginning of data table
                read_in_info.append((skiprows, end_of_data - skiprows - 1, skiprows_meta))     # add the skiprows, caculated number of data lines and skiprows for metadata parameters to
                data_format = 'plate2'

            if item != [] and len(item) > 1 and re.fullmatch(r"G-factor", item[0]):
                g_factor = float(item[4])   
        
        return FA._pre_process_plate(csv_file, read_in_info, data_format, size), g_factor

    def _pre_process_plate(csv_file, read_in_info, data_format, size):    
        """Extracts the data and metadata from the csv file, processes it and returns a nested dictionary containing data and metadata for each repeat and channel.

        :param csv_file: File path of the raw data file in .csv format
        :type csv_file: str
        :param read_in_info: Tuples with read in parameters for each channel.
        :type read_in_info: list
        :param data_format: Plate type (plate1 or plate2)
        :type data_format: str
        :param well_ids: A list of well IDs for the pre-processed data frames
        :type well_ids: list
        :return: A dictionary containing data and metadata 
        :rtype: dict """ 
        
        data_frames = {}   # dictionary to store data frames
        counter = 1   # counter incremented by 0.5 to enable alternating labelling of data frames as 'p' or 's'
        row_letters = list(string.ascii_uppercase)[0: plate_dim[size][0]]   # list of letters for well IDs
        col_numbers = list(np.arange(1, plate_dim[size][1] + 1).astype(str))   # list of numbers for well IDs
        well_ids = ['%s%s' % (item[0], item[1]) for item in product(row_letters, col_numbers)]   # list of well IDs for the pre-processed data frames
        
        for index, item in enumerate(read_in_info):   # iterate over all tuples in the list, each tuple contains skiprows, nrows and skiprows_meta for one channel 

            if data_format == 'plate1':   # raw data table does not have row and column names so 'names' parameter passed to omit the last column
                raw_data = pd.read_csv(csv_file, sep=',', names=col_numbers, index_col=False, engine='python', skiprows=item[0], nrows=item[1], encoding='utf-8')

            if data_format == 'plate2':   # raw data table has row and column names, so index_col=0 to set the first column as row labels
                raw_data = pd.read_csv(csv_file, sep=',', index_col=0, engine='python', skiprows=item[0], nrows=item[1], encoding='utf-8')
                if len(raw_data.columns) in [13, 25]:    
                    raw_data.drop(raw_data.columns[-1], axis=1, inplace=True)    # delete the last column because it is empty

            # generate df for metadata (number of rows is always 1) and convert measurement time into datetime object   
            metadata = pd.read_csv(csv_file, sep=',', engine='python', skiprows=item[2], nrows=1, encoding='utf-8').astype({'Measurement date': 'datetime64[ns]'})
            # convert and reshape data frame into 1D array
            data_as_array = np.reshape(raw_data.to_numpy(), (int(size), 1)) 

            if counter % 1 == 0:   
                new_data = pd.DataFrame(data=data_as_array, index=well_ids, columns=['p'])   # generate new 384 (or 96) by 1 data frame with p channel data
                data_frames[f'repeat_{int(counter)}'] = {'metadata':metadata, 'data': {'p': new_data, 's':''}}   # add p channel data and metadata dfs to dictionary

            if counter % 1 != 0:
                new_data = pd.DataFrame(data=data_as_array, index=well_ids, columns=['s'])   # generate new 384 (or 96) by 1 data frame with s channel data
                data_frames[f'repeat_{int(counter-0.5)}']['data']['s'] = new_data   # add s channel data to dictionary

            counter = counter + 0.5
        
        return data_frames


    def _read_in_list(csv_file, size):
        """Reads the raw data file and extracts the data and metadata. Passes the raw data to pre_process_list function and executes it.
        Returns a tuple of two elemnts: dictionary of data frames and g-factor.

        :param csv_file: File path of the raw data file in .csv format
        :type csv_file: str
        :param well_ids: A list of well IDs for the pre-processed data frames
        :type well_ids: list
        :return: A tuple of dictionary of data frames and the g-factor
        :rtype: tuple """

        with open(csv_file) as file:  
            all_data_lines = list(csv.reader(file, delimiter=',')) # read the csv file and cast it into a list containing all lines
 
        blank_indexes = list(index for index, item in enumerate(all_data_lines) if item == [])   # list containing indexes of all blank rows
        if blank_indexes == []:   # case for the raw data file having commas instead of blank spaces
            blank_indexes = list(index for index, item in enumerate(all_data_lines) if set(item) == {''})   # treats a line filled only with commas (empty strings) as balnk
        blanks = np.array(blank_indexes)   # convert the list of blank indexes to a numpy array
        
        # iterate over all lines to find beggining of the data table ('skiprows') and determine the format of data  (list A, B, or C)
        for index, item in enumerate(all_data_lines):   
            
            if item != [] and len(item) == 1 and re.findall(r"Plate information", item[0]) == ["Plate information"]:
                skiprows_meta = index + 1
                end_of_metadata = blanks[blanks > skiprows_meta].min()   # find the end of metadata by finding the smallest blank index after the beginning of metadata
                
            if item != [] and len(item) >= 2 and re.findall(r"PlateNumber", item[0]) == ['PlateNumber'] and re.findall(r"PlateRepeat", item[1]) == ['PlateRepeat']:   # find line number with the beggining of the data
                skiprows = index - 1
                data_format = 'listA'
                end_of_data = blanks[blanks > skiprows].min()

            if item != [] and len(item) >= 2 and re.findall(r"Plate", item[0]) == ['Plate'] and re.findall(r"Barcode", item[1]) == ['Barcode']:   # find line number with the beggining of the data
                skiprows = index
                data_format = 'listB'
                end_of_data = blanks[blanks > skiprows].min()

            if item != [] and len(item) >= 2 and re.findall(r"Plate", item[0]) == ['Plate']  and re.findall(r"Well", item[1]) == ['Well']:
                skiprows = index
                data_format = 'listC'
                end_of_data = blanks[blanks > skiprows].min()

            if item != [] and re.fullmatch(r"G-factor", item[0]):   # find the g factor
                g_factor = float(item[4])

        nrows = end_of_data - skiprows - 1   # calculate the length of data table
        nrows_meta = end_of_metadata - skiprows_meta - 1   # calucalte the length of metadata table (number of rows depends on the number of repeats)

        raw_data = pd.read_csv(csv_file, sep=',', engine='python', skiprows=skiprows, nrows=nrows, encoding='utf-8')
        raw_metadata = pd.read_csv(csv_file, sep=',', engine='python', skiprows=skiprows_meta, nrows=nrows_meta, encoding='utf-8')

        return FA._pre_process_list(raw_data, raw_metadata, data_format, size), g_factor

    def _pre_process_list(raw_data, raw_metadata, data_format, size):
        """Extracts the data and metadata for each channel and repeat from the raw data and raw metadata 
        and returns a nested dictionary containing data and metadata for each repeat and channel.

        :param raw_data: Data frame containing raw data
        :type raw_data: pandas data frame
        :param raw_metadata: Data frame containing raw metadata
        :type raw_metadata: pandas data frame
        :param data_format: Type of list (listA, listB, or listC)
        :type data_format: str
        :param well_ids: A list of well IDs for the pre-processed data frames
        :type well_ids: list
        :return: A dictionary containing data and metadata
        :rtype: dict"""

        # remove the '0' from middle position of well numbers (A01 -> A1), done by reassigning the 'Well' column to a Series containing modified well numbers
        raw_data['Well'] = raw_data['Well'].apply(lambda x: x[0] + x[2] if x[1] == '0' else x)
        
        data_frames = {}   # dictionary to store data frames
        repeats = list(raw_metadata['Repeat'].to_numpy())   # generate a list with repeats based on the metadata table, e.g. for 3 repeats -> [1,2,3]
        row_letters = list(string.ascii_uppercase)[0: plate_dim[size][0]]   # list of letters for well IDs
        col_numbers = list(np.arange(1, plate_dim[size][1] + 1).astype(str))   # list of numbers for well IDs
        well_ids = ['%s%s' % (item[0], item[1]) for item in product(row_letters, col_numbers)]   # list of well IDs for the pre-processed data frames
        
        for index, repeat in enumerate(repeats):   # iterate over the number of repeats
            
            if data_format == 'listA':
                
                groupped_data = raw_data.groupby(raw_data.PlateRepeat).get_group(repeat)   # group and extract the data by the plate repeat column, i.e. in each iteration get data only for the current repeat 
                p_groupped = groupped_data.iloc[::3, :]   # extract data only for the p channel, i.e. each third row starting from the first row
                s_groupped = groupped_data.iloc[1::3, :]   # extract data only for the s channel, i.e. each third row starting from the second row
                p_raw_data = p_groupped[['Well', 'Signal']]   # extract only the two relevant columns
                s_raw_data = s_groupped[['Well', 'Signal']]   # for each channel

            if data_format in ['listB', 'listC']: 
                
                # the column naming is different for the first repeat ('Signal'), then it's 'Signal.1', 'Signal.2', etc.
                if repeat == 1: 
                    p_raw_data = raw_data[['Well', 'Signal']]   
                    s_raw_data = raw_data[['Well', f'Signal.{repeat}']]
                else:
                    p_raw_data = raw_data[['Well', f'Signal.{repeat + index - 1}']]   # the column cotntaining data to be extracted is calculated in each iteration
                    s_raw_data = raw_data[['Well', f'Signal.{repeat + index}']]
            
            # create an empty df with no columns and indexes matching the plate size
            indexes = pd.DataFrame(well_ids, columns=['Wells'])
            empty_frame = indexes.set_index('Wells')
            
            p_raw_data.set_index('Well', inplace=True)   # set the row indexes as the well numbers
            p_raw_data.set_axis(['p'], axis=1, inplace=True)   # rename the 'Signal' column to 'p'
            p_data = empty_frame.join(p_raw_data)   # join the raw data df to an empty frame based on the indexes, assigns 'NaN' to indexes not present in the raw data table
            
            s_raw_data.set_index('Well', inplace=True) 
            s_raw_data.set_axis(['s'], axis=1, inplace=True)
            s_data = empty_frame.join(s_raw_data)
    
            metadata = raw_metadata.iloc[[repeat-1]].astype({'Measurement date': 'datetime64[ns]'})   # extract the row with metadata relevant for each repeat and covert date and time into a datetime object
            data_frames[f'repeat_{repeat}'] = {'metadata': metadata, 'data': {'p': p_data, 's': s_data}}   # add data frames to the dictionary

        return data_frames
    
    
    def visualise(self, labelby='Type', colorby='Type', title="", cmap='rainbow', blank_yellow=True, scale='lin', dpi=250, export=False):
        """Returns a visual representation of the plate map.
        
        The label and colour for each well can be customised to be a platemap variable, for example 'Type', 'Protein Name', 'Protein Concentration', etc.
        It can also be the p or s channel value, calculated anisotropy or intensity, however in such cases the 'colorby' or 'labelby'
        parameters must be passed as tuple of two strings specifying the repeat number and variable to display, for example ('repeat_2', 'p_corrected').
        
        :param labelby: Variable to display on the wells, for example 'Type', 'Protein Name', ('repeat_1', 's_corrected'), defaults to 'Type'.
        :type labelby: str or tuple of str
        :param colorby: Variable to color code by, for example 'Type', 'Contents', 'Protein Concentration', ('repeat_2', 'p'), for non-categorical data the well coulour represnets the magnitude of the number, defaults to 'Type'.
        :type colorby: str or tuple of str
        :param title: Sets the title of the figure, defaults to None.
        :type title: str
        :param cmap: Sets the colormap for the color-coding, defaults to 'rainbow'.
        :type cmap: str
        :param blank_yellow: Sets the colour-coding of blank wells as yellow, defaults to True.
        :type blank_yellow: bool
        :param scale: Determines whether data for colour-coding of non-categorical data (e.g. 'p_chanel', 'r_corrected') is scaled linearly ('lin') or logarithmically ('log', works only if data does not contain values less than or equal 0), wdefaults to 'lin'. 
        :type scale: str
        :param dpi: Resolution of the exported figure in points per inches, defaults to 250.
        :type dpi: int
        :param export: If True, save the figure as .png file, defaults to False.
        :type export: bool
        :return: Visual representation of the plate map.
        :rtype: figure
        """
        plate_map = self.plate_map   # default platemap
        size = plate_map.shape[0]   
        str_format, str_len = None, None   # default string format and lengh (used for categorical types, e.g. 'Type', 'Protein Name', etc.)
        noncat_vars = ['p','s','p_corrected','s_corrected','r_raw','r_corrected','i_raw','i_corrected','i_percent']   # list of non-categorical data
        scinot_vars = noncat_vars[:-1] + ['Protein Concentration', 'Tracer Concentration', 'Competitor Concentration']  # types that may have to be formatted in scinot (all non-categorical types except of i_percent)
        
        if type(labelby) == tuple:   # option for labelling by the a variable and its repeat number
            plate_map = self.plate_map.join(self.data_dict[labelby[0]]['data'][labelby[1]])   # data frame containing variable from specified repeat is added to the platemap
            labelby = labelby[1]   # reassign labelby as the variable name 
            if labelby == 'i_percent':   
                str_format = 'percent'    # display the values to 1 decimal place
                str_len = 3    # determine the length of string to avoid issues with incorrect font scaling

        if type(colorby) == tuple:   # option for colouring by the a variable and its repeat number
            plate_map = self.plate_map.join(self.data_dict[colorby[0]]['data'][colorby[1]])   # data frame containing variable from specified repeat is added to the platemap
            colorby = colorby[1]    # reassign colorby as the variable name
            
        if labelby in scinot_vars:   # check if the data needs to be displyed in scientific notation
            if sum((plate_map[labelby] > 1000) | (plate_map[labelby] < 0)) > 0:   # format in sci notation if the number is greater than 1000 or less than 0
                str_format = 'scinot'
                str_len = 8     # determine the length of string to avoid issues with incorrect font scaling

        if colorby in noncat_vars:
            categorical = False     # colours for colour-coding are generated based on normalised data from colorby column
        else:
            categorical = True     # colurs for colour-coding are generated based on an array of uniformally spaced numbers representing each category

        return pm.visualise(plate_map, title, size, export, cmap, colorby, labelby, dpi, str_format=str_format, str_len=str_len, blank_yellow=blank_yellow, scale=scale, categorical=categorical)
    
    
    def invalidate(self, valid=False, **kwargs):
        """Invalidates wells, entire columns and/or rows. Any of the following keyword arguments, or their combination, 
        can be passed: wells, rows, columns. For example, to invalidate well A1, rows C and D and columns 7 and 8 execute  
        the following: invalidate(wells='A1', rows=['C','D'], columns=[7,8]).
        To validate previously invalidated wells, rows and/or columns, pass the additional 'valid' argument as True.
    
        :param valid: Sets the stipulated well, row or column invalid ('False') or valid ('True'), defaults to False.
        :type valid: bool
        :param wells: Wells to be invalidated passed as a string or list of strings.
        :type wells: str or list of str
        :param rows: Rows to be invalidated passed as a string or list of strings.
        :type rows: str or list of str
        :param columns: Columns to be invalidated passed as an integer or list of integers.
        :type columns: int or list of int
        """
        # execute the corresponding invalidate functon from the platemapping package
        if 'wells' in kwargs:
            pm.invalidate_wells(platemap=self.plate_map, wells=kwargs['wells'], valid=valid)
        
        if 'rows' in kwargs:
            rows = tuple(kwargs['rows']) # convert the rows to tuple because invalidate_rows cannot take in a list
            pm.invalidate_rows(platemap=self.plate_map, rows=rows, valid=valid)
        
        if 'columns' in kwargs:
            pm.invalidate_cols(platemap=self.plate_map, cols=kwargs['columns'], valid=valid)
        
        if len(kwargs) == 0:   # return error if neither of the keyword arguments is passed
            raise TypeError('No arguments were passed. Specify the wells, rows and/or columns to be invalidated!')
      
    
    def background_correct(self):
        """Calculates background corrected values for p and s channel in all repeats.
        
        The backgorund correction is done by subtracting the mean value of blank p (or s) channel intensity for a given 
        protein, tracer or competitor concentration from each non-blank value of the p (or s) channel intensity for that concentration. 
        """
        for key, value in self.data_dict.items(): 
            metadata, data = value.values()   
        
            # calculate p and s corrected data frame using _background_correct func and add it to data dictionary
            self.data_dict[key]['data']['p_corrected'] = FA._background_correct(data['p'], self.plate_map)
            self.data_dict[key]['data']['s_corrected'] = FA._background_correct(data['s'], self.plate_map)
            
        print('Background correction was successfully performed!')
            
    def _background_correct(data, platemap):
        """Calculates background corrected p or s channel values for protein/titration or competition experiment.
        
        :param data: Data frame with raw p or s channel values 
        :type data: pandas df
        :param platemap: Data frame with platemap
        :type platemap: pandas df
        :return: Data frame with background corrected values
        :rtype: pandas df
        """
        df = platemap.join(data)   # join p or s channel data to platemap
        df[df.columns[-1]] = df[df.columns[-1]][df['Valid'] == True]   # replace 'p' or 's' values with NaN if the well is invalidated
        col_name = df.columns[-1] + '_corrected'  
        no_index = df.reset_index()   # move the 'well id' index to df column
        columns = ['Type','Protein Name','Protein Concentration','Tracer Name','Tracer Concentration','Competitor Name','Competitor Concentration']
        
        # create a multindex df to which blank df will be joined
        mindex = pd.MultiIndex.from_frame(no_index[columns])   # create multiindex
        reindexed = no_index.set_index(mindex).drop(columns, axis=1)   # add multiindex to df and drop the columns from which multiindex was created
    
        mean = no_index.groupby(columns, dropna=False).mean().drop('Valid', axis=1).drop('empty', axis=0)   # calculate mean for each group of three wells and remove 'Valid' column
        mean.rename(columns={mean.columns[-1]: 'Mean'}, inplace=True)   # rename the last column to 'Mean' to avoid errors during joining
        blank = mean.xs('blank', level=0, drop_level=True)   # take a group with only blank wells
        
        reset_idx =  blank.reset_index()   # move multiindex to df
        nans = [col for col in reset_idx.columns if reset_idx[col].dropna().empty]    # list of all columns containing only 'nan' values
        d = reset_idx.drop(labels=nans, axis=1)   # delete all columns containing only 'nan' values
        blank2 = d.set_index(pd.MultiIndex.from_frame(d.loc[:,d.columns[:-1]])).drop(d.columns[:-1], axis=1)   # multi index to the remaining columns

        joined = reindexed.join(blank2, on=list(blank2.index.names))   # join the blank mean data on the indexes only from blank df
        joined[col_name] = joined[joined.columns[-2]] - joined[joined.columns[-1]]   # calculate background corrected values
        jindexed = joined.set_index('index', append=True).reset_index(level=[0,1,2,3,4,5,6]).rename_axis(None)   # set index to 'well id' and move multiindex to df columns
        
        return jindexed[[col_name]]   # extract and return df with corrected values

    
    def calc_r_i(self, correct=True, plot_i=True, thr=80):
        """Calculates anisotropy and fluorescence intensity for each well in all repeats using the raw and background corrected p and s channel data.
        
        The fluorescence intensity (I) and anisotropy (r) are calculated using the follwing formulas: I = s + (2*g*p) for intensity and
        r = (s - (g*p)) / I for anisotropy. Results are stored in the following data frames: i_raw and r_raw (calculated using the uncorrected 
        p and s channel values) and i_corrected and r_corrected (calculated using the background corrected p and s channel values).
        
        The function also calculates the percentage intesity of the non blank wells as comapred to the blank corrected wells using the formula:
        (raw intensity - corrected intensity) / raw intensity * 100%. If 'plot_i=True', the graph of percentage intenstiy against the 
        well ids for all repeats is displayed along with a summary of wells above the threshold (defaults to 80%).
        
        :param correct: Calculate the anisotropy and intensity using the background corrected values of p and s channel data, defaults to True.
        :type correct: bool
        :param plot_i: Display plots of the percentage intensity against well ids for all repeats, defaults to True.
        :type plot_i: bool
        :param thr: Percentage intensity above which the wells are included in the summary if plot_i=True, defaults to 80.
        :type thr: int
        """
        FA.th = thr   # assign the threshold value to the class variable so that it can be accessed by functions that are not class methods
    
        for key, value in self.data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            
            # calculate raw intensity and anisotropy using _calc_r_i function and add them to data dictionary
            i, r = FA._calc_r_i(data['p'], data['s'], self.g_factor, 'raw')
            self.data_dict[key]['data']['i_raw'] = i   
            self.data_dict[key]['data']['r_raw'] = r   
            
            if correct:   # calculate intensity and anisotropy using background corrected values of p and s
                
                if 'p_corrected' and 's_corrected' not in data:   # check if background subtraction has been performed
                    raise AttributeError('The corrected anisotropy and intensity can only be calculated after background correction of the raw p and s channel data.')
                
                i_c, r_c = FA._calc_r_i(data['p_corrected'], data['s_corrected'], self.g_factor, 'corrected')
                self.data_dict[key]['data']['i_corrected'] = i_c   
                self.data_dict[key]['data']['r_corrected'] = r_c    
                
                # calculate intensity percentage data and add it to data dict
                self.data_dict[key]['data']['i_percent'] = FA._calc_i_percent(i, i_c, self.plate_map)
        
        if plot_i:   # plot the percentage intensity against the well ids for all repeats
            FA._plot_i_percent(self.data_dict, self.plate_map)
        else:
            print('The fluorescence intensity and anisotropy were successfully calculated!\n')

    def _calc_r_i(p, s, g, col_suffix):
        """Calculates either anisotropy or intensity and labels the resulting dfs according to the col_suffix parameter
        
        :param p: Data frame with p channel data (can be either raw or background corrected)
        :type p: pandas df 
        :param s: Data frame with s channel data (can be either raw or background corrected)
        :type s: pandas df
        :param g: G-factor
        :type g: float
        :param col_suffix: Suffix to add to column name of the resulting intensity or anisotropy data frame, e.g. 'raw', 'corrected'
        :type col_suffix: str
        :return: Two data frames with calculated anisotropy and intensity values
        :rtype: tuple of pandas df"""
        
        p_rn = p.rename(columns={p.columns[0]: s.columns[0]})   # rename the col name in p data frame so that both p and s dfs have the same col names to enable calculation on dfs
        i = s + (2 * g * p_rn)       # calculate intensity
        r = (s - (g * p_rn)) / i     # and anisotropy
        i_rn = i.rename(columns={i.columns[0]: 'i_' + col_suffix})   # rename the col name using the column suffix argument
        r_rn = r.rename(columns={r.columns[0]: 'r_' + col_suffix})           
        return i_rn, r_rn  
    
    def _calc_i_percent(ir, ic, platemap):
        """Calculates the percentage intensity of blank wells compared to non-blank wells.
        
        :param ir: Data frame with corrected intensity 
        :type ir: pandas df
        :param ic: Data frame with raw intensity
        :type ic: pandas df
        :param platemap: Platemap
        :type platemap: pandas df
        :return: Data frame with percentage intensity data
        :rtype: pandas df"""
        
        ir_rn = ir.rename(columns={ir.columns[0]:ic.columns[0]})   # rename the col name in raw intensity df so that it's the same as in corrected intensity df
        percent = (ir_rn - ic) / ir_rn * 100   
        percent.rename(columns={'i_corrected': 'i_percent'}, inplace=True)   
        return percent
        
    def _plot_i_percent(data_d, platemap):
        """Plots the percentage intensity data against the well ids with a horizontal threshold bar and prints a summary of wells above the 
        threshold for all non-blank and non-empty cells in all repeats. A single figure with multiple subplots for each repeat is created.
        
        :param data_d: Data dictionary
        :type data_d: dict 
        :param platemap: Platemap needed to subset only the non-blank and non-empty cells
        :type platemap: pandas df"""
        
        summary = ''   # empty string to which lists of wells to be printed are appended after checking data from each repeat
        fig = plt.figure(figsize=(8*int((len(data_d) + 2 - abs(len(data_d) - 2))/2), 4*int( math.ceil((len(data_d))/2)) ), tight_layout=True)   # plot a figure with variable size depending on the number subplots (i.e. repeats)
        
        for key, value in data_d.items():   # iterate over all repeats
            metadata, data = value.values()
            df = platemap.join(data['i_percent'])
            df_per = df[(df['Type'] != 'blank') & (df['Type'] != 'empty')]   # subset only the non-blank and non-empty cells
            
            plt.subplot(int( math.ceil((len(data_d))/2) ), int( (len(data_d) + 2 - abs(len(data_d) - 2))/2 ), int(key[-1]))
            plt.bar(df_per.index, df_per['i_percent'])   # plot a bar plot with intensity percentage data 
            plt.axhline(FA.th, color='red')   # plot horizontal line representing the threshold on the bar plot
            ax = plt.gca()   # get the axis object
            ax.set_ylabel('')
            ax.set_xlabel('wells')
            ax.set_title(f'Repeat {key[-1]}')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter())   # set formatting of the y axis as percentage
            xlabels = [i if len(i) == 2 and i[1] == '1' else '' for i in list(df_per.index)]   # create a list of xtics and xticklabels consiting only of the first wells from a each row
            ax.set_xticks(xlabels)
            ax.set_xticklabels(xlabels)
        
            wells = list(df_per[df_per['i_percent'] > FA.th].index)   # get a list of well ids above the threshold for this repeat
            if wells != []:   # append wells above the threshold and the repective repeat number to the string with appropriate formatting
                summary = summary + f'\tRepeat {key[-1]}: {str(wells)}\n'
        
        plt.show()   # ensure the figure is displayed before printing the summary message

        if summary != '':   # display the summary of wells above the threshold
            print(f'In the following wells the percentage intensity value was above the {FA.th}% threshold:')
            print(summary)
        else:
            print(f'None of the wells has the percentage intensity value above the {FA.th}% threshold.')
            
    def plot_i_percent(self):
        """Disply the graph of percentage intesity of the non blank wells as comapred to the blank corrected wells against well ids for all repeats."""
        return FA._plot_i_percent(self.data_dict, self.plate_map)
    
    
    def calc_mean_r_i(self):
        """Calculates the mean anisotropy and intensity over the number of replicates for each specific protein, tracer 
        or competitor concentration along with standard deviation and standard error. 
        This data is required for fitting a logistic curve to anisotropy and intensity plots.
        """
        for key, value in self.data_dict.items():
            metadata, data = value.values()
            
            # create dictionaries 'r_mean'and 'i_mean' containing mean anisotropy and intensity data frames for each protein-tracer-competitor
            data['r_mean'] = FA._calc_mean_r_i(data['r_corrected'], self.plate_map)   
            data['i_mean'] = FA._calc_mean_r_i(data['i_corrected'], self.plate_map)   
            
            # create data frame for storing the fitting params and set lambda value to 1
            cols = ['rmin','rmin error', 'rmax', f'rmax error', 'r_EC50', 'r_EC50 error', 'r_hill', 'r_hill error', 'Ifree', 
                    'Ifree error', 'Ibound', 'Ibound error', 'I_EC50', 'I_EC50 error', 'I_hill', 'I_hill error', 'lambda']   
            data['fit_params'] = pd.DataFrame(index=self.final_fit.index, columns=cols)   
            data['fit_params']['lambda'] = 1
            
            if set(self.final_fit.index.get_level_values(2).unique()) != {'-'}:    # if it is a competition experiment create also data frme for storing the ic50 curve fitting params
                cols_comp = ['min','min error', 'max', 'max error', 'IC50', 'IC50 error', 'hill', 'hill error']      
                data['fit_params_com'] = pd.DataFrame(index=self.final_fit.index, columns=cols_comp)
        
        print('The mean anisotropy and intensity were successfully calculated.') 
       
    def _calc_mean_r_i(df, plate_map):
        """Calculates mean anisotropy for each protein (or tracer or competitor) concentration value, its standard deviation and standard error.
        
        :param df: Data frame with anisotropy or intensity values
        :type df: pandas df
        :param plate_map: Plate map data frame
        :type plate_map: pandas df
        :return: A dictionary of data frames for each unique protein-tracer-competitor
        :rtype: dict"""
        
        join = plate_map.join(df)   # join anisotropy or intensity df to platemap df
        subset = join[(join['Type'] != 'blank') & (join['Type'] != 'empty')]   # use only the non-blank and non-empty cells
        noidx = subset.reset_index()
        
        columns = ['Protein Name','Protein Concentration','Tracer Name','Tracer Concentration','Competitor Name','Competitor Concentration']
        group = noidx.groupby(columns, dropna=False)
        mean = group.mean()   
        std = group.std()     
        sem = group.sem()    
        meanr = mean.rename(columns={mean.columns[-1]: 'mean'}).drop('Valid', axis=1)    # rename the mean column and remove the 'Valid' column
        stdr = std.rename(columns={std.columns[-1]: 'std'}).drop('Valid', axis=1)   # rename the std column and remove the 'Valid' column
        semr = sem.rename(columns={sem.columns[-1]: 'sem'}).drop('Valid', axis=1)   # rename the sem column and remove the 'Valid' column
        merge = pd.concat([meanr, stdr, semr], axis=1)
        tosplit = merge.reset_index().fillna({'Competitor Name': '-'})   # remove multiindex and in case of protein/tracer titration set competitor name as '-'
        split = dict(tuple(tosplit.groupby(['Protein Name', 'Tracer Name', 'Competitor Name'], dropna=False)))   # split df based on multiindex so that a new df is created for each unique combination of protein, tracer and competitor
        
        return split
        
            
    def calc_lambda(self, approve=True):
        """Calculates lambda value for each protein-tracer pair for all repeats and, if approve=True, displays them so that
        a single value can be saved for each protein-tracer pair which will be used in subsequent calculations. 

        :param approve: Display lambda, rmin and rmax values for each protein-tracer pair and for all repeats, defaults to True.
        :type approve: bool
        """
        w_info = []   # list of tuples with info (rep no, lambda value, etc) needed to generate the widgets
        
        for key, value in self.data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            df = data['fit_params'].copy()    # create a copy of the fitting params df
            df['lambda'] = df['Ibound'] / df['Ifree']   # calculate the lambda value in a copied data frame
                
            if approve == False:
                self.data_dict[key]['data']['fit_params']['lambda'] = df['lambda']   # add the lambda values to fitting params df
                print('The lambda values were calculated and saved.')
            else:
                for ptc in list(df.index):   # iterate over each protein-tracer pair and create tuples with info needed for generation of widgets
                    rating = 100   # place for the rating function
                    info = (key, ptc, rating, df.loc[ptc, "lambda"],  data['fit_params'].loc[ptc, "rmin"],  data['fit_params'].loc[ptc, "rmax"])   # tuples conataining repeat no., calculated lambda, and protein-tracer names
                    w_info.append(info)

        if approve == True:   # execute the function for displying and handling the widgets
            return FA._widget(self.data_dict, w_info, self.final_fit, df)
            
    def _widget(data_dict, w_info, final_fit, df):
        """Function for generating and displaying the widgets with lambda values.
        It generates widgets for each tuple in the w_info list.
        
        :param data_dict: Data dictionary
        :type data_dict: dict
        :param w_info: A list of tuples containg information needed for the generation of widgets
        :type w_info: list
        :param final_fit: Data frame with final fitting parameters
        :type final_fit: pandas df
        :param df: Data frame with calculated lambda values
        :type df: pandas df
        """
        w_info.sort(key=lambda x: x[1])   # sort the tuples by the protein name so that the widgets are displayed by protein-tracer name
        reps = [wg.HTML(f"Repeat {i[0][-1]}") for i in w_info]   # list of text widgets with repeat numbres
        proteins = [wg.HTML(f"{i[1][0]}") for i in w_info]   # list of text widgets with protein names
        tracers = [wg.HTML(f"{i[1][1]}") for i in w_info]   # list of text widgets with tracer names
        #scores = [wg.HTML(f"Score: {i[2]}") for i in w_info]   
        lambdas = [wg.Checkbox(value=False, description="$\lambda$ = %.4f" % (i[3])) for i in w_info]   # list of checkbox widgets with lambda values
        rminmax = [wg.Checkbox(value=False, description="rmin = %.5f, rmax = %.5f" % (i[4], i[5])) for i in w_info]   # list of checkbox widgets with rmin and rmax values
            
        v_lambdas = wg.VBox(lambdas)   # group all lambda checkbox widgets into a vertical list layout
        v_proteins = wg.VBox(proteins)   # group all protein name widgets into a vertical list layout
        v_tracers = wg.VBox(tracers)   # group all tracer name widgets into a vertical list layout
        v_reps = wg.VBox(reps)   # group all repeat number widgets into a vertical list layout
        #v_scores = wg.VBox(scores)
        v_rminmax = wg.VBox(rminmax)   # group all rmin and rmax checkbox widgets into a vertical list layout
            
        hbox = wg.HBox([v_proteins, v_tracers, v_reps, v_lambdas, v_rminmax])   # arrange the six vertical boxes into one widget box'
        button = wg.Button(description='Save')   # create a button for saving the selected values
        print("""Choose the lambda values that will be saved for each protein-tracer pair. \nIf you choose more than one lambda value for a given protein-tracer pair, only the first choice will be saved.\nIf you do not choose any lambda value for a given protein-tracer pair the default value of 1 will remain but you still need to select the rmin and rmax for this pair.""")
        display(hbox, button)   # display the box with widgets and the button
            
        def btn_eventhandler(obj): 
            """Function that is executed when the 'Save' button is clicked. It checks which checkboxes were ticked and 
            updates the final fit df with the calcualted lambda values and/or rmin and rmax values. 
            Only the first value of lambda for a given protein-tracer will be saved.
            """
            added_lambda = []   # protein-tracer pairs for which lambda values were added
            added_rminmax = []   # protein-tracer pairs for which rmin and rmax values were added
            
            for i in range(0, len(lambdas)):   # iterate over each checkbox widget
                index = (proteins[i].value, tracers[i].value, '-')   # get the tuple with protein-tracer names
                cols = ['rmin','rmin error','rmax','rmax error']
                
                if lambdas[i].value == True:   # if the lambda checkbox was ticked, the widget's 'value' attribute is True 
                    if index not in added_lambda:   # if lambda for this protein-tracer pair has not yet been added 
                        final_fit.loc[index, "lambda"] = df.loc[index, "lambda"]   # add the calculated lambda to the final_fit df
                        final_fit.loc[index, cols] = data_dict[f'repeat_{reps[i].value[-1]}']['data']['fit_params'].loc[index, cols]   #add rmin, rmax and their errors to the final_fit df
                        added_lambda.append(index)  
                        
                if rminmax[i].value == True:
                    if index not in added_lambda and index not in added_rminmax:   # if neither lambda nor rmin/rmax for this protein-tracer pair have been added 
                        final_fit.loc[index, cols] = data_dict[f'repeat_{reps[i].value[-1]}']['data']['fit_params'].loc[index, cols]
                        added_rminmax.append(index)
            
            print('Selected values were saved.')
        
        button.on_click(btn_eventhandler)   #link the button event handler function with actual clicking of the button using 'on_click' function
        
    
    def calc_amount_bound(self):
        """Calculates the amount of fluorescent tracer bound to the protein using the following formula:
        L_B =( ( (λ*(rmin-rmax⁡)) / (r-rmin ) +1) )^(-1) * L_T
        
        The amount bound is calculated as a mean for all replicates for each protein, tracer or competitor concentration
        along with its standard deviation and standard error.
        """
        ptc_list = list(self.final_fit[self.final_fit['rmin'].isna()].index)   # list of indexes for which rmin and rmax are not defined
        if ptc_list != []: 
            raise DataError(f"The 'rmin' and 'rmax' values are not defined for the following proteins and tracers: {ptc_list}.\nUse 'calc_lambda' function or 'set_fitparams' to choose 'rmin' and 'rmax' values.")
                            
        for key, value in self.data_dict.items():
            metadata, data = value.values()
            data['amount_bound'] = FA._calc_amount_bound(data['r_corrected'], self.plate_map, self.final_fit)   # create dictionary 'r_mean' with mean anisotropy data frames for each protein-tracer pair
        
        print('The amount of fluorescent tracer bound was successfully calculated.')
            
    def _calc_amount_bound(df, platemap, final_fit):
        """Calculates the amount from anisotropy data.
        
        :param df: Data frame with anisotropy values
        :type df: pandas df
        :param platemap: Plate map data frame
        :type platemap: pandas df
        :return: A dictionary of data frames for each unique protein-tracer-competitor
        :rtype: dict
        """
        join_pm = platemap.join(df)   # join corrected anisotropy df with the platemap df
        subset = join_pm[(join_pm['Type'] != 'blank') & (join_pm['Type'] != 'empty')].fillna({'Competitor Name': '-'})   # take only non-blank and non-empty wells, in case of protein/tracer titration set competitor name as '-'

        re_idx = subset.set_index(pd.MultiIndex.from_frame(subset[['Protein Name','Tracer Name','Competitor Name']])).rename_axis([None,None,None])   # replace the index with multiindex (protein-tracer-competitor) and remove its names
        join_ff = re_idx.join(final_fit[['rmin','rmax','lambda']])   # join the final_fit df to the anisotropy df on multiindex

        # calcualte the amount bound (all parameters needed are already in the data frame)
        join_ff['mean'] = (((((join_ff["lambda"] * (join_ff['rmax']-join_ff['r_corrected'])) / (join_ff['r_corrected'] - join_ff['rmin']))) +1) **(-1)) * join_ff['Tracer Concentration']   
        # remove the redundant columns and set dtype of 'amount' column as float to avoid pandas DataError
        drop = join_ff.drop(['r_corrected','Valid', 'rmin', 'rmax', "lambda"], axis=1).astype({'mean': 'float64'}) 
        
        columns = ['Protein Name','Protein Concentration','Tracer Name','Tracer Concentration','Competitor Name','Competitor Concentration']
        group = drop.groupby(columns, dropna=False)
        mean = group.mean()   
        std = group.std()     
        sem = group.sem()   
        stdr = std.rename(columns={std.columns[-1]: 'std'})  # rename column to 'std' 
        semr = sem.rename(columns={sem.columns[-1]: 'sem'})  # rename column to 'sem'
        merge = pd.concat([mean, stdr, semr], axis=1)   # merge the amount, std and sem data frames into one df
        tosplit = merge.reset_index().fillna({'Competitor Name': '-'})   # remove multiindex, in case of protein/tracer titration set competitor name as '-'
        split = dict(tuple(tosplit.groupby(['Protein Name', 'Tracer Name', 'Competitor Name'], dropna=False)))   # dictionary a data frame for each protein-tracer pair
        
        return split
    
    def _calc_Ki(ptc, params_df, platemap, final_fit):
        """Calculates Ki, Ki* (based on the actual protein concentration determined from the ic50 plot) and their errors.
        
        :param ptc_pair: Tuple of 3 strings: protein, tracer and competitor name for which the Ki is calculated
        :type ptc: tuple
        :param params_df: Data frame with ic50 fitting parameters
        :type params_df: pandas df
        :param platemap: Platemap data frame
        :type platemap: pandas df
        :return: Values of Ki, Ki*, Pt and their errors
        :rtype: tuple
        """
        IC50, IC50_err, pmax = tuple(params_df.loc[ptc, ['IC50','IC50 error','max']])    # get IC50 and the upper asymptote of IC50 plot   
        Kd, Kd_err = tuple(final_fit.loc[ptc, ['Kd','Kd error']])     # get Kd and its error
        LT = float(platemap['Tracer Concentration'].dropna().unique())
        PT = float(platemap['Protein Concentration'].dropna().unique())
        PT_2 = ( (Kd*pmax) / (LT-pmax) ) + pmax    # protein conc calculated based on upper asymptote of IC50 plot
        Kd_arr = np.random.normal(Kd, Kd_err, 100000)     # simulated Kd values based on the real Kd and its error
        IC50_arr = np.random.normal(IC50, IC50_err, 100000)    # simulated IC50 values based on the real IC50 and its error
    
        def _calc_Ki_val(Kd, LT, PT, IC50):
            """Calculates Ki value"""
            P0 = ( -(Kd+LT-PT) + np.sqrt( ((Kd+LT-PT)**2) - (4*PT*LT) ) ) / 2
            L50 = LT - ( (PT-P0) / 2 )
            I50 = IC50 - PT + ( (PT-P0) / 2 ) * (1 + (Kd/L50) )
            Ki = I50 / ( (L50/Kd) + (P0/Kd) + 1 )
            return Ki
        
        Ki = _calc_Ki_val(Kd, LT, PT, IC50)   # calculate Ki
        Ki_err = np.std(_calc_Ki_val(Kd_arr, LT, PT, IC50_arr))    # calculate Ki error as std of Ki values generated around the real Ki
        Ki_2 = _calc_Ki_val(Kd, LT, PT_2, IC50)    # calculate Ki*
        Ki_2_err = np.std(_calc_Ki_val(Kd_arr, LT, PT_2, IC50_arr))    # calculate Ki* error as std of Ki* values generated around Ki*
        
        return Ki, Ki_err, Ki_2, Ki_2_err, PT, PT_2
    
    
    ##### Curve fitting functions #####                
    def _EC50(pc, rmin, rmax, EC50, hill):
        """Function for fitting a curve to the plot of anisotropy (or intensity) against protein/tracer 
        concentration, where pc is protein (or tracer) concentration, rmin is the lower asymptote, rmax is the upper asymptote, 
        EC50 is midpoint of transition (pc at point of inflection), hill is the slope
        """
        return (rmin - rmax) / (1 + (pc/EC50)**hill) + rmax
    
    def _EC50_com(pc, rmin, rmax, EC50, hill):  
        """Function for fitting a curve to the plot of anisotropy against competitor concentration, for fitting a curve 
        to the plot of amount of fluorescent tracer bound to the target protein against competitor concentration
        where pc is competitor concentration, rmin is the lower asymptote, rmax is the upper asymptote, 
        EC50 is midpoint of transition (pc at point of inflection), hill is the slope
        """
        return (rmax - rmin) / (1 + (pc/EC50)**hill) + rmin
    
    def _LB(LT, PT, Kd):
        """Function for fitting a curve to the plot of concentration of fluorescent tracer bound to the target protein 
        against protein or tracer concentration.
        """
        return ( (LT+PT+Kd) - np.sqrt( ( ((LT+PT+Kd)**2) - (4*LT*PT) ) ) ) / 2 
    
    def _init_params(df, t_type):
        """Estimates initial parameters for the _EC50 function that are passed to the curve fitting function
        
        :param df: Data frame containing mean values of anisotropy or intensity
        :type df: pandas df
        :param t_type: Type of titration ('Protein' or 'Tracer')
        :type t_type: str
        :return: List with estiamted parameters of min, max and EC50, hill is assumed to be 1
        :rtype: list
        """
        rmin = df['mean'].min()
        rmax = df['mean'].max()
        mid = (rmax + rmin) / 2
        mid_idx = df['mean'].sub(mid).abs().argmin()
        EC50 = df.iloc[mid_idx][f'{t_type} Concentration']
        init_params = [rmin, rmax, EC50, 1]
        return init_params
    
    def _curve_fit(func, df, t_type, var, **kwargs):
        """Fits a curve to the plot of specified variable against protein (or tracer) concentration using pre-defined funcion.
        
        :param func: Funcion describing the curve to be fitted to data points
        :type func: func
        :param df: Data frame containing mean values of anisotropy, intensity or amount bound and their errors (std and sem).
        :type df: pandas df
        :param t_type: Type of titration ('Protein', 'Tracer', or 'Competitor'), determines order of parmeters in returned list
        :type t_type: str
        :param **kwargs: Keyword arguments that can be passed into the scipy curve_fit function
        :param var: Type of fitting perormetd, either logisitic ('log') or single site ('ssb').
        :type var: str
        :return: A list of fitting parameters along with their error in proper order so that it can be added to the fitting params data frame
        :rtype: list
        """
        drop = df[df[f'{t_type} Concentration'] != 0].dropna(subset=['mean'])   # exclude the protein concentration = 0 point and any NaN mean values from data fitting
        
        if 'sigma' in kwargs:  
            sigma = drop[kwargs.pop('sigma')]   # take the column with std or sem error data as sigma
        else:
            sigma = None
            
        if 'p0' not in kwargs and var == 'log':   # user did not pass their initial guess
            p0 = FA._init_params(drop, t_type)   # use _init_params function to get the initial guess
        elif 'p0' in kwargs:
            p0 = kwargs.pop('p0')   # remove p0 from kwargs and assign to p0 argument so that there is only one p0 arg passed to curve fit
        else:  
            p0 = None
                              
        popt, pcov = curve_fit(func, drop[f'{t_type} Concentration'], drop['mean'], p0=p0, sigma=sigma, **kwargs)
        perr = np.sqrt(np.diag(pcov))   # calculate the error of the fitting params
        
        if var == 'log':
            all_params = np.insert(popt, obj=[1,2,3,4], values=perr)   # insert the errors after the respective fitting parameter value
        if var == 'ssb':
            all_params = np.insert(popt[::-1], obj=[1,2], values=perr[::-1])   # rearrange the order of parametrs in the array
        
        return list(all_params) 
    
    def logistic_fit(self, prot=['all'], trac=['all'], rep=['all'], var='both', **kwargs):
        """Fits a logistic curve to the plot of anisotropy (or intensity) against protein or tracer concentration.
        Returns the fitting parameters with associated errors for each repeat that are stored in the fitting_params data frame.
        
        :param prot: List of protein names for which fitting is performed, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which fitting is performed, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which fitting is performed, defaults to ['all'].
        :type rep: list of ints
        :param var: A variable for which fitting is performed, (either 'r' for anisotropy or 'i' for inteensity), defaults to 'both'.
        :type var: str
        :param **kwargs: Keyword arguments that can be passed to the SciPy curve_fit function.
        """
        # get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, ['all'], rep)
        errors = []   # list for storing the details of errors due to failed fitting
        
        if len(self.plate_map['Tracer Concentration'].dropna().unique()) == 1:   # protein is titrated to a constant amount of tracer
            t_type = 'Protein'
        if len(self.plate_map['Protein Concentration'].dropna().unique()) == 1:   # tracer is titrated to a constant amount of protein
            t_type = 'Tracer'
        
        for rep, value in data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            
            for ptc in ptc_list:   # iterate over all protein-tracer pairs
                
                if var == 'r' or var == 'both':
                    try:   # try fitting the curve to anisotropy data 
                        cols = ['rmin','rmin error','rmax', 'rmax error', 'r_EC50', 'r_EC50 error', 'r_hill', 'r_hill error']
                        r_mean = data['r_mean'][ptc]   # extract the df with mean anisotropy for a given protein-tracer pair
                        params_r = FA._curve_fit(FA._EC50, r_mean, t_type, 'log', **kwargs)   # fit the data to logistic curve using the initial parameteers
                        data['fit_params'].loc[ptc, cols] = params_r   # add the fitting parameters to the respective df

                    except RuntimeError as e:   # if fitting fails, added details about the error to the errors list and proceed intensity data fitting
                        r_error_info = (rep, 'r', ptc, e)
                        errors.append(r_error_info)
                
                if var == 'i' or var == 'both':
                    try:   # try fitting the curve to intensity data
                        cols = ['Ifree', 'Ifree error', 'Ibound','Ibound error', 'I_EC50', 'I_EC50 error', 'I_hill', 'I_hill error']
                        i_mean = data['i_mean'][ptc]   # extract the df with i mean for a given protein-tracer pair
                        params_i = FA._curve_fit(FA._EC50, i_mean, t_type, 'log', **kwargs)                           
                        data['fit_params'].loc[ptc, cols] = params_i

                    except RuntimeError as e:   # if fitting fails, added details about the error to the errors list and proceed to to the next protein-tracer pair
                        i_error_info = (rep, 'i', ptc, e)
                        errors.append(i_error_info)

        if errors != []:   # raise a warning if fitting failed for any protein-tracer pair
            warnings.warn(f"The curve fitting failed in the following cases:\n\n{errors}\n\nTry passing additional keyword arguments to the fitting function.", RuntimeWarning)
        else:
            print('The logistic curve fitting was successfully performed.')
    
    def logisitc_fit_com(self, prot=['all'], trac=['all'], rep=['all'], com=['all'], **kwargs):
        """Fits a logistic curve to the plot of anisotropy against competitor concentration. Returns the fitting 
        parameters with associated errors for each repeat that are stored in the fitting_params data frame.
        
        :param prot: List of protein names for which fitting is performed, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which fitting is performed, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which fitting is performed, defaults to ['all'].
        :type rep: list of ints
        :param com: List of competitor names for which fitting is performed, defaults to ['all'].
        :type com: list or list of str
        :param var: A variable for which fitting is performed, (either 'r' for anisotropy or 'i' for inteensity), defaults to 'both'.
        :type var: str
        :param **kwargs: Keyword arguments that can be passed to the SciPy curve_fit function.
        """
        # get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, com, rep)
        errors = []   # list for storing the details of errors due to failed fitting
        
        for rep, value in data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            
            for ptc in ptc_list:   # iterate over all protein-tracer pairs
                
                try:   # try fitting the curve to anisotropy data 
                    cols = ['rmin','rmin error','rmax', 'rmax error','r_IC50', 'r_IC50 error', 'r_hill', 'r_hill error']
                    r_mean = data['r_mean'][ptc]   # extract the df with mean anisotropy for a given protein-tracer pair
                    params_r = FA._curve_fit(FA._EC50_com, r_mean, 'Competitor', 'log', **kwargs)
                    data['fit_params'].rename(columns={'r_EC50':'r_IC50', 'r_EC50 error':'r_IC50 error'})
                    data['fit_params'].loc[ptc, cols] = params_r   # add the fitting parameters to the respective df

                except RuntimeError as e:   # if fitting fails, added details about the error to the errors list and proceed intensity data fitting
                    r_error_info = (rep, 'r', ptc, e)
                    errors.append(r_error_info)

        if errors != []:   # raise a warning if fitting failed for any protein-tracer pair
            warnings.warn(f"The curve fitting failed in the following cases:\n\n{errors}\n\nTry passing additional keyword arguments to the fitting function.", RuntimeWarning)
        else:
            print('The logistic curve fitting was successfully performed.')

    def single_site_fit(self, prot=['all'], trac=['all'], rep=['all'], **kwargs):
        """Fits a curve to the plot of concentration of fluorescent tracer bound to the target protein against the 
        protein (or tracer) concentration. The resulting fitting parameters are stored in the final_fit data frame.
        
        :param prot: List of protein names for which fitting is performed, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which fitting is performed, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which fitting is performed, defaults to ['all'].
        :type rep: list of ints
        :param **kwargs: Keyword arguments that can be passed to the SciPy curve_fit function.
        """
        # get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, ['all'], rep)
        errors = []   # list for storing the details of errors due to failed fitting
        
        if len(self.plate_map['Tracer Concentration'].dropna().unique()) == 1:   # protein is titrated to a constant amount of tracer
            t_type = 'Protein'
        if len(self.plate_map['Protein Concentration'].dropna().unique()) == 1:   # tracer is titrated to a constant amount of protein
            t_type = 'Tracer'
        
        for rep, value in data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            
            for ptc in ptc_list:   # iterate over all protein-tracer pairs
                try:   # try fitting the curve to anisotropy data 
                    amount_b = data['amount_bound'][ptc]   # extract the df with mean amount bound for a given protein-tracer pair
                    params = FA._curve_fit(FA._LB, amount_b, t_type, 'ssb', **kwargs)
                    
                    if t_type == 'Protein':
                        self.final_fit.loc[ptc, ['Kd', 'Kd error', 'LT', 'LT error']] = params  
                    if t_type == 'Tarcer':
                        self.final_fit.loc[ptc, ['Kd', 'Kd error', 'PT', 'PT error']] = params
                    
                except RuntimeError as e:  
                    error_info = (rep, ptc, e)
                    errors.append(error_info)
                
        if errors != []:   # raise a warning if fitting failed for any protein-tracer pair
            warnings.warn(f"The curve fitting failed in the following cases:\n\n{errors}\n\nTry passing additional keyword arguments to the fitting function", RuntimeWarning)
        else:
            print('The single site curve fitting was successfully performed.')
    
    def single_site_fit_com(self, prot=['all'], trac=['all'], com=['all'], rep=['all'], **kwargs):
        """Fits a curve to the plot of concentration of fluorescent tracer bound to the target protein against the 
        competitor concentration. The rsulting fitting parameters are stired in the fitting_params_com data frame.
        
        :param prot: List of protein names for which fitting is performed, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which fitting is performed, defaults to ['all'].
        :type trac: list of str
        :param com: List of competitor names for which fitting is performed, defaults to ['all'].
        :type com: list or list of str
        :param rep: List of repeat numbers for which fitting is performed, defaults to ['all'].
        :type rep: list of ints
        :param **kwargs: Keyword arguments that can be passed to the SciPy curve_fit function.
        """
        # get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, com, rep)
        errors = []   # list for storing the details of errors due to failed fitting
        
        for rep, value in data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            
            for ptc in ptc_list:   # iterate over all protein-tracer pairs
                try:   # try fitting the curve to anisotropy data 
                    amount_b = data['amount_bound'][ptc]   # extract the df with mean amount bound for a given protein-tracer pair
                    params = FA._curve_fit(FA._EC50_com, amount_b, 'Competitor', 'log', **kwargs)
                    data['fit_params_com'].loc[ptc, :] = params

                except RuntimeError as e:  
                    error_info = (rep, ptc, e)
                    errors.append(error_info)
                
        if errors != []:   # raise a warning if fitting failed for any protein-tracer pair
            warnings.warn(f"The curve fitting failed in the following cases:\n\n{errors}\n\nTry passing additional keyword arguments to the fitting function", RuntimeWarning)
        else:
            print('The single site curve fitting was successfully performed.')
    
    ##### Anisotropy and biniding constant plotting functions #####
    def _get_items_to_plot(data_d, platemap, prot, trac, com, rep):
        """Creates a list of tuples with protein-tracer-competitor names based on the 'prot', 'trac' and 'com' 
        parameters and a subset of data_dict based on the 'rep' parameter.
        """
        if prot[0] == 'all' and trac[0] == 'all' and com[0] == 'all':   # all proteins and all tracers
            ptc_list = list(data_d['repeat_1']['data']['r_mean'].keys())   # 'r_mean' dict contains all protein-tracer names as dict keys
        else:
            if com[0] == 'all':
                com = list(platemap['Competitor Name'].dropna().unique())
                if com == []:
                    com = ['-']
            if trac[0] == 'all':
                trac = list(platemap['Tracer Name'].dropna().unique())
            if prot[0] == 'all':
                prot = list(platemap['Protein Name'].dropna().unique())
            
            ptc_list = [item for item in product(prot, trac, com)]
        
        # define a data dictionary to iterate through based on the 'rep' parameter:
        if rep[0] == 'all':   # for all repeats use the whole data_dict
            data_dict = data_d
        else:   # for specific repeats use the subset of data_dict containg only the repeats specified in 'rep' parameter
            data_dict = {key: value for key, value in data_d.items() if int(key[-1]) in rep}
        
        return data_dict, ptc_list
    
    def _vir_data(df, t_type, samples):
        """Returns a set of data points (x-axis data) evenly spaced on a logarythmic scale to be used for plotting 
        the curves instead of the real concentration data to make the curve appear smoother.
        
        :param df: Data frame containing conentration data
        :type df: pandas df
        :param t_type: Type of concentration to be used: Protein, Tracer, Competitor.
        :type t_type: str
        :param samples: Number of data points to generate
        :type samples: int
        :return: Array of concentration values evenly spaced between minimal and maximal concentration
        :rtype: numpy array
        """
        minc = df[f'{t_type} Concentration'].min()
        maxc = df[f'{t_type} Concentration'].max()
        return np.logspace(np.log10(minc),np.log10(maxc), samples)
    
    def _plot_ani(data_df, params_df, ptc, t_type, fig, axs, err, var, rep, unit, exp, disp, leg, dpi):
        """General function for plotting the anisotropy and intensity and saving the figures. Returns a single figure.
        
        :param data_df: Data frame with mean values of anisotropy or intensity and their associated errors
        :type data_df: pandas df
        :params_df: Data frame with fitting parameters
        :type params_df: pandas df
        :param ptc: protein-tracer-competitor for which the graph is to be generated
        :type ptc: tuple
        :param t_type: Type of titration ('Protein' or 'Tracer')
        :type t_type: str
        :param fig: Figure on which the data is plotted, needed for saving the figure as png file
        :type fig: matplotlib Figure
        :param axs: Indexed axis object on which the data is to be plotted, (e.g. axs[0, 1])
        :type axs: matplotlib AxesSubplot
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem')
        :type err: str
        :param var: Variable for which the plot is to be generated ('r' or 'i')
        :type var: str
        :param rep: Repeat number for labelling of the graph
        :type rep: 'str'
        :param unit: Concentration units to be displayed on the plots
        :type unit: str
        :param exp: Determines whether the figure will be saved, can be either bool or string with directory path
        :type exp: bool or 'str'
        :param disp: Determines whether the figure will be displayed after plotting, default True
        :type disp: bool
        :param leg: Determines whether the legend and box with fitting parameters will be displayed on the figure, default True
        :type leg: bool
        :param dpi: Resolution of the figure in points per inch
        :type dpi: int
        """
        if var == 'r':   # define the parameters, legend text and legend coordinates characteristic for anisotropy data
            params = tuple(params_df.loc[ptc, ['rmin','rmax','r_EC50','r_hill']])   # fit params for curve plotting
            text = "$r_{min}$ = %.4f \u00B1 %.4f\n$r_{max}$ = %.4f \u00B1 %.4f\n$hill$ = %.2f \u00B1 %.2f\n" % tuple(params_df.loc[ptc, ['rmin',
            'rmin error','rmax','rmax error','r_hill', 'r_hill error']])
            EC50, EC50e = tuple(params_df.loc[ptc, ['r_EC50','r_EC50 error']])
            text_final = text + '$EC_{50}$ = ' + f'{EC50:,.2f} \u00B1 {EC50e:,.2f}'
            ylabel = 'Anisotropy'
            
        if var == 'i':   # define the parameters, legend text and legend coordinates characteristic for intensity data
            params = tuple(params_df.loc[ptc, ['Ifree','Ibound','I_EC50','I_hill']])   # fit params for curve plotting
            If, Ife, Ib, Ibe, EC50, EC50e = tuple(params_df.loc[ptc, ['Ifree','Ifree error', 'Ibound', 'Ibound error','r_EC50','r_EC50 error']])
            text = "$hill$ = %.2f \u00B1 %.2f\n" % tuple(params_df.loc[ptc, ['I_hill', 'I_hill error']])
            text_final = '$I_{free}$ = ' + f'{If:,.1f} \u00B1 {Ife:,.1f}\n' + '$I_{bound}$ = ' + f'{Ib:,.1f} \u00B1 {Ibe:,.1f}\n' + text + '$EC_{50}$ = ' + f'{EC50:,.2f} \u00B1 {EC50e:,.2f}'
            ylabel = 'Intensity'
        
        drop = data_df[data_df[f'{t_type[0]} Concentration'] != 0].dropna(subset=['mean'])   # exclude the protein concentration = 0 point and any NaNs from plotting
        axs.errorbar(drop[f'{t_type[0]} Concentration'], drop['mean'], yerr=drop[err], color='black', fmt='o', capsize=3, marker='s')
        axs.set_xscale('log')
        axs.set_ylabel(ylabel)
        axs.set_xlabel(f'[{ptc[int(t_type[1])]}] ({unit})')
        vir_data = FA._vir_data(drop, t_type[0], 200)    # x-axis data for curve plotting
        axs.plot(vir_data, FA._EC50(vir_data, *params), color='blue')
        
        if leg == True:   # display title and legend with fitting parameters
            axs.set_title(f'Protein: {ptc[0]}, Tracer: {ptc[1]}')
            axs.legend([f'logistic fitted curve\n{text_final}'], frameon=False, fontsize=11)
            
        if exp == True:   # save figures in the same directory as the notebook
            fig.savefig(f"rep_{rep[-1]}_{var}_{str(ptc[0])}_{str(ptc[1])}.png", dpi=dpi)
        if type(exp) == str:   # save figures in the user defined directory
            fig.savefig(f"{exp}rep_{rep[-1]}_{var}_{str(ptc[0])}_{str(ptc[1])}.png", dpi=dpi)
        
        if disp == False:   # if function is called by save_ani_figs then the plotted figures are not displayed
            plt.close(fig)
                
   
    def plot_ani(self, prot=['all'], trac=['all'], rep=['all'], err='std', legend=True):   
        """Plots anisotropy and intensity against protein or tracer concentration with a fitted logistic curve 
        for specified repeats and protein-tracer pairs. A separate figure for each repeat is created with anisotropy 
        and intensity graphs for all specified proteins and tracers arranged in two columns. 
        
        :param prot: List of protein names for which the graphs are created, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which the graphs are created, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which the graphs are created, defaults to ['all'].
        :type rep: list of int
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem'), defaults to 'std'.
        :type err: str
        """
        # get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, ['all'], rep)
        unit = str(self.plate_map['Concentration Units'].dropna().unique()[0])   # concentration units
        
        if len(self.plate_map['Tracer Concentration'].dropna().unique()) == 1:   # protein is titrated to a constant amount of tracer
            t_type = ('Protein', 0)
        if len(self.plate_map['Protein Concentration'].dropna().unique()) == 1:   # tracer is titrated to a constant amount of protein
            t_type = ('Tracer', 1)
        
        for key, value in data_dict.items():   # iterte over all repeats and create a sperate figure for each repeat
            metadata, data = value.values()
            fig, axs = plt.subplots(len(ptc_list), 2, figsize=(2*6.4, len(ptc_list)*4.8), tight_layout=True)   # grid for subplots has two columns and a variable number of rows, figsize automatically scales up
            fit_params = data['fit_params']
            
            if len(data_dict) > 1:   # do not display info about repeat number if there is only one repeat
                fig.suptitle(f"Repeat {key[-1]}", fontsize=16)
                
            for idx, ptc in enumerate(ptc_list):   # for each portein-tracer pair plot two graphs: anisotropy and intensity
                r_data_df, i_data_df = data['r_mean'][ptc], data['i_mean'][ptc]   # extract the df with anisotropy and intensity
        
                if len(ptc_list) == 1:   # for only one protein-tracer pair the subplot grid is 1-dimensional
                    FA._plot_ani(r_data_df, fit_params, ptc, t_type, fig, axs[0], err, 'r', key, unit, exp=False, disp=True, leg=legend, dpi=250)
                    FA._plot_ani(i_data_df, fit_params, ptc, t_type, fig, axs[1], err, 'i', key, unit, exp=False, disp=True, leg=legend, dpi=250)
                
                else:   # for more than one protein-tracer pair the subplot grid is 2-dimensional
                    FA._plot_ani(r_data_df, fit_params, ptc, t_type, fig, axs[idx,0], err, 'r', key, unit, exp=False, disp=True, leg=legend, dpi=250)
                    FA._plot_ani(i_data_df, fit_params, ptc, t_type, fig, axs[idx,1], err, 'i', key, unit, exp=False, disp=True, leg=legend, dpi=250)
    
    
    def _plot_ani_com(data_df, params_df, ptc, err, rep, unit, exp, leg, dpi):
        """General function for plotting the anisotropy figures for competition experiments. Returns a single figure
        
        :param data_df: Data frame with mean values of anisotropy or intensity and their associated errors
        :type data_df: pandas df
        :params_df: Data frame with fitting parameters
        :type params_df: pandas df
        :param ptc: protein-tracer pair for which the graph is to be generated
        :type ptc: tuple
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem')
        :type err: str
        :param rep: Repeat number for labelling of the graph
        :type rep: 'str'
        :param unit: Concentration units to be displayed on the plots
        :type unit: str
        :param exp: Determines whether the figure will be saved, can be either bool or string with directory path
        :type exp: bool or 'str'
        :param leg: Determines whether the legend and box with fitting parameters will be displayed on the figure, default True
        :type leg: bool
        :param dpi: Resolution of the figure in points per inch
        :type dpi: int
        """
        # define the parameters, legend text and legend coordinates characteristic for anisotropy data
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8), tight_layout=True)
        
        params = tuple(params_df.loc[ptc, ['rmin', 'rmax','r_IC50','r_hill']])    # fit params for curve plotting
        text = "$r_{min}$ = %.4f \u00B1 %.4f\n$r_{max}$ = %.4f \u00B1 %.4f\n$hill$ = %.2f \u00B1 %.2f\n" % tuple(params_df.loc[ptc, ['rmin',
            'rmin error','rmax','rmax error','r_hill', 'r_hill error']])
        IC50, IC50e = tuple(params_df.loc[ptc, ['r_IC50','r_IC50 error']])
        text_final = text + '$IC_{50}$ = ' + f'{IC50:,.1f} \u00B1 {IC50e:,.1f}'
        
        drop = data_df[data_df['Competitor Concentration'] != 0].dropna(subset=['mean'])   # exclude the protein concentration = 0 point and any NaNs from plotting
        axs.errorbar(drop['Competitor Concentration'], drop['mean'], yerr=drop[err], color='black', fmt='o', capsize=3, marker='s')
        axs.set_xscale('log')
        axs.set_ylabel('Anisotropy')
        axs.set_xlabel(f'[{ptc[2]}] ({unit})')
        vir_data = FA._vir_data(drop, 'Competitor', 200)    # x-axis data for curve plotting
        axs.plot(vir_data, FA._EC50_com(vir_data, *params), color='blue')
       
        if leg == True:   # display title and legend with fitting parameters on the graph
            axs.set_title(f'{ptc[0]}, {ptc[1]}, {ptc[2]}')
            axs.legend([f'logistic fitted curve\n{text_final}'], frameon=False, fontsize=11)
            
        if exp == True:   # save figures in the same directory as the notebook
            fig.savefig(f"rep_{rep[-1]}_r_{str(ptc[0])}_{str(ptc[1])}_{str(ptc[2])}.png", dpi=dpi)
        if type(exp) == str:   # save figures in the user defined directory
            fig.savefig(f"{exp}rep_{rep[-1]}_r_{str(ptc[0])}_{str(ptc[1])}_{str(ptc[2])}.png", dpi=dpi)

    
    def plot_ani_com(self, prot=['all'], trac=['all'], com=['all'], rep=['all'], err='std', legend=True, export=False, dpi=250):   
        """Plots anisotropy against competitor concentration with a fitted logistic curve for specified repeats and 
        proteins, tracers and competitors. 
        
        :param prot: List of protein names for which the graphs are created, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which the graphs are created, defaults to ['all'].
        :type trac: list of str
        :param com: List of competitor names for which the graphs are created, defaults to ['all'].
        :type com: list or list of str
        :param rep: List of repeat numbers for which the graphs are created, defaults to ['all'].
        :type rep: list of int
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem'), defaults to 'std'.
        :type err: str
        :param export:
        :param export: Save the figures (True) in the same directory as this Notebook or provide a path (str) to specified directory
        :type export: bool or 'str'
        :param legend: Display legend and title on the figures, defaults to False.
        :type legend: bool
        :param dpi: Resolution of the figure in points per inch, defaults to 250.
        :type dpi: int
        """
        warnings.filterwarnings("ignore")
        #get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, com, rep)
        unit = str(self.plate_map['Concentration Units'].dropna().unique()[0])   # concentration units
        
        for key, value in data_dict.items():   # iterte over all repeats and create a sperate figure for each repeat
            metadata, data = value.values()
            fit_params = data['fit_params']
            
            for idx, ptc in enumerate(ptc_list):   # for each portein-tracer pair plot two graphs: anisotropy and intensity
                r_data_df = data['r_mean'][ptc]   # extract the df with anisotropy and intensity
                FA._plot_ani_com(r_data_df, fit_params, ptc, err, key, unit, exp=export, leg=legend, dpi=dpi)
              
            
    def save_ani_figs(self, prot=['all'], trac=['all'], rep=['all'], var='both', path='', err='std', legend=False, dpi=250):
        """Saves single figures of anisotropy and intensity for specified repeats and protein-tracer pairs in the same 
        directory as this notebook or in user defined directory if the path is provided.
        
        :param prot: List of protein names for which the graphs are exported, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which the graphs are exported, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which the graphs are exported, defaults to ['all'].
        :type rep: list of ints
        :param var: A variable for which the graphs are exported, either 'r' for anisotropy or 'i' for inteensity, defaults to 'both'.
        :type var: str
        :param path: A path to directory in which the figures are saved, defaults to '' (the same directory as this Jupyter Notebook).
        :type path: str
        :param err: Type of error data displayed as error bars, either 'std' or 'sem', defaults to 'std'.
        :type err: str
        :param legend: Display legend and title on the figures, defaults to False.
        :type legend: bool
        :param dpi: Resolution of the figure in points per inch, defaults to 250.
        :type dpi: int
        """
        # get data_dict and a list of protein-tracer names
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, ['all'], rep)
        unit = str(self.plate_map['Concentration Units'].dropna().unique()[0])   # concentration units
        
        if len(self.plate_map['Tracer Concentration'].dropna().unique()) == 1:   # protein is titrated to a constant amount of tracer
            t_type = ('Protein', 0)
        if len(self.plate_map['Protein Concentration'].dropna().unique()) == 1:   # tracer is titrated to a constant amount of protein
            t_type = ('Tracer', 1)
        
        for key, value in self.data_dict.items():   # iterate over all repeats
            metadata, data = value.values()
            fit_params = data['fit_params']
            
            for ptc in ptc_list:   # iterate over each protein-tracer pair in
                r_data_df, i_data_df = data['r_mean'][ptc], data['i_mean'][ptc]   # extract the df with anisotropy and intensity dfs
                
                if var == 'r' or var == 'both':
                    fig, axs = plt.subplots(figsize=(6.4, 4.8), tight_layout=True)   # create a figure with a single axis for anisotropy 
                    FA._plot_ani(r_data_df, fit_params, ptc, t_type, fig, axs, err, 'r', key, unit, exp=path, disp=False, leg=legend, dpi=dpi)
                
                if var == 'i' or var == 'both':
                    fig, axs = plt.subplots(figsize=(6.4, 4.8), tight_layout=True)   
                    FA._plot_ani(i_data_df, fit_params, ptc, t_type, fig, axs, err, 'i', key, unit, exp=path, disp=False, leg=legend, dpi=dpi)
        
        print('The figures were successfully exported.')
    
    def _plot_kd(data_df, ptc, final_fit, t_type, err, rep, unit, exp, leg, dpi):
        """Plots amount bound against protein or tracer concentration with a fitted curve on a separate figure for a specific protein-tracer pair.
        
        :param data_df: Data frame with mean values of amount of tracer bound and their associated errors
        :type data_df: pandas df
        :param ptc: Protein-tracer-competitor names for which data will be plotted
        :type ptc: list of tuples
        :param t_type: Type of titration ('Protein' or 'Tracer')
        :type t_type: str
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem')
        :type err: str
        :param rep: Repeat number for labelling of the graph
        :type rep: 'str'
        :param unit: Concentration units to be displayed on the plots
        :type unit: str
        :param leg: Determines whether the legend and box with fitting parameters will be displayed on the figure, default True
        :type leg: bool
        :param exp: Determines whether the figure will be saved, can be either bool or string with directory path
        :type exp: bool or 'str'
        :param dpi: Resolution of the figure in points per inch
        :type dpi: int
        """
        drop = data_df[data_df[f'{t_type} Concentration'] != 0].dropna(subset=['mean'])   # exclude the protein concentration = 0 point and any NaNs from data fitting
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8), tight_layout=True)
        
        # define the x axis data and labels for protein and tracer titration cases
        if t_type == 'Protein':   
            LT, LTe, Kd, Kde = tuple(final_fit.loc[ptc, ['LT','LT error','Kd','Kd error']]) 
            text = '$L_{T}$ = ' + f'{LT:,.2f} \u00B1 {LTe:,.2f}\n' + '$K_{d}$ = ' + f'{Kd:,.2f} \u00B1 {Kde:,.2f}'  
            xlabel = f'[{ptc[0]}] ({unit})'
            params = (LT, Kd)
            
        if t_type == 'Tracer':
            PT, PTe, Kd, Kde = tuple(final_fit.loc[ptc, ['PT','PT error','Kd','Kd error']])
            text = '$P_{T}$ = ' + f'{PT:,.2f} \u00B1 {PTe:,.2f}\n' + '$K_{d}$ = ' + f'{Kd:,.2f} \u00B1 {Kde:,.2f}'
            xlabel = f'[{ptc[1]}] ({unit})'
            params = (PT, Kd)
            
        axs.errorbar(drop[f'{t_type} Concentration'], drop['mean'], yerr=drop[err], color='black', fmt='o', capsize=3, marker='s')
        axs.set_xscale('log')
        axs.set_ylabel(f'[Fluorescent Tracer Bound] ({unit})')
        axs.set_xlabel(xlabel)
        vir_data = FA._vir_data(drop, t_type, 200)
        axs.plot(vir_data, FA._LB(vir_data, *params), color='blue')
        
        if leg == True:   # display the figure title, legend and annotation with fitting params
            if rep[1] > 1:   # do not display info about repeat number if there is only one repeat
                axs.set_title(f'Repeat {rep[0][-1]}, Protein: {ptc[0]}, Tracer: {ptc[1]}')
            else:
                axs.set_title(f'Protein: {ptc[0]}, Tracer: {ptc[1]}')
            
            axs.legend([f'single site fitted curve\n{text}'], fontsize=11, frameon=False)
        
        plt.show()
        
        if exp == True:   # save the figure to the same directory as the notebook
            fig.savefig(f"Kd_plot_rep_{rep[0][-1]}_{str(ptc[0])}_{str(ptc[1])}.png", dpi=dpi)
        if type(exp) == str:   # save the figure to user defined directory
            fig.savefig(f"{exp}Kd_plot_rep_{rep[0][-1]}_{str(ptc[0])}_{str(ptc[1])}.png", dpi=dpi)
    
    def _overlay_kd_plots(plate_map, data_dict, ptc_list, final_fit, t_type, err, unit, exp, leg, dpi):   
        """Creates a figure with overlayed plots for specified protein-tracer pairs and repeats 
        
        :param plate_map: Platemap
        :type plate_map: pandas df
        :param data_dict: Data dictionary containing the specific repeats for which data will be plotted
        :type data_dict: dict
        :param ptc: List of protein-tracer names for which data will be plotted
        :type ptc: list of tuples
        :param t_type: Type of titration ('Protein' or 'Tracer')
        :type t_type: str
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem')
        :type err: str
        :param unit: Concentration units to be displayed on the plots
        :type unit: str
        :param leg: Determines whether the legend and box with fitting parameters will be displayed on the figure, default True
        :type leg: bool
        :param exp: Determines whether the figure will be saved, can be either bool or string with directory path
        :type exp: bool or 'str'
        :param dpi: Resolution of the figure in points per inch
        :type dpi: int
        """
        if len(ptc_list) < 2:
            raise DataError('At least two data sets are required for overlayed plot.')
        
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8), tight_layout=True) 
        text_final = []   # list to store the legend string for each data set
        cmaps = ['Blues', 'Greens', 'Oranges', 'Purples', 'Reds', 'Greys', 'YlOrBr', 'YlOrRd', 
                 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
        iter_cmaps = iter(cmaps)
        
        for key, value in data_dict.items():   # iterte through all repeats of the defined data_dict
            metadata, data = value.values()
            
            for ptc in ptc_list:    # iterate through the list of protein-tracer names to plot its data on the same figure
                data_df = data['amount_bound'][ptc]   # extract the correct df with amount bound for a given protein-tracer pair
                drop = data_df[ data_df[f'{t_type} Concentration'] != 0].dropna(subset=['mean'])   # exclude the protein concentration = 0 point and any NaNs from data fitting
                
                if t_type == 'Protein':  
                    LT, LTe, Kd, Kde = tuple(final_fit.loc[ptc, ['LT','LT error','Kd','Kd error']]) 
                    text = '$L_{T}$ = ' + f'{LT:,.2f} \u00B1 {LTe:,.2f}\n' + '$K_{d}$ = ' + f'{Kd:,.2f} \u00B1 {Kde:,.2f}'  
                    params = (LT, Kd)
                
                if t_type == 'Tracer':   
                    PT, PTe, Kd, Kde = tuple(final_fit.loc[ptc, ['PT','PT error','Kd','Kd error']])
                    text = '$P_{T}$ = ' + f'{PT:,.2f} \u00B1 {PTe:,.2f}\n' + '$K_{d}$ = ' + f'{Kd:,.2f} \u00B1 {Kde:,.2f}'
                    params = (PT, Kd)
                
                if len(data_dict) > 1:   # do not display info about repeat number if there is only one repeat
                    text_long = f"rep {key[-1]}, {ptc[0]}, {ptc[1]}\n{text}"
                else: 
                    text_long = f"{ptc[0]}, {ptc[1]}\n{text}"
                    
                text_final.append(text_long)
                vir_data = FA._vir_data(drop, t_type, 200)
                cmap = plt.cm.get_cmap(next(iter_cmaps))   # take the next color map from the list 
                axs.errorbar(drop[f'{t_type} Concentration'], drop['mean'], yerr=drop[err], fmt='o', capsize=3, marker='s', color=cmap(0.95))
                axs.plot(vir_data, FA._LB(vir_data, *params), color=cmap(0.50))  
                
        axs.set_xscale('log')
        axs.set_ylabel(f'[Fluorescent Tracer Bound] ({unit})')
        axs.set_xlabel(f'[{t_type}] ({unit})')

        if leg == True:   # display the figure title, legend and annotation with fitting params
            axs.set_title(f'Overlayed plot')
            lbox = axs.legend(text_final, fontsize=11, frameon=False, loc='upper left', bbox_to_anchor=(1.03, 0.95))
            fig.canvas.draw()   # draw the  canvas so that figure and legend size is defined
            # calculate length by which the figure will be widened to accomodate the legend
            w = (lbox.get_window_extent().width + (0.06 * axs.get_window_extent().width)) / fig.dpi
            fig.set_size_inches(6.4 + w, 4.8)   # resize the figure
            
        plt.show()
        
        if exp == True:   # save the figure to the same directory as the notebook
            fig.savefig(f"Overlayed_Kd_plot.png", dpi=dpi) 
        if type(exp) == str:   # save the figure to user defined directory
            fig.savefig(f"{exp}Overlayed_Kd_plot.png",dpi=dpi)
        
    def plot_kd(self, prot=['all'], trac=['all'], rep=['all'], err='std', overlay=False, legend=True, export=False, dpi=250):   
        """Plots the concentration of fluorescent tracer bound to target protein against the protein (or tracer) concentration.
        
        :param prot: List of protein names for which the graphs will be created, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which the graphs will be created, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which the graphs will be created, defaults to ['all'].
        :type rep: list of ints
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem'), defaults to 'std'.
        :type err: str
        :param overlay: Overlayes the data on a single figure, defaults to False.
        :type overlay: bool
        :param legend: Display the figure title and legend, defaults to True.
        :type legend: bool
        :param export: Save the figures (True) in the same directory as this Notebook or provide a path (str) to specified directory
        :type export: bool or 'str
        :param dpi: Resolution of the exported figure in dots per inches, defaults to 250.
        :type dpi: int
        """
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, ['all'], rep)
        unit = str(self.plate_map['Concentration Units'].dropna().unique()[0])   # concentration units
        
        if len(self.plate_map['Tracer Concentration'].dropna().unique()) == 1:   # protein is titrated to a constant amount of tracer
            t_type = 'Protein'
        if len(self.plate_map['Protein Concentration'].dropna().unique()) == 1:   # tracer is titrated to a constant amount of protein
            t_type = 'Tracer'
        
        if overlay == False:
            for key, value in data_dict.items():   # iterte through all repeats of the defined data_dict
                metadata, data = value.values()
                rep = (key, len(data_dict))
                
                for ptc in ptc_list:    # iterate through the list of protein-tracer names to create a separate figure for each pair
                    data_df = data['amount_bound'][ptc]   # extract the correct df with amount bound for a given protein-tracer pair
                    FA._plot_kd(data_df, ptc, self.final_fit, t_type, err, rep, unit, export, legend, dpi)
        else:
            FA._overlay_kd_plots(self.plate_map, data_dict, ptc_list, self.final_fit, t_type, err, unit, export, legend, dpi)
        
    def _plot_ic50(data_df, params_df, ptc, err, rep, unit, exp, leg, dpi):
        """Plots amount bound against protein or tracer concentration with a fitted curve on a separate figure. 
        
        :param data_df: Data frame with mean values of amount of tracer bound and their associated errors
        :type data_df: pandas df
        :param ptc: Protein and tracer names for which data will be plotted
        :type ptc: list of tuples
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem')
        :type err: str
        :param rep: Repeat number for labelling of the graph !!!!!!!!!!!!!!!!!!!!!!!
        :type rep: tuple
        :param unit: Concentration units to be displayed on the plots
        :type unit: str
        :param leg: Determines whether the legend and box with fitting parameters will be displayed on the figure, default True
        :type leg: bool
        :param exp: Determines whether the figure will be saved, can be either bool or string with directory path
        :type exp: bool or 'str'
        :param leg: Determines whether the legend and box with fitting parameters will be displayed on the figure, default True
        :type leg: bool
        :param dpi: Resolution of the figure in points per inch
        :type dpi: int
        """
        drop = data_df[data_df['Competitor Concentration'] != 0].dropna(subset=['mean'])   # exclude the protein concentration = 0 point and any NaNs from data fitting
        fig, axs = plt.subplots(1, 1, figsize=(6.4, 4.8), tight_layout=True)
        
        params = tuple(params_df.loc[ptc, ['min','max','IC50','hill']])
        ic50, ic50e, Ki, Kie, Ki2, Ki2e = tuple(params_df.loc[ptc, ['IC50','IC50 error','Ki','Ki error','Ki*','Ki* error']])
        text = "$IC_{50}$" + f" = {ic50:,.2f} \u00B1 {ic50e:,.2f}\n" + "$K_i$" + f" = {Ki:,.2f} \u00B1 {Kie:,.2f}\n" + "$K_i*$" + f" = {Ki2:,.2f} \u00B1 {Ki2e:,.2f}" 
        axs.errorbar(drop['Competitor Concentration'], drop['mean'], yerr=drop[err], color='black', fmt='o', capsize=3, marker='s')
        axs.set_xscale('log')
        axs.set_ylabel(f'[Fluorescent Tracer Bound] ({unit})')
        axs.set_xlabel(f'[{ptc[2]}] ({unit})')
        vir_data = FA._vir_data(drop, 'Competitor', 200)
        axs.plot(vir_data, FA._EC50_com(vir_data, *params), color='blue')
        
        if leg == True:   # display the figure title and legend with fitting params
            if rep[1] > 1:    # do not display info about repeat number if there is only one repeat
                axs.set_title(f'Repeat {rep[0][-1]}, {ptc[0]}, {ptc[1]}, {ptc[2]}')
            else:
                axs.set_title(f'{ptc[0]}, {ptc[1]}, {ptc[2]}')
            axs.legend([f'single site fitted curve\n{text}'], fontsize=11, frameon=False)
            
        plt.show()
        
        if exp == True:   # save the figure to the same directory as the notebook
            fig.savefig(f"IC50_plot_rep_{rep[0][-1]}_{str(ptc[0])}_{str(ptc[1])}_{str(ptc[2])}.png", dpi=dpi)
        if type(exp) == str:   # save the figure to user defined directory
            fig.savefig(f"{exp}IC50_plot_rep_{rep[0][-1]}_{str(ptc[0])}_{str(ptc[1])}_{str(ptc[2])}.png", dpi=dpi)
        
        
    def plot_ic50(self, prot=['all'], trac=['all'], rep=['all'], com=['all'], err='std', legend=True, export=False, dpi=250):   
        """Plots the concentration of the fluoorescent tracer bound to the target protein against competitor concentration.
        Calculates the binding constant (Ki) for each competitor.
        
        :param prot: List of protein names for which the graphs will be created, defaults to ['all'].
        :type prot: list of str
        :param trac: List of tracer names for which the graphs will be created, defaults to ['all'].
        :type trac: list of str
        :param rep: List of repeat numbers for which the graphs will be created, defaults to ['all'].
        :type rep: list of ints
        :param com: List of competitor names for which the graphs are created, defaults to ['all'].
        :type com: list or list of str
        :param err: Type of error data displayed as error bars, either standard deviation ('std') or standard error ('sem'), defaults to 'std'.
        :type err: str
        :param legend: Display the figure title and legend, defaults to True.
        :type legend: bool
        :param export: Save the figures (True) in the same directory as this Notebook or provide a path (str) to specified directory
        :type export: bool or 'str
        :param dpi: Resolution of the exported figure in dots per inches, defaults to 250.
        :type dpi: int
        """
        warnings.filterwarnings("ignore")
        data_dict, ptc_list = FA._get_items_to_plot(self.data_dict, self.plate_map, prot, trac, com, rep)
        unit = str(self.plate_map['Concentration Units'].dropna().unique()[0])
        
        print("The Ki* is calculated based on the total protein concentration calculated from the measured anisotropy. Below each figure the values of total protein concentration from platemap (LT) and as calculated from measured anisotropy (LT*) are stated.")
        
        for key, value in self.data_dict.items():   # iterte through all repeats of the defined data_dict
            metadata, data = value.values()
            params_df = data['fit_params_com']
            rep = (key, len(data_dict))
            
            for ptc in data['amount_bound'].keys():    # iterate through the list of protein-tracer names to create a separate figure for each pair
                data_df = data['amount_bound'][ptc]   # extract the correct df with amount bound for a given protein-tracer pair
                params = FA._calc_Ki(ptc, params_df, self.plate_map, self.final_fit)    # calculate Ki, Ki* and LT
                params_df.loc[ptc, ['Ki','Ki error','Ki*','Ki* error']] = params[0:4]   # instert Ki into the fit_params_com df
                
                if key in data_dict.keys() and ptc in ptc_list:   # plot figures only for user specified proteins, tracersandcompetitors
                    FA._plot_ic50(data_df, params_df, ptc, err, rep, unit, export, legend, dpi)
                    print(f'LT = {params[4]:,.1f} {unit}, LT* = {params[5]:,.1f} {unit}')
    
    
    ##### Fittig params set, export and import functions #####
    def set_fitparams(self, prot, trac, **kwargs):
        """Allows to set a value of any parameter in the final fit data frame for a specific protein-tracer pair.
        
        :param prot: Protein name.
        :type prot: str
        :param trac: Tracer name.
        :type trac: str
        :param **kwargs: Keyword arguments represeting the parameter and its value, e.g. lambda=1.5, rmin=0.30
        """
        wrong_cols = []
        
        for key, value in kwargs.items():   # iterate over the kwargs dictionary
            if key not in self.final_fit.columns:
                wrong_cols.append(key)
            else:
                self.final_fit.loc[(prot, trac), key] = value   # overwrite the parameters in fitting params df with all params passed as keyword arguments
        
        if wrong_cols != []:
            warnings.warn(f'No such columns in the final_fit data frame:\n{wrong_cols}')
        
    def export_params(self, path='', file_type='csv'):
        """Export the final_fit, fitting_params and, in case of competition data, fitting_params_com for each repeat to csv or excel files.
        
        :param path: A path to directory in which the file is saved, defaults to '' (i.e. the same directory as this Jupyter Notebook)
        :type path: str
        :param file_type: Type of file generated, either 'csv' or 'excel' file, defaults to csv
        :type file_type: 'str'
        """
        if file_type == 'csv':   # export as csv file
            self.final_fit.to_csv(path_or_buf=f"{path}final_fit_params.csv")
        if file_type == 'excel':   # export as excel file
            self.final_fit.to_excel(excel_writer=f"{path}all_fit_params.xlsx", sheet_name="final_fit_params")
    
        for key, value in self.data_dict.items():   #iterate over all repeats
            metadata, data = value.values()
            
            if file_type == 'csv':   # export as csv file
                data['fit_params'].to_csv(path_or_buf=f"{path}rep_{key[-1]}_fit_params.csv")
                if 'fit_params_com' in data.keys():
                    data['fit_params_com'].to_csv(path_or_buf=f"{path}rep_{key[-1]}_fit_params_com.csv")
            
            if file_type == 'excel':   # export as excel file
                with pd.ExcelWriter(f"{path}all_fit_params.xlsx", engine='openpyxl', mode='a') as writer:
                    data['fit_params'].to_excel(writer, sheet_name=f"{key}_fit_params")
                    if 'fit_params_com' in data.keys():
                        data['fit_params_com'].to_excel(writer, sheet_name=f"{key}_fit_params_com")
                    
        print(f'The fitting parameters were exported to the {file_type} files.')
        
        
    def import_params(self, csv_file):
        """Allows to import a csv file with final_fit parameters (i.e. rmin, rmax, lamda, Kd and their errors).
        
        :param csv_file: A csv file path with parameters to be imported
        :type csv_file: str
        """
        with open(csv_file) as file:   # read the csv into pandas df
            df = pd.read_csv(file, sep=',', index_col=[0,1], engine='python', encoding='utf-8')   # import with multiindex
        
        if list(df[df.columns[0]].unique()) != ['-']:   # df contains copetitor names
            df = df.set_index(df.columns[0], append=True).rename_axis([None,None,None])   # add competitor name column to multiinex
        else:   # no competitor name, so delete the first column containg only '-' 
            df.drop(df.columns[0], axis=1, inplace=True)

        cols = df.columns.intersection(self.final_fit.columns)   # columns common to imported df and final_fit df

        for index in list(df.index):   # iterate over the indexes of imported df
            self.final_fit.loc[index, cols] = df.loc[index, cols].tolist()   # overwrite the existing values in the final_fit df with the ones from imported df

        col_diff = list(df.columns.difference(self.final_fit.columns))   # list of clumns present in imported df but absent from final_fit df
        if col_diff != []:   # display warning about missing columns in the final_fit
            warnings.warn(f"The final_fit data frame does not contain following columns:\n'{col_diff}'")
        
       
    def export_data(self, path=''):
        """Saves the mean anisotropy, intensity and amount bound data along with their standard deviation 
        and standard error into excel file.
        
        :param path: Path to the folder in wchich the excel file is saved.
        :type path: str
        """
        for key, value in self.data_dict.items():
            
            metadata, data = value.values()
            ptc_list = list(data['r_mean'].keys())   # list of all protein-tracer names

            for ptc in ptc_list:
                
                # remove redundant columns and rename the remaining ones for anisotropy, intensity and amount bound dfs
                r_df = data['r_mean'][ptc].drop(['Protein Name','Tracer Name','Competitor Name'], axis=1)
                r_df2 = r_df.rename(columns={'mean': 'anisotropy mean', 'std': 'ani std', 'sem': 'ani sem'}).set_index('Protein Concentration')
                i_df = data['i_mean'][ptc].drop(['Protein Name','Tracer Name','Competitor Name'], axis=1)
                i_df2 = i_df.rename(columns={'mean': 'intensity mean', 'std': 'int std', 'sem': 'int sem'}).set_index('Protein Concentration')
                ab_df = data['amount_bound'][ptc].drop(['Protein Name','Tracer Name','Competitor Name'], axis=1)
                ab_df2 = ab_df.rename(columns={'mean': 'amount bound mean', 'std': 'ab std', 'sem': 'ab sem'}).set_index('Protein Concentration')
                
                # join the anisotropy, intensity and amount bound dfs together 
                m = pd.concat([r_df2, i_df2, ab_df2], axis=1)
                
                if ptc == ptc_list[0]:   # for the first iteration create the excel file
                    m.to_excel(f"{path}Anisotropy Data.xlsx", sheet_name=f'rep_{key[-1]}_{ptc[0][:7]}_{ptc[1][:7]}_{ptc[2][:7]}')
                    
                else:     # for next iterations append new sheet to the existing excel file 
                    with pd.ExcelWriter(f"{path}Anisotropy Data.xlsx", engine='openpyxl', mode='a') as writer:
                        m.to_excel(writer, sheet_name=f'rep_{key[-1]}_{ptc[0][:7]}_{ptc[1][:7]}_{ptc[2][:7]}')
            