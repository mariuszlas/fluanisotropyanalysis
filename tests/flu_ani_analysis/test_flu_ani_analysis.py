import pandas as pd
import pytest
import pickle
import os
import matplotlib.pyplot as plt
from matplotlib.testing.compare import compare_images
from flu_ani_analysis.flu_ani_analysis_module import FA as fa    

# raw data files in .csv format
plate_1 = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\plate1.csv"
plate_1_repeat = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\plate1_repeat.csv"
plate_2_1 = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\plate2_1.csv"
plate_2_repeat = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\plate2_repeat.csv"
plate_2_repeat_96 = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\plate2_repeat _96.csv"
list_A = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\listA.csv"
list_A_repeat = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\listA_repeat.csv"
list_B = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\listB.csv"
list_B_repeat_end = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\listB_repeat _end.csv"
list_B_repeat_96 = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\listB_repeat _96.csv"
list_C = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\listC.csv"

# files containing lists of expected data frames in .pkl format
plate_1_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\plate_1_test.pkl"
plate_1_repeat_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\plate_1_repeat_test.pkl"
plate_2_1_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\plate_2_1_test.pkl"
plate_2_repeat_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\plate_2_repeat_test.pkl"
plate_2_repeat_96_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\plate_2_repeat_96_test.pkl"
list_A_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\list_A_test.pkl"
list_A_repeat_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\list_A_repeat_test.pkl"
list_B_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\list_B_test.pkl"
list_B_repeat_end_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\list_B_repeat_end_test.pkl"
list_B_repeat_96_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\list_B_repeat_96_test.pkl"
list_C_test = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\list_C_test.pkl"

inval_platemap = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\test_inval_platemap.csv"


HsHis6_PEX5C_vs_HsPEX5C_p_s_corrected = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\HsHis6_PEX5C_vs_HsPEX5C_p_s_corrected.pkl"
HsHis6_PEX5C_vs_HsPEX5C_calc_r_I = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\example outputs\\HsHis6_PEX5C_vs_HsPEX5C_calc_r_I_out.pkl"


HsHis6_PEX5C_vs_HsPEX5C = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\real data\\Hs-His6-PEX5C vs HsPEX5C.csv"
HsPEX5C_Y467C_vs_AtPEX5C_WT = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\real data\\HsPEX5C Y467C vs AtPEX5C WT.csv"
F606C_vs_AtPEX5C_WT_1_hour = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\real data\\F606C vs AtPEX5C WT 1 hour.csv"

plate_map_file = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\plate_map.csv"
Hs_His6_PEX5C_vs_HsPEX5C_platemap = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\real data\\Hs-His6-PEX5C vs HsPEX5C platemap.csv"
HsPEX5C_Y467C_vs_AtPEX5C_WT_platemap = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\real data\\HsPEX5C Y467C vs AtPEX5C WT platemap.csv"
F606C_vs_AtPEX5C_WT_1_hour_platemap = "C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\Test data\\real data\\F606C vs AtPEX5C WT 1 hour platemap.csv"

# expected values of g-factor
expected_g = {'plate_1': 1.15, 'plate_1_repeat': 1.15, 'plate_2_1': 1.15, 'plate_2_repeat': 1.15, 'plate_2_repeat_96': 1.0, 
        'list_A': 1.15, 'list_A_repeat': 1.15, 'list_B': 1.15, 'list_B_repeat_96': 1.0, 'list_B_repeat_end': 1.15, 'list_C': 1.15}


@pytest.fixture
def get_testing_data():
    """Set up for the first 11 tests."""
    
    def _get_testing_data(data_csv, platemap_csv, data_type, size, pkl_file):
        
        with open(pkl_file, 'rb') as file:   # load the list with expexcted data frames from .pkl file
            expected_list = pickle.load(file)

        actual_output = fa.read_in_envision(data_csv = data_csv, platemap_csv=platemap_csv, data_type=data_type, size=size)   # execute the tested function
        actual_g = actual_output.g_factor
        actual_list = []

        for repeat in actual_output.data_dict.values():   # unpack the dictionary with df from the tested function
            metadata, data = repeat.values()
            p_channel, s_channel = data.values()
            actual_list.append(metadata)
            actual_list.append(p_channel)
            actual_list.append(s_channel)
        
        return actual_list, expected_list, actual_g
    
    return _get_testing_data


def test_plate_1(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=plate_1, platemap_csv=plate_map_file, data_type='plate', size=384, pkl_file=plate_1_test)
    assert g_factor == expected_g['plate_1']   # test for the g-factor
    for actual_df, expected_df in zip(actual_list, expected_list):   # zip the two lists containing actual and expected data frames into a list of tuples and iterate over all tuples
        pd.testing.assert_frame_equal(actual_df, expected_df)   # test whether actual and expected dfs are the same
        
        
def test_plate_1_repeat(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=plate_1_repeat, platemap_csv=plate_map_file, data_type='plate', size=384, pkl_file=plate_1_repeat_test)
    assert g_factor == expected_g['plate_1_repeat']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_plate_2_1(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=plate_2_1, platemap_csv=plate_map_file, data_type='plate', size=384, pkl_file=plate_2_1_test)
    assert g_factor == expected_g['plate_2_1']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_plate_2_repeat(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=plate_2_repeat, platemap_csv=plate_map_file, data_type='plate', size=384, pkl_file=plate_2_repeat_test)
    assert g_factor == expected_g['plate_2_repeat']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_plate_2_repeat_96(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=plate_2_repeat_96, platemap_csv=plate_map_file, data_type='plate', size=96, pkl_file=plate_2_repeat_96_test)
    assert g_factor == expected_g['plate_2_repeat_96']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_A(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=list_A, platemap_csv=plate_map_file, data_type='list', size=384, pkl_file=list_A_test)
    assert g_factor == expected_g['list_A']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_A_repeat(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=list_A_repeat, platemap_csv=plate_map_file, data_type='list', size=384, pkl_file=list_A_repeat_test)
    assert g_factor == expected_g['list_A_repeat']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_B(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=list_B, platemap_csv=plate_map_file, data_type='list', size=384, pkl_file=list_B_test)
    assert g_factor == expected_g['list_B']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_B_repeat_end(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=list_B_repeat_end, platemap_csv=plate_map_file, data_type='list', size=384, pkl_file=list_B_repeat_end_test)
    assert g_factor == expected_g['list_B_repeat_end']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_B_repeat_96(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=list_B_repeat_96, platemap_csv=plate_map_file, data_type='list', size=96, pkl_file=list_B_repeat_96_test)
    assert g_factor == expected_g['list_B_repeat_96']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_C(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(data_csv=list_C, platemap_csv=plate_map_file, data_type='list', size=384, pkl_file=list_C_test)
    assert g_factor == expected_g['list_C']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
@pytest.mark.raises()
def test_plate_size_error():
    """Test for error raised if size is not 384 or 96."""
    
    test_object = fa.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='plate', size=100)


@pytest.mark.raises()
def test_incorrect_data_type_list():
    """Test for error if data_type = list but raw data file is in plate format."""
    
    test_object = fa.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='list', size=384)
    
    
@pytest.mark.raises()
def test_incorrect_data_type_plate():
    """Test for error if data_type = plate but raw data file is in list format."""
        
    test_object = fa.read_in_envision(data_csv=list_A, platemap_csv=plate_map_file, data_type='plate', size=384)
    
    
@pytest.mark.raises()
def test_incorrect_data_type():
    """Test for error if data_type argument is neither plate nor list."""
        
    test_object = fa.read_in_envision(data_csv=list_A, platemap_csv=plate_map_file, data_type='typo', size=384)
    

def test_invalidate():
    """Test whether the invalidate function turns the value of the 'Valid' column to False in a given set of well ids, columns and rows."""
    
    example_platemap = pd.read_csv(inval_platemap, index_col=[0])   # read in an example platemap with invalidated well ids, rows and columns
    test_object = fa.read_in_envision(data_csv=plate_2_repeat, platemap_csv=plate_map_file, data_type='plate', size=384)   # read in actual data and plate map
    test_object.invalidate(wells=['A2', 'B3', 'E4'], rows=['C', 'G'], columns=[7,8,12,20])   # invalidate specific well ids, rows and columns
    pd.testing.assert_frame_equal(test_object.plate_map, example_platemap, check_dtype=False)   # compare the two dfs without checking the data types because the example df was not read in using the read_in_envision function
    
@pytest.mark.raises()
def test_invalidate_error():
    """Test whether the 'invalidate' function raises an error if no arguments are passed."""
    
    test_object = fa.read_in_envision(data_csv=plate_2_repeat, platemap_csv=plate_map_file, data_type='plate', size=384)
    test_object.invalidate()   # execute the invalidate function without specifying well ids, rows or columns to be invalidated
    

def test_background_correct():
    """Tests whether the background correction function performs correct calculations to get the background corrected values of p and s channel signal"""
    
    with open(HsHis6_PEX5C_vs_HsPEX5C_p_s_corrected, 'rb') as file:   # load the list with expexcted data frames from .pkl file
        expected_list = pickle.load(file)

    test_object = fa.read_in_envision(data_csv=HsHis6_PEX5C_vs_HsPEX5C, platemap_csv=Hs_His6_PEX5C_vs_HsPEX5C_platemap, data_type='plate', size=384)   # execute the tested function
    test_object.background_correct()
    
    # assert the p_corrected and s_corrected data frames are the same as the reference data frames 
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['p_corrected'], expected_list[0], atol=1E-6)
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['s_corrected'], expected_list[1], atol=1E-6)
    
    
def test_calculate_r_i():
    """Tests whether the calculate_r_I function performs correct calculations to get the raw and background corrected values of intensity and anisotropy"""
    
    with open(HsHis6_PEX5C_vs_HsPEX5C_calc_r_I, 'rb') as file:   # load the list with expexcted data frames from .pkl file
        expected_list = pickle.load(file)
    
    test_object = fa.read_in_envision(data_csv=HsHis6_PEX5C_vs_HsPEX5C, platemap_csv=Hs_His6_PEX5C_vs_HsPEX5C_platemap, data_type='plate', size=384)
    test_object.background_correct()
    test_object.calculate_r_i(correct=True, plot_i=False, thr=50)
    
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['i_raw'], expected_list[0], atol=1E-6)
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['r_raw'], expected_list[1], atol=1E-6)
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['i_corrected'], expected_list[2], atol=1E-6)
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['r_corrected'], expected_list[3], atol=1E-6)
    pd.testing.assert_frame_equal(test_object.data_dict['repeat_1']['data']['i_percent'], expected_list[4], atol=1E-6)
    

@pytest.mark.raises()
def test_no_backg_subt():
    """Test for an error raised if the calculate_r_i function is called with the correct parameter as True prior to the background subtraction"""
    
    test_object = fa.read_in_envision(data_csv=HsHis6_PEX5C_vs_HsPEX5C, platemap_csv=Hs_His6_PEX5C_vs_HsPEX5C_platemap, data_type='plate', size=384)
    test_object.calculate_r_i(correct=True, plot_i=False, thr=80)

"""
def test_visualise():
    
    actual_img = 'C:\\Users\\Bartek\\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\result_images\\actual_img.png'
    expected_img = 'C:\\Users\\Bartek\Documents\\Fluorescence-Anisotropy-Analysis\\tests\\baseline_images\\expected_img.png'
    
    test_object = fa.read_in_envision(data_csv=HsHis6_PEX5C_vs_HsPEX5C, platemap_csv=Hs_His6_PEX5C_vs_HsPEX5C_platemap, data_type='plate', size=384)
    test_object.visualise(labelby='Protein Concentration', colorby='Type', title='Test plot', export=actual_img)
    
    result = compare_images(expected_img, actual_img, tol=1)
    
    if os.path.isfile(actual_img):
        os.remove(actual_img)
    
    assert result == None"""