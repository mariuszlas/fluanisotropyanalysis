import pandas as pd
import pytest
import pickle
from flu_ani_analysis import flu_ani_analysis as fa

# define well plate dimensions
plate_dimensions = {'96':(8, 12), '384':(16, 24)}
    

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

# expected values of g-factor
expected_g = {'plate_1': 1.15, 'plate_1_repeat': 1.15, 'plate_2_1': 1.15, 'plate_2_repeat': 1.15, 'plate_2_repeat_96': 1.0, 
        'list_A': 1.15, 'list_A_repeat': 1.15, 'list_B': 1.15, 'list_B_repeat_96': 1.0, 'list_B_repeat_end': 1.15, 'list_C': 1.15}


@pytest.fixture
def get_testing_data():
    """Set up for the first 11 tests."""
    
    def _get_testing_data(csv_file, data_type, size, pkl_file):
        
        with open(pkl_file, 'rb') as file:   # load the list with expexcted data frames from .pkl file
            expected_list = pickle.load(file)

        actual_output = fa.FA.read_in_envision(csv_file = csv_file, data_type=data_type, size=size)   # execute the tested function
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
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=plate_1, data_type='plate', size=384, pkl_file=plate_1_test)
    assert g_factor == expected_g['plate_1']   # test for the g-factor
    for actual_df, expected_df in zip(actual_list, expected_list):   # zip the two lists containg actual and expected data frames into a list of tuples and iterate over all tuples
        pd.testing.assert_frame_equal(actual_df, expected_df)   # test whether actual and expected dfs are the same
        
        
def test_plate_1_repeat(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=plate_1_repeat, data_type='plate', size=384, pkl_file=plate_1_repeat_test)
    assert g_factor == expected_g['plate_1_repeat']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_plate_2_1(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=plate_2_1, data_type='plate', size=384, pkl_file=plate_2_1_test)
    assert g_factor == expected_g['plate_2_1']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_plate_2_repeat(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=plate_2_repeat, data_type='plate', size=384, pkl_file=plate_2_repeat_test)
    assert g_factor == expected_g['plate_2_repeat']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_plate_2_repeat_96(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=plate_2_repeat_96, data_type='plate', size=96, pkl_file=plate_2_repeat_96_test)
    assert g_factor == expected_g['plate_2_repeat_96']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_A(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=list_A, data_type='list', size=384, pkl_file=list_A_test)
    assert g_factor == expected_g['list_A']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_A_repeat(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=list_A_repeat, data_type='list', size=384, pkl_file=list_A_repeat_test)
    assert g_factor == expected_g['list_A_repeat']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_B(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=list_B, data_type='list', size=384, pkl_file=list_B_test)
    assert g_factor == expected_g['list_B']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_B_repeat_end(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=list_B_repeat_end, data_type='list', size=384, pkl_file=list_B_repeat_end_test)
    assert g_factor == expected_g['list_B_repeat_end']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_B_repeat_96(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=list_B_repeat_96, data_type='list', size=96, pkl_file=list_B_repeat_96_test)
    assert g_factor == expected_g['list_B_repeat_96']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
def test_list_C(get_testing_data):
    
    actual_list, expected_list, g_factor = get_testing_data(csv_file=list_C, data_type='list', size=384, pkl_file=list_C_test)
    assert g_factor == expected_g['list_C']
    for actual_df, expected_df in zip(actual_list, expected_list):
        pd.testing.assert_frame_equal(actual_df, expected_df)
        
        
@pytest.mark.raises()
def test_plate_size_error():
    test_object = fa.FA.read_in_envision(csv_file=plate_1, data_type='plate', size=100)


@pytest.mark.raises()
def test_incorrect_data_type_list():
    test_object = fa.FA.read_in_envision(csv_file=plate_1, data_type='list', size=384)
    
    
@pytest.mark.raises()
def test_incorrect_data_type_plate():
    test_object = fa.FA.read_in_envision(csv_file=list_A, data_type='plate', size=384)