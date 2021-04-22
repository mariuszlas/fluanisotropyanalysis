import os
import pytest
import pickle
import pandas as pd
from fluanisotropyanalysis.flu_ani_analysis import FA   
from fluanisotropyanalysis.flu_ani_analysis import DataError
from fluanisotropyanalysis.flu_ani_analysis import DataTypeError
from fluanisotropyanalysis.flu_ani_analysis import PlateSizeError

# platemap file for read_in_envision and invalidated tests
plate_map_file = "tests/test_data/plate_map_for_read_in_env_&_inval.csv"

# raw data files in csv format for read_in_envision tests
plate_1 = "tests/test_data/read_in_envision/plate1.csv"
plate_1_repeat = "tests/test_data/read_in_envision/plate1_repeat.csv"
plate_2_1 = "tests/test_data/read_in_envision/plate2_1.csv"
plate_2_repeat = "tests/test_data/read_in_envision/plate2_repeat.csv"
plate_2_repeat_96 = "tests/test_data/read_in_envision/plate2_repeat _96.csv"
list_A = "tests/test_data/read_in_envision/listA.csv"
list_A_repeat = "tests/test_data/read_in_envision/listA_repeat.csv"
list_B = "tests/test_data/read_in_envision/listB.csv"
list_B_repeat_end = "tests/test_data/read_in_envision/listB_repeat _end.csv"
list_B_repeat_96 = "tests/test_data/read_in_envision/listB_repeat _96.csv"
list_C = "tests/test_data/read_in_envision/listC.csv"

# files containing lists of expected data frames in pkl format for read_in_envision tests
plate_1_test = "tests/example_output/read_in_envision/plate_1_out.pkl"
plate_1_repeat_test = "tests/example_output/read_in_envision/plate_1_repeat_out.pkl"
plate_2_1_test = "tests/example_output/read_in_envision/plate_2_1_out.pkl"
plate_2_repeat_test = "tests/example_output/read_in_envision/plate_2_repeat_out.pkl"
plate_2_repeat_96_test = "tests/example_output/read_in_envision/plate_2_repeat_96_out.pkl"
list_A_test = "tests/example_output/read_in_envision/list_A_out.pkl"
list_A_repeat_test = "tests/example_output/read_in_envision/list_A_repeat_out.pkl"
list_B_test = "tests/example_output/read_in_envision/list_B_out.pkl"
list_B_repeat_end_test = "tests/example_output/read_in_envision/list_B_repeat_end_out.pkl"
list_B_repeat_96_test = "tests/example_output/read_in_envision/list_B_repeat_96_out.pkl"
list_C_test = "tests/example_output/read_in_envision/list_C_out.pkl"
ff_df = "tests/example_output/read_in_envision/final_fit_df.pkl"    # final_fit data frame

# expected values of g-factor for read_in_envision tests
exp_g = {'plate_1': 1.15, 'plate_1_repeat': 1.15, 'plate_2_1': 1.15, 'plate_2_repeat': 1.15, 'plate_2_repeat_96': 1.0, 
        'list_A': 1.15, 'list_A_repeat': 1.15, 'list_B': 1.15, 'list_B_repeat_96': 1.0, 'list_B_repeat_end': 1.15, 'list_C': 1.15}

# invalidated platemap for invalidate test
inval_platemap = "tests/example_output/invalidate/inval_platemap_out.csv"

# files with protein titration test data and the example outputs
prot_trac_data = "tests/test_data/protein_tracer_data_set.csv"
prot_trac_platemap = "tests/test_data/protein_tracer_platemap.csv"
prot_trac_platemap_df = "tests/example_output/read_in_envision/protein-tracer_platemap_df.csv"
prot_trac_empty_ff = "tests/example_output/read_in_envision/prot_trac_empty_final_fit_df.csv"
prot_trac_p_s_correct = "tests/example_output/background_correct/protein-tracer_p_s_correct.pkl"
prot_trac_r_i = "tests/example_output/calc_r_i/protein-tracer_r_i_i_percent.pkl"
prot_trac_mean_r_i = "tests/example_output/calc_mean_r_i/protein-tracer_mean_r_i_dicts.pkl"
prot_trac_log_fit_params = "tests/example_output/logistic_fit/protein-tracer_log_fit_params.csv"
prot_trac_amount_b = "tests/example_output/calc_amount_bound/protein-tracer_mean_ab_dict.pkl"
prot_trac_ss_final_fit = "tests/example_output/single_site_fit/protein-tracer_final_fit_params.csv"

# files with competition experiment test data and the example outputs
comp_data = "tests/test_data/competition_data_set.csv"
comp_platemap = "tests/test_data/competition_experiment_platemap.csv"
comp_platemap_df = "tests/example_output/read_in_envision/competition_platemap_df.csv"
comp_empty_ff_df = "tests/example_output/read_in_envision/comp_empty_final_fit_df.csv"
comp_p_s_correct = "tests/example_output/background_correct/comp_p_s_correct.pkl"
comp_r_i = "tests/example_output/calc_r_i/competition_r_i_i_percent.pkl"
comp_mean_r_i = "tests/example_output/calc_mean_r_i/competition_mean_r_i_dicts.pkl"
com_log_fit_params = "tests/example_output/logistic_fit/com_log_fit_params.csv"
final_fit_params_for_com = "tests/example_output/calc_amount_bound/final_fit_params_for_comp_experiemnt.csv"
com_amount_b = "tests/example_output/calc_amount_bound/com_mean_ab_dict.pkl"
com_ss_final_fit = "tests/example_output/single_site_fit/com_ss_final_fit.csv"
com_ss_fit_params = "tests/example_output/single_site_fit/com_ss_fit_params.csv"


@pytest.fixture
def get_testing_data():
    """Set up for the first 11 tests. The _get_testing_data reads in the reference data stored in .pkl file, 
    executes the read_in_envision function, assigns its output to new variables and returns a set of actual and 
    reference data that is passed to the test functions."""
    
    def _get_testing_data(data_csv, platemap_csv, data_type, size, pkl_file):
        
        with open(pkl_file, 'rb') as file:   # load the list with expexcted data frames from .pkl file
            exp_list = pickle.load(file)   # dfs in this list are ordered according to repeats, i.e. repeat_1 - metadata, repeat_1 - p, repeat_1 - s, repeat_2 - metadata, etc.

        act_output = FA.read_in_envision(data_csv, platemap_csv, data_type, size)   # execute the tested function
        act_g = act_output.g_factor
        act_list = []

        for rep in act_output.data_dict.values():   # unpack the dictionary with dfs from the tested function
            
            metadata, data = rep.values()
            p_channel, s_channel = data.values()
            act_list.append(metadata)
            act_list.append(p_channel)
            act_list.append(s_channel)
        
        return act_list, exp_list, act_g
    
    return _get_testing_data


# the following 11 tests check reading in the raw data in various formats (plate 1 or 2, list A, B, or C), 
# plate sizes (96 or 384) and number of repeats (1 or 2)
# only data frames containg raw data from p and s channels with corresponding metadata and g-factor values are 
# compared to the reference data

def test_plate_1(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(plate_1, plate_map_file, 'plate', 384, plate_1_test)
    assert g_factor == exp_g['plate_1']   # test for the g-factor
    for act_df, exp_df in zip(act_list, exp_list):   # zip the two lists containing actual and expected data frames into a list of tuples and iterate through all tuples
        pd.testing.assert_frame_equal(act_df, exp_df)   # test whether actual and expected dfs are the same
        
        
def test_plate_1_repeat(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(plate_1_repeat, plate_map_file, 'plate', 384, plate_1_repeat_test)
    assert g_factor == exp_g['plate_1_repeat']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)
        
        
def test_plate_2_1(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(plate_2_1, plate_map_file, 'plate', 384, plate_2_1_test)
    assert g_factor == exp_g['plate_2_1']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)
        
        
def test_plate_2_repeat(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(plate_2_repeat, plate_map_file, 'plate', 384, plate_2_repeat_test)
    assert g_factor == exp_g['plate_2_repeat']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
        
        
def test_plate_2_repeat_96(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(plate_2_repeat_96, plate_map_file, 'plate', 96, plate_2_repeat_96_test)
    assert g_factor == exp_g['plate_2_repeat_96']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)   

        
def test_list_A(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(list_A, plate_map_file, 'list', 384, list_A_test)
    assert g_factor == exp_g['list_A']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
        
        
def test_list_A_repeat(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(list_A_repeat, plate_map_file, 'list', 384, list_A_repeat_test)
    assert g_factor == exp_g['list_A_repeat']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
          
        
def test_list_B(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(list_B, plate_map_file, 'list', 384, list_B_test)
    assert g_factor == exp_g['list_B']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
         
        
def test_list_B_repeat_end(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(list_B_repeat_end, plate_map_file, 'list', 384, list_B_repeat_end_test)
    assert g_factor == exp_g['list_B_repeat_end']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
          
        
def test_list_B_repeat_96(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(list_B_repeat_96, plate_map_file, 'list', 96, list_B_repeat_96_test)
    assert g_factor == exp_g['list_B_repeat_96']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
          
        
def test_list_C(get_testing_data):
    
    act_list, exp_list, g_factor = get_testing_data(list_C, plate_map_file, 'list', 384, list_C_test)
    assert g_factor == exp_g['list_C']
    for act_df, exp_df in zip(act_list, exp_list):
        pd.testing.assert_frame_equal(act_df, exp_df)    
           

def test_plate_size_error():
    """Test for error raised by read_in_envision function if the 'size' parameter passed by the user is not either 384 or 96."""
    with pytest.raises(PlateSizeError):
        test_obj = FA.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='plate', size=100)


def test_incorrect_data_type_list():
    """Test for error raised during data read in by read_in_envision function if data_type = list but raw data file is in plate format."""
    with pytest.raises(DataError):
        test_obj = FA.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='list', size=384)
    
    
def test_incorrect_data_type_plate():
    """Test for error raised during data read in by read_in_envision function if data_type = plate but raw data file is in list format."""
    with pytest.raises(DataError):
        test_obj = FA.read_in_envision(data_csv=list_A, platemap_csv=plate_map_file, data_type='plate', size=384)
    
    
def test_incorrect_data_type():
    """Test for error raised by read_in_envision function if the 'data_type' argument passed by the user is neither plate nor list."""
    with pytest.raises(DataTypeError):
        test_obj = FA.read_in_envision(data_csv=list_A, platemap_csv=plate_map_file, data_type='typo', size=384)

"""    
def test_invalidate():
    #Test whether the invalidate function sets the 'Valid' column as False for the specified well ids, columns and rows.
    
    test_obj = FA.read_in_envision(plate_2_repeat, plate_map_file, 'plate', 384)   # read in actual data and plate map
    test_obj.invalidate(wells=['A2', 'B3', 'E4'], rows=['C', 'G'], columns=[7,8,12,20])   # invalidate specific well ids, rows and columns
    exp_platemap = pd.read_csv(inval_platemap, index_col=[0])   # read in an example platemap with invalidated well ids, rows and columns
    act_platemap = test_obj.plate_map
    
    pd.testing.assert_frame_equal(act_platemap, exp_platemap, check_dtype=False)   # compare the two dfs without checking the data types because the example df contains 'object' data type in all columns
"""

def test_invalidate_error():
    """Test whether the 'invalidate' function raises an error if no arguments are passed."""
    with pytest.raises(TypeError):
        test_obj = FA.read_in_envision(plate_2_repeat, plate_map_file, 'plate', 384)
        test_obj.invalidate()   # execute the invalidate function without specifying well ids, rows or columns to be invalidated


########################################## PROTEIN/TRACER TITRATION TESTS ############################################

test_obj = FA.read_in_envision(prot_trac_data, prot_trac_platemap, 'plate', 384)
 
def test_final_fit_df():
    """Test whether final fit data frame was constructed properly
    """
    exp_ff = pd.read_csv(prot_trac_empty_ff, index_col=[0,1,2])
    act_ff = test_obj.final_fit
    pd.testing.assert_frame_equal(exp_ff, act_ff, check_dtype=False)
 
"""
def test_platemap_df():
    #Test whether the platemap data frame was constructed properly
    
    exp_platemap_df = pd.read_csv(prot_trac_platemap_df, index_col=0)
    act_platemap_df = test_obj.plate_map
    pd.testing.assert_frame_equal(act_platemap_df, exp_platemap_df, check_dtype=False)
"""

def test_no_backg_subt():
    """Test for an error raised if the calc_r_i function is called with 'correct=True' but background subtraction has not been performed
    """
    with pytest.raises(AttributeError):
        test_obj.calc_r_i(correct=True, plot_i=False)    # try executing the tested function without prior execution of the background_correct() funciton
    

def test_background_correct():
    """Tests whether the background correction function performs correct calculations to get the background corrected values 
    of p and s channel signal in the protein-tracer data.
    """
    with open(prot_trac_p_s_correct, 'rb') as file:  #  load the list with expexcted data frames from .pkl file
        exp_list = pickle.load(file)
    
    test_obj.background_correct()
    exp_p, exp_s = exp_list[0], exp_list[1]
    act_p = test_obj.data_dict['repeat_1']['data']['p_corrected']
    act_s = test_obj.data_dict['repeat_1']['data']['s_corrected']

    # assert the p_corrected and s_corrected values are the same as the reference data up to six decimal points 
    pd.testing.assert_frame_equal(act_p, exp_p, atol=1E-6)
    pd.testing.assert_frame_equal(act_s, exp_s, atol=1E-6)

    
def test_calc_r_i():
    """Tests whether the calc_r_i function performs correct calculations to get the raw and background 
    corrected values of intensity and anisotropy
    """
    with open(prot_trac_r_i, 'rb') as file:   # load the list with expexcted data frames from .pkl file
        exp_list = pickle.load(file)
    
    exp_iraw, exp_rraw, exp_icorr, exp_rcorr, exp_iperc = exp_list[0], exp_list[1], exp_list[2], exp_list[3], exp_list[4]
    
    test_obj.calc_r_i(correct=True, plot_i=False)
    act_iraw = test_obj.data_dict['repeat_1']['data']['i_raw']
    act_rraw = test_obj.data_dict['repeat_1']['data']['r_raw']
    act_icorr = test_obj.data_dict['repeat_1']['data']['i_corrected']
    act_rcorr = test_obj.data_dict['repeat_1']['data']['r_corrected']
    act_iperc = test_obj.data_dict['repeat_1']['data']['i_percent']
    
    # assert the calcualted anisotropy and intensity values are the same as the reference data up to six decimal points
    pd.testing.assert_frame_equal(act_iraw, exp_iraw, atol=1E-6)
    pd.testing.assert_frame_equal(act_rraw, exp_rraw, atol=1E-6)
    pd.testing.assert_frame_equal(act_icorr, exp_icorr, atol=1E-6)
    pd.testing.assert_frame_equal(act_rcorr, exp_rcorr, atol=1E-6)
    pd.testing.assert_frame_equal(act_iperc, exp_iperc, atol=1E-6)
    

def test_calc_mean_r_i():
    """Test whether the calc_mean_r_i function performs correct calculations of mean anisotropy and intesity and whether the 
    empty fit_parms data frame is construced correctly.
    """
    with open(prot_trac_mean_r_i, 'rb') as file:
        exp_r_dict, exp_i_dict, exp_fit_params = pickle.load(file)   # read in a tuple with example r and i dictionaries and fit params data frame
    
    test_obj.calc_mean_r_i()

    act_fit_params = test_obj.data_dict['repeat_1']['data']['fit_params']
    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params)   # test the fit params data frame
    
    act_r_dict = test_obj.data_dict['repeat_1']['data']['r_mean']
    act_i_dict = test_obj.data_dict['repeat_1']['data']['i_mean']
    
    for ptc in exp_r_dict.keys():   # for each protein-tracer pair compare the actual r mean df (and i mean df) to the example ones
        pd.testing.assert_frame_equal(act_r_dict[ptc], exp_r_dict[ptc])
        pd.testing.assert_frame_equal(act_i_dict[ptc], exp_i_dict[ptc])


def test_logistic_fit():
    """Test whether the logisitc_fit function correctly fits a curve to the aniostropy and intensity data (the resulting fit_params df is tested).
    """
    exp_fit_params = pd.read_csv(prot_trac_log_fit_params, index_col=[0,1,2])
    
    test_obj.logistic_fit()
    test_obj.logistic_fit(prot=['Protein 1'], trac=['Tracer'], var='i', rep=[1], sigma='std', p0=[500000, 300000, 1300, 2])
    act_fit_params = test_obj.data_dict['repeat_1']['data']['fit_params']

    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params, check_dtype=False) 
         

def test_calc_amount_bound():
    """Test whetehr the calc_amount bound function performs correct calculation of the amount bound for the protein titration data.
    """
    with open(prot_trac_amount_b, 'rb') as file:
        exp_ab_dict = pickle.load(file)    # load a dictionary with expample data frames from pkl file
    
    # update the final_fit df with rmin and rmax values because calc_lambda is not tested
    cols = ['rmin','rmin error','rmax','rmax error']
    params = test_obj.data_dict['repeat_1']['data']['fit_params'].loc[:, cols]
    test_obj.final_fit.loc[:, cols] = params

    test_obj.calc_amount_bound()                                                                   
    
    for ptc in exp_ab_dict.keys():   # for each protein-tracer pair compare the amount bound dfs to the example ones
        act_df = test_obj.data_dict['repeat_1']['data']['amount_bound'][ptc]
        exp_df = exp_ab_dict[ptc]
        pd.testing.assert_frame_equal(act_df, exp_df)


def test_single_site_fit():
    """Test whether the single_site_fit function correctly fits a curve to the amount bound data (the resulting final_fit df is tested).
    """
    exp_fit_params = pd.read_csv(prot_trac_ss_final_fit, index_col=[0,1,2])
    test_obj.single_site_fit()
    act_fit_params = test_obj.final_fit
    
    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params, check_dtype=False)  


############################################### COMPETITION TESTS #######################################################

test_obj_com = FA.read_in_envision(comp_data, comp_platemap, 'plate', 384)   

def test_final_fit_df_com():
    """Test whether final fit data frame was constructed properly for competition data set"""
    exp_com_ff = pd.read_csv(comp_empty_ff_df, index_col=[0,1,2])
    act_com_ff = test_obj_com.final_fit
    pd.testing.assert_frame_equal(act_com_ff, exp_com_ff, check_dtype=False)

"""
def test_platemap_df_com():
    #Test whether the platemap data frame was constructed properly
    exp_platemap_df = pd.read_csv(comp_platemap_df, index_col=0)
    act_platemap_df = test_obj_com.plate_map
    pd.testing.assert_frame_equal(act_platemap_df, exp_platemap_df, check_dtype=False)
"""

def test_background_correct_com():
    """Tests whether the background correction function performs correct calculations to get the background corrected values 
    of p and s channel signal in the competition data set.
    """
    with open(comp_p_s_correct, 'rb') as file:  #  load the list with expexcted data frames from .pkl file
        exp_list = pickle.load(file)
    
    test_obj_com.background_correct()
    exp_p, exp_s = exp_list[0], exp_list[1]
    act_p = test_obj_com.data_dict['repeat_1']['data']['p_corrected']
    act_s = test_obj_com.data_dict['repeat_1']['data']['s_corrected']
    
    pd.testing.assert_frame_equal(act_p, exp_p, atol=1E-6)
    pd.testing.assert_frame_equal(act_s, exp_s, atol=1E-6)


def test_calc_r_i_com():
    """Tests whether the calc_r_i function performs correct calculations to get the raw and background 
    corrected values of intensity and anisotropy for competition data.
    """
    with open(comp_r_i, 'rb') as file:   # load the list with expexcted data frames from .pkl file
        exp_list = pickle.load(file)
    
    exp_rcorr, exp_icorr, exp_rraw, exp_iraw, exp_iperc = exp_list[0], exp_list[1], exp_list[2], exp_list[3], exp_list[4]

    test_obj_com.calc_r_i(correct=True, plot_i=False)
    act_iraw = test_obj_com.data_dict['repeat_1']['data']['i_raw']
    act_rraw = test_obj_com.data_dict['repeat_1']['data']['r_raw']
    act_icorr = test_obj_com.data_dict['repeat_1']['data']['i_corrected']
    act_rcorr = test_obj_com.data_dict['repeat_1']['data']['r_corrected']
    act_iperc = test_obj_com.data_dict['repeat_1']['data']['i_percent']

    # assert the calcualted anisotropy and intensity values are the same as the reference data up to six decimal points
    pd.testing.assert_frame_equal(act_rcorr, exp_rcorr, atol=1E-6)
    pd.testing.assert_frame_equal(act_icorr, exp_icorr, atol=1E-6)
    pd.testing.assert_frame_equal(act_rraw, exp_rraw, atol=1E-6)
    pd.testing.assert_frame_equal(act_iraw, exp_iraw, atol=1E-6)
    pd.testing.assert_frame_equal(act_iperc, exp_iperc, atol=1E-6)


def test_calc_mean_r_i_com():
    """Test whether the calc_mean_r_i function performs correct calculations of mean anisotropy and intesity and whether the 
    empty fit_parms and fit_paramas_com data frames are construced correctly for competiton data.
    """
    # read in a tuple with example r and i dictionaries and fit params data frame
    with open(comp_mean_r_i, 'rb') as file:
        exp_r_dict, exp_i_dict, exp_fit_params, exp_fit_params_com = pickle.load(file)

    test_obj_com.calc_mean_r_i()
    act_fit_params = test_obj_com.data_dict['repeat_1']['data']['fit_params']
    act_fit_params_com = test_obj_com.data_dict['repeat_1']['data']['fit_params_com']
    act_r_dict = test_obj_com.data_dict['repeat_1']['data']['r_mean']
    act_i_dict = test_obj_com.data_dict['repeat_1']['data']['i_mean']

    # test the fit params data frames
    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params)
    pd.testing.assert_frame_equal(act_fit_params_com, exp_fit_params_com)

    for ptc in exp_r_dict.keys():   # for each protein-tracer pair compare the actual r mean df (and i mean df) to the example ones
        pd.testing.assert_frame_equal(act_r_dict[ptc], exp_r_dict[ptc])
        pd.testing.assert_frame_equal(act_i_dict[ptc], exp_i_dict[ptc])


def test_logistic_fit_com():
    """Test whether the logisitc_fit function correctly fits a curve to the aniostropy and intensity data (the resulting fit_params df is tested).
    """
    test_obj_com.logisitc_fit_com()
    act_fit_params = test_obj_com.data_dict['repeat_1']['data']['fit_params']
    exp_fit_params = pd.read_csv(com_log_fit_params, index_col=[0,1,2])

    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params, check_dtype=False)


def test_calc_amount_bound_com():
    """Test whetehr the calc_amount bound function performs correct calculation of the amount bound for the protein titration data.
    """
    test_obj_com.import_params(final_fit_params_for_com)   # import the rmin and rmax params from a csv file
    test_obj_com.calc_amount_bound()

    with open(com_amount_b, 'rb') as file:
        exp_ab_dict = pickle.load(file)    # load a dictionary with example data frames                                                             
    
    for key in exp_ab_dict.keys():   # for each protein-tracer pair compare the amount bound dfs to the example ones
        act_df = test_obj_com.data_dict['repeat_1']['data']['amount_bound'][key]
        exp_df = exp_ab_dict[key]
        pd.testing.assert_frame_equal(act_df, exp_df)


def test_single_site_fit_com():
    """Test whether the single_site_fit function correctly fits a curve to the amount bound data 
    (the resulting final_fit and fit_params_com data frames are tested).
    """
    test_obj_com.single_site_fit_com()
    
    act_final_fit = test_obj_com.final_fit
    act_fit_params = test_obj_com.data_dict['repeat_1']['data']['fit_params_com']
    exp_final_fit = pd.read_csv(com_ss_final_fit, index_col=[0,1,2])
    exp_fit_params = pd.read_csv(com_ss_fit_params, index_col=[0,1,2])

    pd.testing.assert_frame_equal(act_final_fit, exp_final_fit, check_dtype=False)
    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params, check_dtype=False)
    