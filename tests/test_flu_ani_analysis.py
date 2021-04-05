import os
import pytest
import pickle
import pandas as pd
from matplotlib.testing.compare import compare_images    
from fluanisotropyanalysis.flu_ani_analysis import FA 

# platemap file for read_in_envision and invalidated tests
plate_map_file = "tests/test_data/plate_map_for_read_in_env_&_inval.csv"

# raw data files in .csv format for read_in_envision tests
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

# files containing lists of expected data frames in .pkl format for read_in_envision tests
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

prot_trac_data = "tests/test_data/protein_tracer_data_set.csv"
prot_trac_platemap = "tests/test_data/protein_tracer_platemap.csv"
prot_trac_p_s_correct = "tests/example_output/background_correct/protein-tracer_p_s_correct.pkl"
prot_trac_r_i = "tests/example_output/calc_r_i/protein-tracer_r_i_i_percent.pkl"
prot_trac_mean_r_i = "tests/example_output/calc_mean_r_i/protein-tracer_mean_r_i_dicts.pkl"
prot_trac_log_fit_params = "tests/example_output/logistic_fit/protein-tracer_log_fit_params.csv"
prot_trac_amount_b = "tests/example_output/calc_amount_bound/protein-tracer_mean_ab_dict.pkl"
prot_trac_ss_final_fit = "tests/example_output/single_site_fit/protein-tracer_final_fit_params.csv"


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
    for act_df, exp_df in zip(act_list, exp_list):   # zip the two lists containing actual and expected data frames into a list of tuples and iterate over all tuples
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
      

def test_final_fit_df():
    
    with open(ff_df, 'rb') as file:   # load the final_fit data frame from .pkl file
        exp_ff = pickle.load(file)
    
    test_obj = FA.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='plate', size=384)
    act_ff = test_obj.final_fit
    pd.testing.assert_frame_equal(act_ff, exp_ff)
        
#@pytest.mark.raises()
def test_plate_size_error():
    """Test for error raised by read_in_envision function if the 'size' parameter passed by the user is not either 384 or 96."""
    with pytest.raises(PlateSizeError):
        test_obj = FA.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='plate', size=100)

@pytest.mark.raises()
def test_incorrect_data_type_list():
    """Test for error raised during data read in by read_in_envision function if data_type = list but raw data file is in plate format."""
    
    test_obj = FA.read_in_envision(data_csv=plate_1, platemap_csv=plate_map_file, data_type='list', size=384)
    
@pytest.mark.raises()
def test_incorrect_data_type_plate():
    """Test for error raised during data read in by read_in_envision function if data_type = plate but raw data file is in list format."""
        
    test_obj = FA.read_in_envision(data_csv=list_A, platemap_csv=plate_map_file, data_type='plate', size=384)
     
@pytest.mark.raises()
def test_incorrect_data_type():
    """Test for error raised by read_in_envision function if the 'data_type' argument passed by the user is neither plate nor list."""
        
    test_obj = FA.read_in_envision(data_csv=list_A, platemap_csv=plate_map_file, data_type='typo', size=384)

@pytest.mark.raises()
def test_invalidate_error():
    """Test whether the 'invalidate' function raises an error if no arguments are passed."""
    
    test_obj = FA.read_in_envision(plate_2_repeat, plate_map_file, 'plate', 384)
    test_obj.invalidate()   # execute the invalidate function without specifying well ids, rows or columns to be invalidated


test_obj = FA.read_in_envision(prot_trac_data, prot_trac_platemap, 'plate', 384)
    
@pytest.mark.raises()
def test_no_backg_subt():
    """Test for an error raised if the calc_r_i function is called with 'correct=True' but background subtraction has not been performed"""
    
    # try executing the tested function without prior execution of the background_correct() funciton
    test_obj.calc_r_i(correct=True, plot_i=False)    
    
    
def test_background_correct():
    """Tests whether the background correction function performs correct calculations to get the background corrected values 
    of p and s channel signal
    """
    test_obj.background_correct()
    with open(prot_trac_p_s_correct, 'rb') as file:  #  load the list with expexcted data frames from .pkl file
        exp_list = pickle.load(file)

    # assert the p_corrected and s_corrected values are the same as the reference data up to six decimal points 
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['p_corrected'], exp_list[0], atol=1E-6)
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['s_corrected'], exp_list[1], atol=1E-6)

    
def test_calc_r_i():
    """Tests whether the calculate_r_i function performs correct calculations to get the raw and background 
    corrected values of intensity and anisotropy
    """
    test_obj.calc_r_i(correct=True, plot_i=False)
    with open(prot_trac_r_i, 'rb') as file:   # load the list with expexcted data frames from .pkl file
        exp_list = pickle.load(file)
 
    # assert the calcualted anisotropy and intensity values are the same as the reference data up to six decimal points
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['i_raw'], exp_list[0], atol=1E-6)
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['r_raw'], exp_list[1], atol=1E-6)
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['i_corrected'], exp_list[2], atol=1E-6)
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['r_corrected'], exp_list[3], atol=1E-6)
    pd.testing.assert_frame_equal(test_obj.data_dict['repeat_1']['data']['i_percent'], exp_list[4], atol=1E-6)
    

def test_calc_mean_r_i():
    
    test_obj.calc_mean_r_i()
    act_fit_params = test_obj.data_dict['repeat_1']['data']['fit_params']
    
    # read in a tuple with example r and i dictionaries and fit params data frame
    with open(prot_trac_mean_r_i, 'rb') as file:
        exp_r_dict, exp_i_dict, exp_fit_params = pickle.load(file)
        ptc_list = exp_r_dict.keys()#(test_obj.data_dict['repeat_1']['data']['r_mean'].keys())   # list of all protein-tracer pairs
    
    # test the fit params data frame
    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params)
    
    for ptc in ptc_list:   # for each protein-tracer pair compare the actual r mean df (and i mean df) to the example ones
        act_r_dict = test_obj.data_dict['repeat_1']['data']['r_mean']
        act_i_dict = test_obj.data_dict['repeat_1']['data']['i_mean']
        
        pd.testing.assert_frame_equal(act_r_dict[ptc], exp_r_dict[ptc])
        pd.testing.assert_frame_equal(act_i_dict[ptc], exp_i_dict[ptc])


def test_logistic_fit():
    
    # execute the tested function
    test_obj.logistic_fit()
    test_obj.logistic_fit(prot=['Protein 1'], trac=['Tracer'], var='i', rep=[1], sigma='std', p0=[500000, 300000, 1300, 2])
    act_fit_params = test_obj.data_dict['repeat_1']['data']['fit_params']
    exp_fit_params = pd.read_csv(prot_trac_log_fit_params, index_col=[0,1,2])

    pd.testing.assert_frame_equal(act_fit_params, exp_fit_params, check_dtype=False)   # compare the empty fit params df to the example one
         

def test_calc_amount_bound():

    # update the final_fit df with rmin and rmax values because calc_lambda is not tested
    cols = ['rmin','rmin error','rmax','rmax error']
    params = test_obj.data_dict['repeat_1']['data']['fit_params'].loc[:, cols]
    test_obj.final_fit.loc[:, cols] = params

    test_obj.calc_amount_bound()

    with open(prot_trac_amount_b, 'rb') as file:
        exp_ab_dict = pickle.load(file)                                                                     
        ptc_list = exp_ab_dict.keys()
    
    for ptc in ptc_list:   # for each protein-tracer pair compare the amount bound dfs to the example ones
        act_ab_dict = test_obj.data_dict['repeat_1']['data']['amount_bound']
        pd.testing.assert_frame_equal(act_ab_dict[ptc], exp_ab_dict[ptc])
