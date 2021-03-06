import streamlit as st
from run_profile import run_model, load_data
from process_model import load_enron_models, load_wapo_model, load_spooky_model, predict_enron_models, \
    unpickle_enron_dictionary, unpickle_wapo_dictionary, unpickle_spooky_dictionary, predict_wapo_model, \
    predict_spooky_model

output = None
secondary = None

st.session_state.output = ''
st.session_state.secondary = ''

@st.cache(allow_output_mutation=False)
def load_model():
    # enron_model = load_enron_models()
    # wapo_model = load_wapo_model()
    # spooky_model = load_spooky_model()
    # enron_label_dict = unpickle_enron_dictionary()
    # wapo_label_dict = unpickle_wapo_dictionary()
    # spooky_label_dict = unpickle_spooky_dictionary()
    data = load_data()
    # return enron_model, wapo_model, spooky_model, enron_label_dict, wapo_label_dict, spooky_label_dict, data
    return data




def predictModelMethod(input):
    # enron_model, wapo_model, spooky_model, enron_label_dict, wapo_label_dict, spooky_label_dict, data = load_model()
    data = load_model()
    # print('option',  option)
    secondary_Result = run_model(data,input)
    st.session_state.secondary = secondary_Result

if __name__ == '__main__':
    st.title('Characteristics of Author Profiling')
    input  = st.text_area('Input', height=350, value="", max_chars=None)
    # option = st.select_slider('Choose Model', options=['enron', 'wapo', 'spooky'])
    button = st.button('Submit', on_click=predictModelMethod(input), args=None, kwargs=None)
    # st.header('output:')
    # output = st.text(st.session_state.output)
    st.header('secondary results:')
    secondary = st.text(st.session_state.secondary)
