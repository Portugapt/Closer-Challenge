import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns



def main():
    st.title('Observation of Premium LOB changes')

    @st.cache(persist = True)
    def load_data():
        data = pd.read_csv('data/bk1_dataset_A.csv')

        data = 
        return data
    
    dfInsurance = load_data()

    VAR_LOB = st.selectbox(
                    'Line of Business Selected',
                    ('Life', 'Household', 'Motor','Health', 'Work Compensation'))
    RT_THRESHOLD = st.slider(f'Threshold for LOB "{VAR_LOB}" premium in relation to total premium', 0.0, 1.0, 0.3)



if __name__ == '__main__':
    main()