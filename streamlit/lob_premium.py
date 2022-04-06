import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns



def main():
    st.title('Observation of Premium LOB changes')

    lobs_map = {'Life': '_plob_life', 'Household': '_plob_household',
                'Motor':'_plob_motor', 'Health':'_plob_health',
                'Work Compensation':'_plob_wcomp'}

    var_map = {'Ratio LOB Premium to Total': 'rt',
                'True Premium':'amt'}

    var_map_descriptive = {'Ratio LOB Premium to Total': 'Ratio',
                            'True Premium':'Premium'}

    lobs_list = ['Life', 'Household', 'Motor','Health', 'Work Compensation']

    var_map_splits = {'Above-Below': 'SPLIT_FEATURE',
                'Classes':'fe_bin_plob_motor'}
    
    @st.cache(persist = True, allow_output_mutation=True)
    def load_data():
        data = pd.read_csv('data/bk1_dataset_A.csv')
        return data
    

    def split_dataset_lobs(dataset, feature, threshold):
        dataset = dataset.copy()
        dataset['SPLIT_FEATURE'] = np.where(dataset[feature] >= threshold, 'Above','Below')
        return dataset
    
    dfInsurance = load_data()

    SHOW_DATAFRAME = st.checkbox('Show Dataframe examples')
    if SHOW_DATAFRAME:
        st.dataframe(dfInsurance.head(5))

    VAR_LOB = st.selectbox(
                    'Line of Business Selected',
                    ('Life', 'Household', 'Motor','Health', 'Work Compensation'))
    RT_THRESHOLD = st.slider(f'Threshold for LOB "{VAR_LOB}" premium in relation to total premium', 0.0, 1.0, 0.3)
    VAR_TO_VIEW = st.selectbox(
                    'Type of Variable to analyse',
                    ('Ratio LOB Premium to Total', 'True Premium'))

    SPLIT_FEATURE = st.selectbox(
                    'Split Feature',
                    ('Above-Below', 'Classes'))

    SHOW_PLOTS = st.checkbox('Show Plots')


    FILTER_FEATURE = f'rt{lobs_map[VAR_LOB]}'
    VIEW_FEATURE = f'{var_map[VAR_TO_VIEW]}{lobs_map[VAR_LOB]}'


    if SHOW_PLOTS:
        dfSplitted = split_dataset_lobs(dfInsurance, FILTER_FEATURE, RT_THRESHOLD)
        SHOW_DATAFRAMES = st.checkbox('Show Dataframes examples')
        if SHOW_DATAFRAMES:
            st.dataframe(dfSplitted.loc[dfSplitted['SPLIT_FEATURE'] == 'Above'].head(5))
            st.dataframe(dfSplitted.loc[dfSplitted['SPLIT_FEATURE'] == 'Below'].head(5))
        
        
        ## Plot Total Premium Distribution on Above and Below Datasets
        st.subheader(f'Plot Total Premium')
        plot_ttp = sns.FacetGrid(dfSplitted, col=var_map_splits[SPLIT_FEATURE], col_wrap=2)
        plot_ttp.map(sns.histplot, 'amt_premium_total')
        for ax, lbl in zip(plot_ttp.axes.flatten(), [0, 1]):
            # print(ax, lbl)
            ax.set_xlabel('Total Premium Amount')
        st.pyplot(plot_ttp)

        ## Selected Feature histogram
        st.subheader(f'{VAR_LOB}:{RT_THRESHOLD}, {var_map_descriptive[VAR_TO_VIEW]} of {VAR_LOB}')
        plot_sfh = sns.FacetGrid(dfSplitted, col=var_map_splits[SPLIT_FEATURE], col_wrap=2)
        plot_sfh.map(sns.histplot, VIEW_FEATURE)
        for ax, lbl in zip(plot_sfh.axes.flatten(), [0, 1]):
            # print(ax, lbl)
            ax.set_xlabel(f'LOB: {VAR_LOB} Premium')
        st.pyplot(plot_sfh)

        ## Cycle Histogram Plot of other lobs
        plot_vars = [lob for lob in lobs_list if lob != VAR_LOB]
        for lob_plot in plot_vars:
            st.subheader(f'{VAR_LOB}:{RT_THRESHOLD}, {var_map_descriptive[VAR_TO_VIEW]} of {lob_plot}')
            plot_pol = sns.FacetGrid(dfSplitted, col=var_map_splits[SPLIT_FEATURE], col_wrap=2)
            plot_pol.map(sns.histplot, f'{var_map[VAR_TO_VIEW]}{lobs_map[lob_plot]}')
            for ax, lbl in zip(plot_pol.axes.flatten(), [0, 1]):
                # print(ax, lbl)
                ax.set_xlabel(f'LOB: {lob_plot} Premium')
            st.pyplot(plot_pol)

if __name__ == '__main__':
    main()