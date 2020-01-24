"""
    Author: Asma G
"""

from my_constants import MyConstants
import numpy as np
import dimensionality_reduction

all_df, x_df, y_df = dimensionality_reduction.preprocess_survey_x_y()
x_df_nonan = x_df.fillna(0)
reduced_x_df_standardized, reduced_x_df, reduced_n, = dimensionality_reduction.reduce_dimensionality(x_df_nonan, max_n=50, threshold=MyConstants.EXPLAINED_VARIANCE_THRESHOLD)

# y = np.array(y_df['HAMD'])
# print (np.shape(y))

x1 = np.array(reduced_x_df[['PCA_'+str(i) for i in range(reduced_n)]])
x2 = np.array(reduced_x_df[['KernelPCA_'+str(i) for i in range(reduced_n)]])
x3 = np.array(reduced_x_df[['TruncatedSVD_'+str(i) for i in range(reduced_n)]])



dimensionality_reduction.plot_reduced_feature(x1, 'PCA')
dimensionality_reduction.plot_reduced_feature(x2, 'Kernel PCA')
dimensionality_reduction.plot_reduced_feature(x3, 'Truncated SVD')