# Refining Diagnostic Indicators: Correlation Analysis of Overlapping Metrics in Breast Cancer Cell Nuclei

## Purpose of the Analysis

This analysis is designed to identify the most influential nuclear features extracted from digitized FNA images that differentiate between malignant and benign breast masses. By focusing on overlapping metric ranges, the study seeks to pinpoint the key indicators most strongly correlated with the diagnosis, thereby enhancing the precision of computer-aided diagnostic methods.

## Data Sources

The analysis utilizes the Wisconsin Breast Cancer Dataset which contains features computed from digitized images of fine needle aspirate (FNA) of a breast mass.

_Original dataset can be found here_:

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

Attribute Information:

- ID number

- Diagnosis (M = malignant, B = benign)

3-32)

Ten real-valued features are computed for each cell nucleus:

- radius (mean of distances from center to points on the perimeter)

- texture (standard deviation of gray-scale values)

- perimeter

- area

- smoothness (local variation in radius lengths)

- compactness (perimeter^2 / area - 1.0)

- concavity (severity of concave portions of the contour)

- concave points (number of concave portions of the contour)

- symmetry

- fractal dimension ("coastline approximation" - 1)

The mean, standard error and "worst" or largest (mean of the three

largest values) of these features were computed for each image,

resulting in 30 features. For instance, field 3 is Mean Radius, field

13 is Radius SE, field 23 is Worst Radius.

All feature values are recoded with four significant digits.

Missing attribute values: none

Class distribution: 357 benign, 212 malignant

## Methodology

The analysis began with an exploratory data analysis (EDA) of 30 continuous numeric features. A correlation matrix was computed to identify variables with high inter-correlation. For instance, metrics such as radius, area, and perimeter exhibited correlations above 0.94, prompting a feature selection process where only the radius was retained from this mathematically linked group.

Subsequently, features were organized into groups based on their measurement types (mean, standard error, and worst values), resulting in grouped variables such as radius_mean, radius_se, and radius_worst. To capture the dynamic relationship between these measures, a new metric was defined—radius_diff, calculated as the difference between the worst and mean values.

The dataset was then split into malignant and benign categories. Histograms were generated to visualize the distribution and overlapping regions of the two groups. A focused histogram on the overlapping region was used to compute the probability distribution for malignant cases. Pearson correlation was then calculated between these metrics and the diagnosis label (encoded as malignant = 1 and benign = 0), and the metric with the highest correlation was identified.

Finally, for each feature group, chi-square tests and Cramér's V were computed. Only those metrics exhibiting a Cramér's V value above 0.5 and a chi-square p-value below 0.05 were deemed statistically significant and selected as the four “winner” metrics for the final analysis summary.

## Implementation & Statistical Analysis Tools

The analysis was implemented in Python, leveraging several powerful libraries to facilitate data ingestion, manipulation, visualization, and statistical testing:

**Data Access & Manipulation:**

kagglehub was used to seamlessly import datasets from Kaggle.

pandas provided robust DataFrame functionality for data cleaning, transformation, and managing panel data.

**Visualization:**

matplotlib.pyplot served as the foundation for plotting histograms, correlation matrices, and other visual insights.

seaborn enhanced these visualizations with high-definition aesthetics, ensuring clarity when displaying distributions and overlapping regions between malignant and benign cases.

_Numerical Operations:_

numpy was utilized for efficient numerical computations, enabling the handling of large arrays and supporting vectorized operations.

**Statistical Testing:**

scipy.stats.chi2_contingency was employed to perform chi-square tests, evaluating the significance of relationships between categorical features and the diagnosis.

scipy.stats.contingency.association facilitated the computation of Cramér's V, quantifying the strength of associations identified in the chi-square analysis.

scipy.stats.pearsonr computed Pearson correlation coefficients, which were crucial in measuring the linear relationship between the newly derived features (e.g., the difference between worst and mean values) and the diagnosis outcome.

## How to Run the Project

Clone the Repository:

- Open your teminal or command prompt and run:

```

git clone https://github.com/v4nui/breast_cancer_analysis

cd breast-cancer-analysis

```

- Launch Jupyter Notebook by running:

```

jupyter notebook

```

This will open a web interface in your default browser. Navigate to the notebook file ( `bc_analysis.ipynb `) and open it.
For group specific analysis pick the `bc_<group name>_analysis.ipynb ` file and open it.

- Execute the Notebook:

It is recommended to run the cells sequentially from top to bottom to ensure all dependencies and variables are properly defined.

Follow any in-notebook instructions or comments that guide you through the analysis steps.

## Results & Discussion

Analysis identified four metrics with the strongest correlation to the diagnosis: compactness_mean, concave_points_worst, concavity_mean, and radius_worst. These metrics were selected based on their extremely low chi-square p-values and high Cramér's V values, indicating a robust association with the diagnostic outcome. The chi-square p-values for these metrics were:

compactness_mean: 3.77e-32

concave_points_worst: 1.24e-39

concavity_mean: 2.02e-48

radius_worst: 1.39e-26

Additionally, the corresponding Cramér's V values were:

compactness_mean: 0.58

concave_points_worst: 0.71

concavity_mean: 0.74

radius_worst: 0.68

These findings were visually reinforced by a bar chart, which can be seen in the bc_global Jupyter Notebook. Below is the code snippet used to generate the bar chart for the top four metrics:

python

```

metrics = ['compactness_mean', 'concave_points_worst', 'concavity_mean', 'radius_worst']

chi_values = [3.7694541006106594e-32, 1.235542576364234e-39, 2.022185081770995e-48, 1.394378336723893e-26]

cramer_values = [0.5813766795517757, 0.7119599662792855, 0.7374721424144816, 0.679853961429309]



df_chi = pd.DataFrame(chi_values, index=metrics, columns=['chi_values'])

df_cramer = pd.DataFrame(cramer_values, index=metrics, columns=['cramer_values'])

print(df_chi)



x = np.arange(len(metrics))

width = 0.5 # width of the bars



plt.figure(figsize=(8, 6))

plt.bar(x, cramer_values, width, color='blue')

plt.xticks(x, metrics)

plt.ylabel('Cramer V Values')

plt.title('Cramer V Values for Top 4 Metrics')



# Adding value labels on top of bars



for i, value in enumerate(cramer_values):

plt.text(x[i] - width/4, value + 0.02, f'{value:.2f}', fontweight='bold')

plt.show()

```

### Conclusion:

Based on the statistical tests and visualizations, the metrics compactness_mean, concave_points_worst, concavity_mean, and radius_worst have been determined as key indicators in distinguishing between malignant and benign cases. Their strong associations suggest that they are effective predictors and have been incorporated into the final predictive model summary.

## Content of the repository:

```

├── cleaned_data.csv # processed data file

├── bc\_<name of the metric>\_analysis # notebooks where you can find targeted group analysis per name of the metric

├── bc_global # notebook file where I did the initial spets of the analysis and the summary after analysisng each individual group

├── plots # folder with all generaled plot images(histograms, heatmaps, boxplots)

├── README.md # This readme file

├── correlation_analysis_breast_cancer.pd # presentation file

└── requirements.txt # Python dependencies

```

## Contributions & Acknowledgments

**Contributions:**

Contributions to this project are welcome! If you have suggestions, improvements, or additional analyses, please feel free to open an issue or submit a pull request. Your feedback and contributions will help enhance the project's quality and reproducibility.

**Acknowledgments:**

The excellent documentation and tools provided by Pandas, Matplotlib, and Seaborn were instrumental in the data manipulation and visualization processes throughout this project.

I am grateful for the valuable learning resources provided by Ironhack, which have significantly contributed to my understanding and implementation of the project methodologies.

OpenAI's ChatGPT was used as a supplementary tool to refine the language; however, all core analyses, interpretations, and conclusions are solely my own.
