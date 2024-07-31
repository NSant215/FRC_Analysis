# Quantification of Free Residual Chlorine for Water Purification

## Introduction
This repository is to present the results of my Masters Thesis analysing the quantification of Free Residual Chlorine (FRC) for purposes in water purification methods.

## Repository Details
All data used for analysis in this project is within the /Data folder. Naming conventions in the folder are in the subdirectory README. All analysis is done in the python notebooks, using helper functions for cleaning and presenting the data in data.py, and analysis helper functions in regressions.py. The final thesis is shown in the PDF "Quantification_of_Free_Residual_Chlorine.pdf"

## Project Inspirations and Technical Abstract

The subject of water quality is extremely important, as access to clean water is essential for human survival. Chlorination is a common method of disinfection, but dosing water samples with appropriate amounts of chlorine is an issue; too little can leave the water still infected, but too much can make the water unpotable. Measuring the Free Residual Chlorine (FRC) of a sample identifies whether a sample has been dosed appropriately, but current methods of measuring FRC are expensive and have limitations.

The aim for this project was to establish whether it is feasible to use the measurement of common water parameters to quantify the FRC of drinking water samples that have been treated with chlorine. The purpose of this was to explore possible ways of developing a low-cost FRC sensor to use in low-income areas.

The first stage of this project concerned the collection of data, in order to build a dataset that could be used for analysis. This involved the development of a method to produce predictable chlorine doses and consistent measurements across sensors. This meant we could be confident in using the method for building the dataset. Samples over a range of water sources were gathered and measured, aiming to get a general representation of waters that might be analysed in the field.

The parameters that were analysed for use in this project were water conductivity, pH, dissolved oxygen concentration, turbidity and Oxidation-Reduction Potential (ORP). The measurement of the ORP presented unexpected results. One was that the ORP probe’s reading had a strong dependency on its recent history (in the previous hour). The meas- urement also had long stabilisation times (longer than 5 minutes). This led to developing a technique for taking measurements with the ORP probe involving long periods between measurements, and measuring samples of similar compositions together, to mitigate these effects. This was also an indicator that this measurement might be less reliable to take in the field.

Each parameter measured was evaluated on whether it would be useful for predicting the FRC. This was partly done by looking at the correlation coefficient (R2) between the parameter and the measured FRC for each water source. R2 gives a number between 0 and 1, and higher magnitudes show stronger correlation, so it was used as a ‘score’ for the correlations. On looking at the direct relationships between each measured parameter, ORP provided the most valuable information, as it held a relationship with FRC that was reasonably independent of the source. The conductivity measurements had some correlation with the FRC (on average, R2 = 0.506 for each source), but were very source- dependent. This suggested they would be possibly useful measurements for determining FRC, but less important than the ORP. Both dissolved oxygen and turbidity had little to no direct correlation with FRC, and dissolved oxygen also had a very small range of variation. This suggested that these measurements would be less useful than the others mentioned, or useless, when trying to predict FRC.

The second stage of this project involved using different data analysis techniques to find mappings from the measured parameters to a continuous quantification of FRC, or to form classifiers that could classify samples into having too little, enough, or too much chlorine. Scores of 0.75 for correlation, and 0.8 for classification, were decided as thresholds for success on trials conducted. These values were chosen to show that a method had significant viability in being assessed further for practical application.

For classification, the score was the proportion of correctly classified points. Classification was done in two approaches, both by splitting the total FRC range into ranges for each class. The first approach was a 3-class split, with the classes being ‘0 → 0.5 FRC’ (not enough chlorine), ‘0.5 → 2 FRC’ (enough chlorine), and ‘> 2 FRC’ (too much chlorine). The second approach was a 2-class split with classes of greater than or less than 0.5 FRC. This had the purpose of being able to label whether a sample had enough chlorine or not.

Linear Discriminant Analysis (LDA) performed best at finding a relationship between other common water parameters that correlated to the FRC. It obtained a correlation of 0.703, which was lower than the defined threshold, suggesting that it was not possible to find a continuous quantitative measure of FRC. Quadratic Discriminant Analysis (QDA) performed best at classifying points, and both scores, especially the classification score on the 2-class split (deciding whether a sample’s FRC was above or below 0.5), were very promising. The 3-class score was 0.777, which was close to, but lower than, the desired threshold of 0.8. The 2-class score of 0.891 was very strong, and well above the desired threshold. This means the analysis correctly classified 89.1% of the test datapoints, which is strong evidence for the use of this method in the future. All the analysis techniques showed turbidity and dissolved oxygen to have a negligible impact, as they both reduced the correlation and classification scores when they were included. This suggested they are not useful for determining FRC.

In conclusion, this project shows that there is scope for using the measurements of other water parameters to be able to classify water samples into having enough chlorine or not, but that the complexity of the problem is too high for providing a continuous quantitative measure of FRC. With a much larger dataset, and exploring non-linear data analysis techniques, there is possibility for finer-grain classification. However, this outcome is highly dependent on reliable measurements of ORP, as the data analysis showed high dependency on this measurement. If it is too difficult to get reliable measurement of ORP in the field, the outcome of this project suggests it would not be possible to use measurements of other water parameters to determine FRC.


