# Data inputs

`Preprocessed/AI4I` and `Preprocessed/PdM` contain the fixed task populations,
observable inputs and offline target/configuration files used by this package.
Each task manifest labels search, validation and test roles explicitly.

AI4I 2020 is a synthetic predictive-maintenance benchmark distributed through
the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)
under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Source citation: *AI4I 2020 Predictive Maintenance Dataset* (2020), UCI Machine
Learning Repository, DOI [10.24432/C5HS5C](https://doi.org/10.24432/C5HS5C).
This package contains a selected 300-case population, fixed partitions,
derived features and study-specific proxy annotations; it is not the full
10,000-record source dataset. The derived files are identified as study
transformations rather than original UCI annotations.

Azure PdM uses telemetry, errors, maintenance, machine metadata and failure
events from the [Microsoft predictive-maintenance example](https://github.com/microsoft/sqlworkshops/tree/master/SQLServerAndAzureMachineLearning/ML%20Services%20for%20SQL%20Server).
The example describes simulated data. Microsoft distributes that repository
under its [MIT license](https://github.com/microsoft/sqlworkshops/blob/master/LICENSE).
The fixed 300-case selection, pre-event summaries, study labels and proxy
annotations in this package are derived research inputs. Microsoft attribution
and the applicable source license notice must accompany a public release.

Generation-visible observation tables and evaluator-only proxy tables are
separate. File availability to an offline researcher is not permission for
test annotations to enter generation, fitting, optimization or selection.
