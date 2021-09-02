# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details

* Developed for personal purposes to explore CD/CI.
* SVM classifier with `rbf` kernel
* pretrained on [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income) and fine tuned using grid search.

## Intended Use

* Intended to be used for educational purposes; mainly learning deploying end-to-end machine learning models using various technologies. [**DVC, FastAPI, Heroku**]
* Intended for curious audiences.
* Not suitable for business applications. A lot of assumptions and simplifications were considered

## Training Data

* [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)
* Data split (80/20)

## Evaluation Data

* [Census Income Data Set](https://archive.ics.uci.edu/ml/datasets/census+income)
* Chosen as a basic proof-of-concept.

## Metrics

* Evaluation metrics include: [[definition source]](https://en.wikipedia.org/wiki/Precision_and_recall)
    * `Precision` or `positive predictive value`: the fraction of relevant instances among the retrieved instances
    * `Recall` or `sensitivity`: the fraction of relevant instances that were retrieved
    * `f1_score`: the harmonic mean of precision and recall

* The above metrics provide values for different errors that can be calculated from the confusion matrix for binary classification systems.

* Score Summary for Training Data:
```
--------------------------------------------------

    precision: 0.8278457196613358

    recall: 0.7327227310574521

    fbeta: 0.7773851590106007

==================================================
```

* Score Summary for Testing Data:

```
--------------------------------------------------

    precision: 0.6804511278195489

    recall: 0.4823451032644903

    fbeta: 0.5645224171539961

==================================================
```
* An extensive presentation for the above scores computed for given categorical variables when their values are held fixed is under `/src/mode/slice_output.txt`


## Ethical Considerations

* The app is intended for learning purposes and a simple model was utilized. No new information is to be inferred or business decisions to be made.

* It is ,however, worth commenting on **model fairness** when the above metrics were computed on a given categorical variable when its value is held fixed. Without loss of generality, considering category _education_ for example, the precision was 1.0 when _5th-6th_ feaure was held fixed and droped to 0.0 when _10th_ or _7th-8th_ was held fixed. Similar interesting trends can be observed for other categories `sex`, `race` ..etc. results can be found under `/src/mode/slice_output.txt` 
## Caveats and Recommendations

* Does not rank features by their importance. Moreover, some features are redundant. Some effort has been done to feature engineer the data, but a more advanced method could be used. This in turn can be reported as a source of errors.
* An ideal evaluation dataset would additionally include information about the work environment, work shift, and company size.