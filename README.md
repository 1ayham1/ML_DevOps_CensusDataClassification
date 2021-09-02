# Census Data Classification

**Project starter code:**
```
https://github.com/udacity/nd0821-c3-starter-code 
```
# Overview

Develop a classification model on publicly available [Census Bureau data](https://archive.ics.uci.edu/ml/datasets/census+income).  
Unit tests are created to monitor the model performance on various slices of the data, before the model is deployed using the FastAPI package and create API tests. Both the slice-validation and the API tests will be incorporated into a **continuous integration/continuous deployment** `CI/CD` framework using GitHub Actions. Both the dataset and the model are updated using `git` and `DVC`

# Environment Set up

* Download and install conda if you don’t have it already.
    * Use the supplied requirements file to create a new environment, or
    * conda create -n [envname] "python=3.8" scikit-learn dvc pandas numpy pytest jupyter jupyterlab fastapi uvicorn -c conda-forge
    * Install git either through conda (“conda install git”) or through your CLI, e.g. sudo apt-get git.

* AWS IAM:
get credintials after from IAM (AWS Acess Key ID, AWS Secret Access Key)
    * [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html)
    * [Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)
    * Add to github actions:

    `root repo --> settings --> Secrets -->New repo secret
    --> name{to be used from gethub actions} --> value {you wnat to be hidden. example AWS secret key}`

* Extract requirements from within a given project not the whole envirnoment.

* To creat Amazon S3 remote using DVC. [link](https://dvc.org/doc/command-reference/remote/add)

    * First create a bucket [link](https://awscli.amazonaws.com/v2/documentation/api/latest/reference/s3api/create-bucket.html):

    `aws s3api create-bucket --bucket ayham-bucket --region us-east-1`

    * create DVC remote:

    `dvc remote add -d ayham-remote s3://ayham-bucket/CensusProject` 

    * move to ./data and push to new remote

    `dvc push --remote ayham-remote`

    * [DVC_Basics](https://www.youtube.com/watch?v=kLKBcPonMYw)

    * [Heroku example](https://www.youtube.com/watch?v=QdhwYWwYfc0)

    `git push heroku main`

```
pip install pipreqs
pipreqs [project_folder]

pip freeze > requirements.txt [extract all depen. in an envriorment] 
```

* [Setup DCV Actions](https://github.com/iterative/setup-dvc)
## Repositories

* Setup GitHub Actions on your repo. You can use one of the pre-made GitHub Actions if at a minimum it runs pytest and flake8 on push and requires both to pass without error.
    * Make sure you set up the GitHub Action to have the same version of Python as you used in development.
* Set up a remote repository for dvc.

* push to heroku after connecting the repo: `git push heroku main`



