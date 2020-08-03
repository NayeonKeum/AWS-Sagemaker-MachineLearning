# AWS 의 Sagemaker 를 이용한 머신 러닝 시스템 구축
## 강의 홈 : https://github.com/mintbass/sm_edu

 
### 강의에서 다룰 내용
* AWS를 활용한 개발환경 이해하기
* AWS IAM을 이용한 루트키 및 액세스 키 관리화기
* AWS를 활용한 시스템 연동 이해하기
* Github 를 이용한 개발 협업 이해하기
* 로컬 컴퓨터를 이용한 머신 러닝 개발 환경 이해하기
* Anaconda 를 활용한 python 가상환경 이해하기
* Jupyter notebook 개발환경 이해하기
* AWS Sagemaker 이해하기
* AWS S3 활용법 이해하기
* 시계열 데이터 예측을 위한 DeepAR 알고리듬 이해하기
* AWS Sagemaker 를 이용한 데이터 훈련 및 테스트 실습
* pandas, boto3 라이브러리 활용


# Day #1

# Class #1

### AWS 계정 준비
* AWS 접속 및 크레딧 확인
    - 로그인 : https://aws.amazon.com/
    - 크레딧 : Account > Billing > Credits

* AWS IAM을 이용한 루트키 및 액세스 키 관리
    - Root Key 관리
    - My Security Credentials > Access Keys
    - NEVER EVER create root access keys
    - IAM 을 이용한 계정 및 액세스키 관리

* AWS CLI 를 이용한 AWS 리소스 액세스
    - Putty Windows 64bit Download : https://the.earth.li/~sgtatham/putty/latest/w64/putty-64bit-0.70-installer.msi
    - AWS CLI Download : https://s3.amazonaws.com/aws-cli/AWSCLI64.msi
    - Amazon CLI(Command Line Interface) 소개 및 실습 : https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/awscli-install-windows.html
    - 빠른 시작: https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/cli-chap-getting-started.html
    - CLI 를 이용한 S3 사용 실습: https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/using-s3-commands.html

* AWS 서비스 둘러보기
    - EC2 (Elastic Compute Cloud)
    - S3 (Simple Storage Service)
    - EKS (Elastic Kubernetes Service)
    - Lambda
    - Batch
    - Sagemaker
    - Forecast
    - SQS (Simple Queue Service)
    - SNS (Simple Notification Service)
    - SES (Simple Email Service)
    - RDS
    - DynamoDB
    - ElasticCache
    - Redshift
    - CloudWatch
    - VPC
    - CloudFront
    - Route53
    - API Gateway
    - Cognito
    - IoT Core


### Github
* Github 설치 : https://git-scm.com/downloads
* 학습과정 Clone 하기
 ```
git clone https://github.com/mintbass/sm_edu.git
 ```

### 로컬 개발 환경 구축
* Python 3.7 설치 : https://www.python.org/downloads/release/python-376/
* Anaconda를 활용한 python 가상환경 설정하기 : http://bitly.kr/2DIFGaDe
    - Anaconda 설치 : https://www.anaconda.com/distribution/
    - 패키지 목록 보기
    ```
    conda list
    ```
    - 가상환경 생성
    ```
    conda create -n smedu pandas tensorflow
    ```
    - 가상환경 시작
    ```
    conda activate smedu
    ```
    - 가상환경 종료
    ```
    conda deactivate
    ```
    - 가상환경 내보내기 (export)
    ```
    pwd
    /Users/mint/git/sm_edu
    conda env export > smedu.yaml
    ```

* Jupyter 설치하기 (https://lsjsj92.tistory.com/531)
    - 가상환경에 jupyter notebook 설치
    ```
    conda install jupyter notebook
    ```
    - jupyter notebook에서 python 패키지를 관리할 수 있도록 해주는 nb_conda 설치
    ```
    conda install nb_conda
    ```
    - jupyter notebook 시작하기
    ```
    (smedu) mint@Marcuss-MacBook-Pro sm_edu % jupyter notebook
    [I 15:01:43.393 NotebookApp] [nb_conda_kernels] enabled, 2 kernels found
    [W 15:01:44.295 NotebookApp] WARNING: The notebook server is listening on all IP addresses and not using encryption. This is not recommended.
    [I 15:01:44.452 NotebookApp] [nb_conda] enabled
    [I 15:01:44.453 NotebookApp] Serving notebooks from local directory: /Users/mint/git/sm_edu
    [I 15:01:44.453 NotebookApp] The Jupyter Notebook is running at:
    [I 15:01:44.453 NotebookApp] http://Marcuss-MacBook-Pro.local:8888/?token=1c427019f02e1e5a13665e157679ecd4bbf8c40ef0e2df3f
    [I 15:01:44.453 NotebookApp]  or http://127.0.0.1:8888/?token=1c427019f02e1e5a13665e157679ecd4bbf8c40ef0e2df3f
    [I 15:01:44.453 NotebookApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
    [C 15:01:44.476 NotebookApp]

        To access the notebook, open this file in a browser:
            file:///Users/mint/Library/Jupyter/runtime/nbserver-13592-open.html
        Or copy and paste one of these URLs:
            http://Marcuss-MacBook-Pro.local:8888/?token=1c427019f02e1e5a13665e157679ecd4bbf8c40ef0e2df3f
         or http://127.0.0.1:8888/?token=1c427019f02e1e5a13665e157679ecd4bbf8c40ef0e2df3f
    ```
  
    
# Class #2

### Python 개발 툴
* PyCharm : https://www.jetbrains.com/ko-kr/pycharm/download/#section=windows
    - Interpreter 설정
* VSCode (Visual Studio Code) : https://code.visualstudio.com/download
    - Interpreter 설정
    
### Amazon CLI(Command Line Interface)
* Download : AWS CLI Download - https://s3.amazonaws.com/aws-cli/AWSCLI64.msi
* Amazon CLI(Command Line Interface) 소개 및 실습 : https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/awscli-install-windows.html
* 빠른 시작 : https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/cli-chap-getting-started.html
* 프로파일 등록
```
aws configure
AWS Access Key ID [None]: AKIAI44QH8DHBEXAMPLE
AWS Secret Access Key [None]: je7MtGbClwBF/2Zp9Utk/h3yCo8nvbEXAMPLEKEY
Default region name [None]: ap-northeast-2
Default output format [None]: 
```
* CLI 를 이용한 S3 사용 실습 : https://docs.aws.amazon.com/ko_kr/cli/latest/userguide/using-s3-commands.html

### Python Library 맛보기
* pandas
    - Dataframe 다루기 : https://3months.tistory.com/292
    - csv 데이터 다루기
* boto3 를 이용한 S3 리소스 액세스
    - conda install boto3
    - S3에 버킷 만들기 (smedu)
    - IAM을 이용한 액세스 키 생성하기
    - aws configure 를 이용해서 AWS 프로파일 생성
    - boto3_S3_access.py 실행

# Day #2

# Class #1
### AWS Sagemaker
* AWS Sagemaker 서비스 구조
    - ![Service](./res/sm-service.png)
* 아키텍처
    - ![Archiecture](./res/architecture.png)
        
* 시작하기 : https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html

### Sagemaker 를 이용한 데이터 훈련 및 테스트 실습
* DeepAR 을 이용한 시계열 데이터 예측 on Sagemaker Jupyter Notebook 실습

# Class #2
* DeepAR 을 이용한 시계열 데이터 예측 on 로컬 컴퓨터 실습

* How DeepAR works : https://docs.aws.amazon.com/ko_kr/sagemaker/latest/dg/deepar_how-it-works.html 
* AWS Sagemaker Examples : https://github.com/awslabs/amazon-sagemaker-examples
* Image classfying (https://aws.amazon.com/blogs/machine-learning/classify-your-own-images-using-amazon-sagemaker/)

