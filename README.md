# AWS 의 Sagemaker 를 이용한 머신 러닝 시스템 구축
##강의 홈 : https://github.com/mintbass/sm_edu

### 강의에서 다룰 내용
* AWS를 활용한 개발환경 이해하기
* AWS IAM을 이용한 루트키 및 액세스 키 관리화기
* AWS를 활용한 시스템 연동 이해하기
* Github 를 이용한 개발 협업 이해하기
* 로컬 컴퓨터를 이용한 머신 러닝 개발 환경 이해하기
* Anaconda 를 활용한 python 가상환경 이해하기
* Jupyter notebook 개발환경 이해하기
* Docker 를 이용한 개발환경 구축 이해하기
* AWS Sagemaker 이해하기
* AWS S3 활용법 이해하기
* 시계열 데이터 예측을 위한 DeepAR 알고리듬 이해하기
* AWS Sagemaker 를 이용한 데이터 훈련 및 테스트 실습
* pandas, boto3 라이브러리 활용


# Day #1

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

### 로컬 개발 환경 구축
* Python 3.7 설치 : https://www.python.org/downloads/release/python-376/
* Anaconda를 활용한 python 가상환경 설정하기 : http://bitly.kr/2DIFGaDe
    - Anaconda 설치 : https://www.anaconda.com/distribution/
    - 패키지 목록 보기
    ```
    conda list
    ```
* Jupyter 설치하기 (https://lsjsj92.tistory.com/531)

### Github
* Github 설치 (https://git-scm.com/downloads)
* 학습과정 Clone 하기 (>git clone https://github.com/mintbass/sm_edu.git)

### Docker를 이용한 개발환경 구축
* 도커 소개 (https://subicura.com/2017/01/19/docker-guide-for-beginners-1.html)
* Docker 설치(https://steemit.com/kr/@mystarlight/docker)
* pycharm 개발환경 이미지 만들기(https://tobelinuxer.tistory.com/25)
* Docker 를 활용한 jupyter 개발환경 구축하기

# Day #2
### AWS Sagemaker
* 아키텍처
* AWS Sagemaker 서비스 구조
* 데이터 저장소를 위한 S3 사용법
* 시작하기 (https://docs.aws.amazon.com/sagemaker/latest/dg/gs-console.html)

# Day #3
### Sagemaker 를 이용한 데이터 훈련 및 테스트 실습
* AWS Sagemaker Examples (https://github.com/awslabs/amazon-sagemaker-examples)
* Image classfying (https://aws.amazon.com/blogs/machine-learning/classify-your-own-images-using-amazon-sagemaker/)

# Day #4
### Sagemaker 를 이용한 데이터 훈련 및 테스트 실습
* pandas 를 이용한 csv 데이터 다루기
* boto3 를 이용한 S3 리소스 액세스
* DeepAR 을 이용한 시계열 데이터 예측 실습



