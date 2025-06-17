from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from datetime import datetime, timedelta
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, f1_score, recall_score
import joblib
import logging

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG(
    dag_id='pipeline_dag',
    default_args=default_args,
    description='Extract, train, evaluate and upload to S3 daily',
    schedule_interval='@daily',
    start_date=datetime(2025, 6, 18),
    catchup=False,
)

def extract(**kwargs):
    logging.info('Dataset extracted')
    url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data'
    df = pd.read_csv(url, header=None)
    df.to_csv('/tmp/data_wdbc.csv', index=False)        
    df['Cat'] = df[1].map({"M": 0, "B":1})
    logging.info('Data extracted and saved to /tmp/data_wdbc.csv')

def train_model(**kwargs):
    logging.info('Starting model training')
    df = pd.read_csv('/tmp/data_wdbc.csv')
    X = df.drop(columns=[0, 1, 'Cat'])
    y = df["Cat"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = LogisticRegression().fit(X_train, y_train)
    reg.score(X_test, y_test)   
    reg.save()  
    
    joblib.dump(reg, '/tmp/model.pkl')  
    logging.info('Model trained and saved to /tmp/model.pkl')   
    
def evaluate_model(**kwargs):   
    logging.info('Starting model evaluation')
    df = pd.read_csv('/tmp/data_wdbc.csv')
    df = pd.read_csv('/tmp/data_wdbc.csv')
    X = df.drop(columns=[0, 1, 'Cat'])
    y = df["Cat"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    reg = joblib.load('/tmp/model.pkl')
    y_pred = reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision  = precision_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Accuracy на тесте {accuracy:.2f}")
    print(f"recall на тесте: {recall:.2f}")
    print(f"precision на тесте: {precision:.2f}")
    print(f"F1 на тесте {f1:.2f}")

def upload_to_s3(**kwargs):
    logging.info('Starting upload to S3')
    s3 = S3Hook(aws_conn_id='aws_default')

    s3.load_file(
        filename='/tmp/data_wdbc.csv',
        key=f'data/wdbc_{datetime.now().strftime("%Y%m%d")}.csv',
        bucket_name='my-bucket',
        replace=True
    )

    s3.load_file(
        filename='/tmp/model.pkl',
        key=f'models/model_{datetime.now().strftime("%Y%m%d")}.pkl',
        bucket_name='my-bucket',
        replace=True
    )
    logging.info('Files uploaded to S3://my-bucket')


t1 = PythonOperator(
    task_id='extract',
    python_callable=extract,
    dag=dag,
)
t2 = PythonOperator(
    task_id='train_model',
    python_callable=train_model,
    dag=dag,
)
t3 = PythonOperator(
    task_id='evaluate_model',
    python_callable=evaluate_model,
    dag=dag,
)
t4 = PythonOperator(
    task_id='upload_to_s3',
    python_callable=upload_to_s3,
    dag=dag,
)

t1 >> t2 >> t3 >> t4